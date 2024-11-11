import json
import logging
import math
from collections import defaultdict
from itertools import groupby
from operator import itemgetter
from pathlib import Path
from typing import Union

import numpy as np
import spacy
from sklearn.metrics import roc_curve
from spacy import Language
from spacy.tokens import Doc, Token
from torch import tensor
from torch.nn import functional
from transformers import DebertaV2TokenizerFast, DebertaV2Model

from diffcheck.setup.system_check import SystemResources


class TokenizerError(Exception):
    """Custom exception for tokenizer errors"""
    pass


class ContextAwareTokenizer:
    word_dictionary: dict
    word_dictionary_id: int
    logger: logging.Logger
    unknown_token_id: int
    nlp: Language

    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.word_dictionary = {}
        self.word_dictionary_id = 0

        # Initialize system check
        sys_check = SystemResources()
        recommended_models = sys_check.get_recommended_models()

        try:
            # Get model paths
            cache_dir = Path.home() / '.cache' / 'tokenizer_models'
            
            # Initialize models based on system capabilities
            self.deberta_model = recommended_models['hf']
            self.spacy_model = recommended_models['spacy']
            
            # Load DeBERTa from cache
            model_dir = cache_dir / self.deberta_model.replace('/', '-')
            self.tokenizer = DebertaV2TokenizerFast.from_pretrained(
                str(model_dir),
                local_files_only=True,
                trust_remote_code=True,
                use_fast=False
            )
            # Try loading with auto-detection of format
            try:
                self.model = DebertaV2Model.from_pretrained(
                    str(model_dir),
                    local_files_only=True,
                    trust_remote_code=False
                )
                self.logger.info("Successfully loaded model with auto-detected format")
            except Exception as e:
                self.logger.error(f"Failed to load model: {e}")
                # Try explicit format loading
                safetensors_path = model_dir / "model.safetensors"
                pytorch_path = model_dir / "pytorch_model.bin"
                
                if safetensors_path.exists():
                    model_size = safetensors_path.stat().st_size / (1024**3)  # Size in GB
                    self.logger.info(f"Found safetensors model: {model_size:.2f}GB")
                    self.model = DebertaV2Model.from_pretrained(
                        str(model_dir),
                        local_files_only=True,
                        use_safetensors=True,
                        trust_remote_code=False
                    )
                    self.logger.info(f"Successfully loaded model using safetensors")
                elif pytorch_path.exists():
                    model_size = pytorch_path.stat().st_size / (1024**3)  # Size in GB
                    self.logger.info(f"Found PyTorch model: {model_size:.2f}GB")
                    self.model = DebertaV2Model.from_pretrained(
                        str(model_dir),
                        local_files_only=True,
                        use_safetensors=False,
                        trust_remote_code=False
                    )
                    self.logger.info(f"Successfully loaded model using PyTorch format")
                else:
                    raise FileNotFoundError("No valid model file found (tried safetensors and PyTorch formats)")
            
            # Load spaCy model
            self.nlp = spacy.load(self.spacy_model)

            self.logger.info(f"Initialized with DeBERTa={self.deberta_model}, spaCy={self.spacy_model}")
        except Exception as e:
            self.logger.error(f"Error initializing tokenizer: {e}")
            raise

        self.unknown_token_id = self.tokenizer.convert_tokens_to_ids(self.tokenizer.unk_token)

    def _merge_wordpieces(self, tokens):
        """Merge DeBERTa wordpieces into whole words."""
        if not isinstance(tokens, list):
            raise TokenizerError("Tokens must be a list")

        merged = []
        current_word = []

        for token in tokens:
            if not isinstance(token, str):
                raise TokenizerError(f"Invalid token type: {type(token)}")

            # DeBERTa uses '▁' for new words
            if token.startswith('▁'):
                if current_word:
                    merged.append(''.join(current_word))
                    current_word = []
                current_word.append(token[1:])
            else:
                current_word.append(token)

        if current_word:
            merged.append(''.join(current_word))

        return merged

    def get_word_id(self, word_id):
        if word_id != self.unknown_token_id:
            return word_id

        word_id = self.word_dictionary.get(word_id, None)
        if word_id is None:
            self.word_dictionary_id -= 1
            self.word_dictionary[word_id] = self.word_dictionary_id
            word_id = self.word_dictionary_id

        return word_id

    @staticmethod
    def cosine_similarity(tensor1: tensor, tensor2: tensor):
        return functional.cosine_similarity(tensor1.unsqueeze(0), tensor2.unsqueeze(0)).item()

    def calculate_similarity_scores(self, hidden_states):
        similarities = []
        for i in range(len(hidden_states) - 1):
            similarity = self.cosine_similarity(hidden_states[i], hidden_states[i + 1])
            similarities.append(similarity)
        return similarities

    @staticmethod
    def map_deberta_to_spacy(tokens, offsets) -> list[list[Token]]:
        token_mapping: list[list[Token]] = [[] for _ in range(len(offsets))]
        for j, (deberta_start, deberta_end) in enumerate(offsets):
            for token in tokens:
                spacy_start = token.idx
                spacy_end = token.idx + len(token.text)
                if spacy_start < deberta_end and spacy_end > deberta_start:
                    token_mapping[j].append(token)
                    break
        return token_mapping

    @staticmethod
    def map_spacy_to_deberta(spacy_tokens, deberta_offsets) -> list[list]:
        token_mapping: list[list] = [[] for i in range(len(spacy_tokens))]
        for j, deberta_offset in enumerate(deberta_offsets):
            for i, token in enumerate(spacy_tokens):
                spacy_start = token.idx
                spacy_end = token.idx + len(token.text)
                if spacy_start < deberta_offset[1] and spacy_end > deberta_offset[0]:
                    token_mapping[i].append(deberta_offset)
                    break
        return token_mapping

    @staticmethod
    def generate_relationships(token_mapping, similarities):
        """Generate relationships between tokens using both POS and DEP tags."""
        relationships = []
        for i in range(len(similarities)):
            current_token = token_mapping.get(i)
            next_token = token_mapping.get(i + 1)
            if current_token and next_token:
                # Use tuple of (POS, DEP) for each token
                current_info = (current_token.pos_, current_token.dep_)
                next_info = (next_token.pos_, next_token.dep_)
                relationships.append(((current_info, next_info), similarities[i]))
        return relationships

    @staticmethod
    def preprocess_tokens(deberta_offsets, similarities, token_mapping, spacy_tokens):
        """Split DeBERTa tokens based on spaCy token boundaries and adjust similarities."""
        new_offsets = []
        new_similarities = []
        new_token_mapping = {}
        next_token_id = 0

        # For each DeBERTa token
        for i, (start, end) in enumerate(deberta_offsets):
            # Find overlapping spaCy tokens
            overlapping_spacy = []
            for spacy_token in spacy_tokens:
                token_start = spacy_token.idx
                token_end = spacy_token.idx + len(spacy_token.text)

                # Check if DeBERTa token overlaps with this spaCy token
                if start < token_end and end > token_start:
                    overlapping_spacy.append((spacy_token, token_start, token_end))

            # If token needs to be split (overlaps with multiple spaCy tokens)
            if len(overlapping_spacy) > 1:
                # Create subtokens
                for spacy_token, token_start, token_end in overlapping_spacy:
                    subtoken_start = max(start, token_start)
                    subtoken_end = min(end, token_end)
                    new_offsets.append((subtoken_start, subtoken_end))
                    new_token_mapping[next_token_id] = spacy_token

                    # If not last subtoken, add similarity of 1 since it came from same DeBERTa token
                    if len(new_offsets) > 1:
                        new_similarities.append(1.0)
                    next_token_id += 1
            else:
                # Token doesn't need splitting
                new_offsets.append((start, end))
                new_token_mapping[next_token_id] = overlapping_spacy[0][0]
                if len(new_offsets) > 1:
                    # Use original similarity if this wasn't first token
                    if i > 0:
                        new_similarities.append(similarities[i - 1])
                next_token_id += 1

        return new_offsets, new_similarities, new_token_mapping

    @staticmethod
    def calculate_thresholds(relationships, tokens):
        """Calculate thresholds based on natural clusters in similarity scores."""
        pos_pairs = defaultdict(list)
        pos_tokens = defaultdict(list)

        # Group similarities by POS pairs
        for index, relationship in enumerate(relationships):
            (pos1, pos2), similarity = relationship
            pos_pairs[(pos1, pos2)].append(similarity)
            pos_tokens[(pos1, pos2)].append(tokens[index])

        for key, value in pos_pairs.items():
            tokens = pos_tokens[key]
            for token, similarity in zip(tokens, value):
                print(f"{token} {key}: {round(similarity, 3)}")

        thresholds = {}
        for pos_pair, similarities in pos_pairs.items():
            # Need multiple samples to compute meaningful threshold
            if len(similarities) < 2:
                continue

            similarities = np.array(similarities)

            # Use KMeans to find natural clusters
            from sklearn.cluster import KMeans
            kmeans = KMeans(n_clusters=2, random_state=42)
            clusters = kmeans.fit_predict(similarities.reshape(-1, 1))

            # Get cluster centers
            centers = kmeans.cluster_centers_.flatten()

            # Use midpoint between clusters as threshold
            threshold = (max(centers) + min(centers)) / 2
            thresholds[pos_pair] = threshold

        return thresholds

    def tokenize_with_spacy(self, text) -> Doc:
        try:
            spacy_tokens: Doc = self.nlp(text)
            return spacy_tokens
            # merged_tokens = []
            # for token in doc:
            #     if token.pos_ == 'PART':
            #         merged_tokens[-1] = (merged_tokens[-1][0], token.idx + len(token.text))
            #     else:
            #         merged_tokens.append((token.idx, token.idx + len(token.text)))
            # for token in doc:
            #     print(f'Token: {token.text}, Head: {token.head.text}, Dependency: {token.dep_}, Pos: {token.pos_}, Offset: ({token.idx}, {token.idx + len(token.text)})')
        except Exception as e:
            raise TokenizerError(f"spaCy processing failed: {str(e)}")

    def tokenize_with_deberta(self, text) -> tuple[list[tuple[int, int]], tensor]:
        try:
            wordpieces = self.tokenizer(text, return_offsets_mapping=True, return_tensors='pt')
            offsets = wordpieces['offset_mapping'][0].tolist()[1:-1]
            del wordpieces['offset_mapping']
            outputs = self.model(**wordpieces, output_hidden_states=True)
            hidden_states = outputs.hidden_states[-1]
            hidden_states = hidden_states.squeeze(0)
            hidden_states = hidden_states[1:-1]

            return offsets, hidden_states
        except Exception as e:
            raise TokenizerError(f"DeBERTa tokenization failed: {str(e)}")

    @staticmethod
    def calculate_offsets(offsets, similarities, token_mapping, thresholds) -> list[tuple[int, int]]:
        """Calculate final token offsets using DeBERTa similarity scores and spaCy PART tags."""
        final_offsets = []
        i = 0

        while i < len(offsets):
            start = offsets[i][0]
            end = offsets[i][1]

            # Check if we should merge based on similarity
            if i < len(similarities):
                current_token = token_mapping.get(i)
                next_token = token_mapping.get(i + 1)

                if current_token and next_token:
                    pos_pair = (current_token.pos_, next_token.pos_)
                    threshold = thresholds.get(pos_pair, np.mean(list(thresholds.values())))

                    # First filter: DeBERTa similarity scores
                    if similarities[i] > threshold:
                        # Second filter: Check if next token is a PART
                        if next_token.pos_ == 'PART':
                            end = offsets[i + 1][1]
                            i += 1

            final_offsets.append((start, end))
            i += 1

        return final_offsets

    @staticmethod
    def determine_compound_pos(sequence_tokens, next_token=None):
        """
        Determine if sequence is a compound and if so, what POS it should have.
        Returns None if not a compound.
        """
        # Get the dependency pattern and POS of content words
        pos = [t.pos_ for t in sequence_tokens if t.pos_ != 'PUNCT']
        texts = [t.text.lower().rstrip('.') for t in sequence_tokens if t.pos_ != 'PUNCT']

        # Check if it's a month-month range
        months = {'jan', 'feb', 'mar', 'apr', 'may', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov', 'dec',
                  'january', 'february', 'march', 'april', 'june', 'july', 'august', 'september',
                  'october', 'november', 'december'}

        # If exactly 2 content words and both are months, it's a range
        if len(texts) == 2 and all(t in months for t in texts):
            return None

        # If it's a month-number combination, split it
        if len(texts) == 2 and texts[0] in months and pos[-1] == 'NUM':
            return None

        # Handle brand names and proper nouns that aren't dates
        if any(p == 'PROPN' for p in pos):
            return 'PROPN'

        # Split pure number sequences
        if len(set(pos)) == 1 and pos[0] == 'NUM':
            return None

        # If starts with ADV, likely adjectival compound
        if sequence_tokens[0].pos_ == 'ADV':
            return 'ADJ'

        # Check if it's likely a compound noun by looking at dependency structure
        last_token = sequence_tokens[-1]
        if any(t.dep_ == 'compound' for t in sequence_tokens):
            return 'NOUN'

        # Multi-part compounds (more than one hyphen) are usually nouns
        if len(sequence_tokens) > 3:
            return 'NOUN'

        # If modifying a noun and not already identified as a noun compound
        if last_token.dep_ == 'amod' and next_token and next_token.pos_ == 'NOUN':
            return 'ADJ'

        return None

    @staticmethod
    def neighbors(left, right):
        if isinstance(left, Token):
            if isinstance(right, Token):
                return left.idx + len(left.text) == right.idx
            else:
                return left.idx + len(left.text) == right[0]
        elif isinstance(right, Token):
            return left[1] == right.idx
        else:
            return left[1] == right[0]

    @staticmethod
    def join_tokens(left, right, deberta_tokens, token_type='JOINED'):
        return (
        left[0], right.idx + len(right.text), left[2] + ''.join(deberta_token[2] for deberta_token in deberta_tokens),
        token_type)

    @staticmethod
    def join_tokens_sequence(spacy_tokens, deberta_tokens, token_type):
        """Join a sequence of tokens together, preserving necessary information."""
        start_pos = spacy_tokens[0].idx
        end_pos = spacy_tokens[-1].idx + len(spacy_tokens[-1].text)
        text = ''.join(deberta_token[2] for deberta_token in deberta_tokens)
        return (start_pos, end_pos, text, token_type)

    def tokenize(self, text):
        """Context-aware tokenization using DeBERTa and spaCy's analysis."""
        if not isinstance(text, str):
            raise TokenizerError(f"Input must be string, not {type(text)}")

        if not text:
            return []

        # print('Input: ', text)
        spacy_tokens = self.tokenize_with_spacy(text)
        deberta_output = self.tokenizer(text, return_offsets_mapping=True)
        # deberta_ids = deberta_output['input_ids'][1:-1]
        deberta_offsets = deberta_output['offset_mapping'][1:-1]
        deberta_tokens = self.tokenizer.tokenize(text)
        deberta_offsets = [
            (start + 1 if deberta_tokens[i].startswith('▁') and end - start > 1 else start, end)
            for i, (start, end) in enumerate(deberta_offsets)
        ]
        deberta_tokens = [
            (*deberta_offsets[i], token)
            for i, token in enumerate(deberta_tokens)
        ]
        spacy_to_deberta = ContextAwareTokenizer.map_spacy_to_deberta(spacy_tokens, deberta_tokens)

        tokens = []
        i = 0
        while i < len(spacy_tokens):
            token = spacy_tokens[i]
            deberta_tokens = spacy_to_deberta[i]
            token_type = token.pos_
            print(deberta_tokens)
            print(f'{token.text} {token.pos_} {token.dep_ if token.has_dep() else None} {len(deberta_tokens)}')

            if token.pos_ in {'PRON'} and len(deberta_tokens) > 1:
                print('RULE 1: SPLIT')
                for deberta_token in deberta_tokens:
                    new_token = self.tokenize_with_spacy(deberta_token[2])[-1]
                    tokens.append((*deberta_token, new_token.pos_))
                i += 1
                continue

            if token.pos_ == 'SPACE':
                print('RULE 2: IGNORE')
                i += 1
                continue

            if token.text in {'_', '▁'}:
                print('RULE 3: JOIN')
                if len(tokens) > 0 and tokens[-1][3] not in {'PUNCT', 'SPACE'} and ContextAwareTokenizer.neighbors(tokens[-1], token):
                    if tokens[-1][3] == 'UNDERSCORE':
                        tokens[-1] = ContextAwareTokenizer.join_tokens(tokens[-1], token, deberta_tokens, 'UNDERSCORE')
                    else:
                        tokens[-1] = ContextAwareTokenizer.join_tokens(tokens[-1], token, deberta_tokens, 'NOUN')
                else:
                    tokens.append((token.idx, token.idx + len(token.text), ''.join(deberta_token[2] for deberta_token in deberta_tokens), 'UNDERSCORE'))

                if i < len(spacy_tokens) - 1 and spacy_tokens[i + 1].pos_ not in {'PUNCT', 'SPACE'} and ContextAwareTokenizer.neighbors(tokens[-1], spacy_tokens[i + 1]):
                    tokens[-1] = ContextAwareTokenizer.join_tokens(tokens[-1], spacy_tokens[i + 1], spacy_to_deberta[i + 1], 'NOUN')
                    i += 1

                i += 1
                continue

            if token.pos_ in {'AUX', 'PART'} and len(deberta_tokens) > 1:
                tokens[-1] = ContextAwareTokenizer.join_tokens(tokens[-1], token, deberta_tokens)
                i += 1
                continue

            # if len(tokens) > 0 and i < len(spacy_tokens) - 1 and token.pos_ == 'PUNCT' and tokens[-1][3] == 'PROPN' and spacy_tokens[i + 1].pos_ == 'PROPN' and ContextAwareTokenizer.neighbors(tokens[-1], token) and ContextAwareTokenizer.neighbors(token, spacy_tokens[i + 1]):
            #     print('RULE 4: JOIN')
            #     tokens[-1] = ContextAwareTokenizer.join_tokens(tokens[-1], token, deberta_tokens, 'PROPN')
            #     i += 1
            #     continue

            if len(tokens) > 0 and tokens[-1][3] not in {'PUNCT', 'SPACE'} and token.pos_ not in {'PUNCT', 'SPACE'} and ContextAwareTokenizer.neighbors(tokens[-1], token):
                print('RULE 5: JOIN')
                token_type = token.pos_ if token.pos_ == tokens[-1][3] else 'JOINED'
                tokens[-1] = ContextAwareTokenizer.join_tokens(tokens[-1], token, deberta_tokens, token_type)
                i += 1
                continue

            if (i < len(spacy_tokens) - 2 and
                    spacy_tokens[i + 1].text == '-' and
                    spacy_tokens[i + 1].pos_ == 'PUNCT'):

                # Get the full hyphenated sequence
                start_idx = i
                curr_idx = i
                sequence_tokens = []

                while (curr_idx < len(spacy_tokens) - 2 and
                       spacy_tokens[curr_idx + 1].text == '-' and
                       spacy_tokens[curr_idx + 1].pos_ == 'PUNCT'):
                    sequence_tokens.extend([
                        spacy_tokens[curr_idx],
                        spacy_tokens[curr_idx + 1],
                        spacy_tokens[curr_idx + 2]
                    ])
                    curr_idx += 2
                    if curr_idx + 2 >= len(spacy_tokens):
                        break

                if sequence_tokens:
                    # Get next token for context if available
                    next_token = spacy_tokens[curr_idx + 1] if curr_idx + 1 < len(spacy_tokens) else None

                    # Check if it's a compound and get its POS
                    compound_pos = ContextAwareTokenizer.determine_compound_pos(sequence_tokens, next_token)

                    if compound_pos:  # It's a compound, join it
                        print('RULE 6: JOIN')
                        all_deberta = []
                        for idx in range(start_idx, curr_idx + 1):
                            all_deberta.extend(spacy_to_deberta[idx])

                        tokens.append(ContextAwareTokenizer.join_tokens_sequence(sequence_tokens, all_deberta, compound_pos))
                        i = curr_idx + 1
                        continue

            print('APPENDING NEW TOKEN')
            tokens.append((token.idx, token.idx + len(token.text), ''.join(deberta_token[2] for deberta_token in deberta_tokens), token_type))

            i += 1

        tokens = [
            token
            for token in tokens
            if token[3] not in {'UNDERSCORE', 'PUNCT'}
        ]

        print(tokens)

        token_text = [
            text[token[0]:token[1]].lower()
            for token in tokens
        ]
        token_ids = self.tokenizer.convert_tokens_to_ids(token_text)
        unknown_tokens = [
            token_text[i]
            for i, token_id in enumerate(token_ids)
            if token_id == self.unknown_token_id
        ]
        self.tokenizer.add_tokens(unknown_tokens)
        token_ids = self.tokenizer.convert_tokens_to_ids(token_text)
        tokens = [
            (token[0], token[1], token_ids[i])
            for i, token in enumerate(tokens)
        ]
        print(tokens)

        return tokens




# deberta_tokens, deberta_token_tensors = self.tokenize_with_deberta(text)
# similarities = self.calculate_similarity_scores(deberta_token_tensors)
# token_mapping = ContextAwareTokenizer.map_deberta_to_spacy(spacy_tokens, deberta_tokens)
# relationships = ContextAwareTokenizer.generate_relationships(token_mapping, similarities)
# thresholds = ContextAwareTokenizer.calculate_thresholds(relationships, token_mapping)
# offsets = ContextAwareTokenizer.calculate_offsets(deberta_tokens, similarities, token_mapping, thresholds)
