import logging
from typing import Union

from spacy import Language
from spacy.language import PipeCallable
from spacy.tokens import Token

from tokenizer.deberta_tokenizer import DebertaTokenizer


# don't combine dashes, just leave them separate
# use duckling to get ranges do not combine them
# combine remaining dashes because if they are not a range then they should be combined
# how to handle 7-Eleven? Will duckling grab it as a range?


SpanToken = tuple[int, int, Union[int, list[int]]]


class TokenizerError(Exception):
    """Custom exception for tokenizer errors"""
    pass


class ContextAwareTokenizer:
    logger: logging.Logger
    spacy_tokenizer: Language
    deberta_tokenizer: DebertaTokenizer
    # transformer: PipeCallable

    def __init__(self, deberta_tokenizer: DebertaTokenizer, spacy_tokenizer: Language):
        self.logger = logging.getLogger(__name__)

        self.deberta_tokenizer = deberta_tokenizer
        self.spacy_tokenizer = spacy_tokenizer
        # self.transformer = self.spacy_tokenizer.get_pipe("transformer")

    def to_text(self, word_id: int) -> str:
        """Convert tokens back to text"""
        return self.deberta_tokenizer.tokenizer.convert_ids_to_tokens(word_id)

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
    def are_neighbors(left, right):
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
    def join_tokens(left, right, deberta_tokens, pos='JOINED', dep=None):
        return left[0], right.idx + len(right.text), left[2] + ''.join(deberta_token[2] for deberta_token in deberta_tokens), pos, dep

    @staticmethod
    def join_tokens_sequence(spacy_tokens, deberta_tokens, pos, dep) -> tuple[int, int, str, str, Union[str, None]]:
        """Join a sequence of tokens together, preserving necessary information."""
        start_pos = spacy_tokens[0].idx
        end_pos = spacy_tokens[-1].idx + len(spacy_tokens[-1].text)
        text = ''.join(deberta_token[2] for deberta_token in deberta_tokens)
        return start_pos, end_pos, text, pos, dep

    # def analyze_compound(self, doc, token) -> bool:
    #     """
    #     Determine if compound is compositional using transformer embeddings
    #     Returns True if compositional, False if lexical
    #     """
    #     if token.dep_ != "compound":
    #         return False
    #
    #     # Get transformer embeddings
    #     outputs = self.transformer.predict([doc])
    #     hidden_states = outputs.last_hidden_layer_states[0]
    #
    #     # Convert to torch tensors and flatten
    #     compound_embedding = torch.tensor(hidden_states[token.i].dataXd).unsqueeze(0)
    #     head_embedding = torch.tensor(hidden_states[token.head.i].dataXd).unsqueeze(0)
    #
    #     # Ensure both embeddings have the same size along dimension 1
    #     min_size = min(compound_embedding.size(1), head_embedding.size(1))
    #     compound_embedding = compound_embedding[:, :min_size]
    #     head_embedding = head_embedding[:, :min_size]
    #
    #     # Calculate cosine similarity
    #     similarity = torch.nn.functional.cosine_similarity(
    #         compound_embedding,
    #         head_embedding,
    #         dim=1
    #     ).mean().item()
    #
    #     # print(f'{similarity} > 0.494')
    #     return similarity > 0.494  # Return True if compositional

    def tokenize(self, text: str) -> list[SpanToken]:
        """Context-aware tokenization using DeBERTa and spaCy's analysis."""
        if not isinstance(text, str):
            raise TokenizerError(f"Input must be string, not {type(text)}")

        if not text:
            return []

        # print('Input: ', text)
        spacy_tokens = self.spacy_tokenizer(text)
        deberta_output = self.deberta_tokenizer.tokenizer(text, return_offsets_mapping=True)
        deberta_offsets = deberta_output['offset_mapping'][1:-1]
        deberta_tokens = self.deberta_tokenizer.tokenizer.tokenize(text)
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
            pos = token.pos_
            dep = token.dep_
            tag = token.tag_
            head = token.head
            # print(deberta_tokens)
            # print(f'{token.text} {pos} {dep} {tag} {head} {len(deberta_tokens)}')

            # if token.dep_ == "compound":
            #     # is_compositional = self.analyze_compound(spacy_tokens, token)
            #     # if not is_compositional:
            #     #     print('COMPOUND RULE')
            #     #     # Lexical compound - join with head
            #     head = token.head
            #     compound_tokens = [token, head]
            #     all_deberta = []
            #     for t in compound_tokens:
            #         all_deberta.extend(spacy_to_deberta[t.i])
            #     tokens.append(self.join_tokens_sequence(
            #         compound_tokens,
            #         all_deberta,
            #         'NOUN',
            #         None
            #     ))
            #     i = head.i + 1  # Skip past the head token
            #     continue

            if token.pos_ in {'PRON'} and len(deberta_tokens) > 1:
                # print('RULE 1: SPLIT')
                for deberta_token in deberta_tokens:
                    new_token = self.spacy_tokenizer(deberta_token[2])[-1]
                    tokens.append((*deberta_token, new_token.pos_, new_token.dep_))
                i += 1
                continue

            if token.pos_ == 'SPACE':
                # print('RULE 2: IGNORE')
                i += 1
                continue

            if token.text in {'_', '▁'}:
                # print('RULE 3: JOIN')
                if len(tokens) > 0 and tokens[-1][3] not in {'PUNCT', 'SPACE'} and ContextAwareTokenizer.are_neighbors(tokens[-1], token):
                    if tokens[-1][3] == 'UNDERSCORE':
                        tokens[-1] = ContextAwareTokenizer.join_tokens(tokens[-1], token, deberta_tokens, 'UNDERSCORE', None)
                    else:
                        tokens[-1] = ContextAwareTokenizer.join_tokens(tokens[-1], token, deberta_tokens, 'NOUN', None)
                else:
                    tokens.append((token.idx, token.idx + len(token.text), ''.join(deberta_token[2] for deberta_token in deberta_tokens), 'UNDERSCORE', None))

                if i < len(spacy_tokens) - 1 and spacy_tokens[i + 1].pos_ not in {'PUNCT', 'SPACE'} and ContextAwareTokenizer.are_neighbors(tokens[-1], spacy_tokens[i + 1]):
                    tokens[-1] = ContextAwareTokenizer.join_tokens(tokens[-1], spacy_tokens[i + 1], spacy_to_deberta[i + 1], 'NOUN', None)
                    i += 1

                i += 1
                continue

            if token.pos_ in {'AUX', 'PART'} and len(deberta_tokens) > 1:
                # print('AUX RULE')
                tokens[-1] = ContextAwareTokenizer.join_tokens(tokens[-1], token, deberta_tokens)
                i += 1
                continue

            # if len(tokens) > 0 and i < len(spacy_tokens) - 1 and token.pos_ == 'PUNCT' and tokens[-1][3] == 'PROPN' and spacy_tokens[i + 1].pos_ == 'PROPN' and ContextAwareTokenizer.neighbors(tokens[-1], token) and ContextAwareTokenizer.neighbors(token, spacy_tokens[i + 1]):
            #     print('RULE 4: JOIN')
            #     tokens[-1] = ContextAwareTokenizer.join_tokens(tokens[-1], token, deberta_tokens, 'PROPN')
            #     i += 1
            #     continue

            if len(tokens) > 0 and tokens[-1][3] not in {'PUNCT', 'SPACE'} and token.pos_ not in {'PUNCT', 'SPACE'} and ContextAwareTokenizer.are_neighbors(tokens[-1], token):
                # print('RULE 5: JOIN')
                token_type = token.pos_ if token.pos_ == tokens[-1][3] else 'JOINED'
                tokens[-1] = ContextAwareTokenizer.join_tokens(tokens[-1], token, deberta_tokens, token_type, None)
                i += 1
                continue

            # print('APPENDING NEW TOKEN')
            tokens.append((token.idx, token.idx + len(token.text), ''.join(deberta_token[2] for deberta_token in deberta_tokens), pos, dep))

            i += 1

        tokens = [
            token
            for token in tokens
            if token[3] not in {'UNDERSCORE', 'PUNCT', 'DET'} and (token[3] != 'CCONJ' or token[4] != 'cc')
        ]

        # print(tokens)

        token_text = [
            text[token[0]:token[1]].lower()
            for token in tokens
        ]
        token_ids = self.deberta_tokenizer.tokenizer.convert_tokens_to_ids(token_text)
        unknown_tokens = [
            token_text[i]
            for i, token_id in enumerate(token_ids)
            if token_id == self.deberta_tokenizer.unknown_token_id
        ]
        self.deberta_tokenizer.tokenizer.add_tokens(unknown_tokens)
        token_ids = self.deberta_tokenizer.tokenizer.convert_tokens_to_ids(token_text)
        tokens = [
            (token[0], token[1], token_ids[i])
            for i, token in enumerate(tokens)
        ]
        # print(tokens)

        return tokens




# deberta_tokens, deberta_token_tensors = self.tokenize_with_deberta(text)
# similarities = self.calculate_similarity_scores(deberta_token_tensors)
# token_mapping = ContextAwareTokenizer.map_deberta_to_spacy(spacy_tokens, deberta_tokens)
# relationships = ContextAwareTokenizer.generate_relationships(token_mapping, similarities)
# thresholds = ContextAwareTokenizer.calculate_thresholds(relationships, token_mapping)
# offsets = ContextAwareTokenizer.calculate_offsets(deberta_tokens, similarities, token_mapping, thresholds)
