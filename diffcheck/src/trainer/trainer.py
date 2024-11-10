import json
from collections import defaultdict
from pathlib import Path
import numpy as np
from datasets import load_dataset
from tqdm import tqdm
import spacy
from transformers import DebertaV2TokenizerFast, DebertaV2Model
import torch
from torch.nn import functional as F
import nltk
from nltk.corpus import brown


class ThresholdTrainer:
    def __init__(self, cache_dir='./cache'):
        # Initialize models
        self.nlp = spacy.load('en_core_web_sm')
        self.tokenizer = DebertaV2TokenizerFast.from_pretrained('microsoft/deberta-v2-xlarge')
        self.model = DebertaV2Model.from_pretrained('microsoft/deberta-v2-xlarge')
        self.model.eval()

        # Store similarities with corpus information
        self.similarities = defaultdict(lambda: defaultdict(list))
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def get_deberta_embeddings(self, text):
        """Get DeBERTa embeddings for a text."""
        with torch.no_grad():
            inputs = self.tokenizer(text, return_tensors='pt', return_offsets_mapping=True)
            offsets = inputs['offset_mapping'][0].tolist()[1:-1]
            del inputs['offset_mapping']

            outputs = self.model(**inputs, output_hidden_states=True)
            hidden_states = outputs.hidden_states[-1]
            hidden_states = hidden_states.squeeze(0)[1:-1]

            return offsets, hidden_states

    def calculate_similarity(self, emb1, emb2):
        """Calculate cosine similarity between embeddings."""
        return F.cosine_similarity(emb1.unsqueeze(0), emb2.unsqueeze(0)).item()

    def process_text(self, text, corpus_name):
        """Process a single text, collecting similarity scores."""
        if not text.strip():
            return

        # Get spaCy analysis
        doc = self.nlp(text)

        # Get DeBERTa embeddings
        try:
            offsets, embeddings = self.get_deberta_embeddings(text)
        except Exception as e:
            print(f"Error processing text: {e}")
            return

        # Calculate similarities between adjacent tokens
        for i in range(len(embeddings) - 1):
            emb1, emb2 = embeddings[i], embeddings[i + 1]
            similarity = self.calculate_similarity(emb1, emb2)

            # Map tokens to spaCy tokens
            for token_idx, spacy_token in enumerate(doc):
                if token_idx == len(doc) - 1:
                    continue

                next_token = doc[token_idx + 1]
                relationship = (
                    (spacy_token.pos_, spacy_token.dep_),
                    (next_token.pos_, next_token.dep_)
                )

                # Store similarity if tokens overlap with DeBERTa offsets
                start1, end1 = offsets[i]
                start2, end2 = offsets[i + 1]

                if (start1 < spacy_token.idx + len(spacy_token.text) and
                        end1 > spacy_token.idx and
                        start2 < next_token.idx + len(next_token.text) and
                        end2 > next_token.idx):
                    self.similarities[str(relationship)][corpus_name].append(similarity)

    def train_on_wikitext(self, batch_size=32):
        """Train on WikiText dataset."""
        dataset = load_dataset("wikitext", "wikitext-103-v1", split="train")

        for i in tqdm(range(0, len(dataset), batch_size), desc="Processing WikiText"):
            batch = dataset[i:i + batch_size]
            for text in batch['text']:
                self.process_text(text, 'wikitext')

    def train_on_brown(self):
        """Train on Brown corpus."""
        nltk.download('brown')
        for fileid in tqdm(brown.fileids(), desc="Processing Brown Corpus"):
            text = ' '.join(brown.words(fileid))
            self.process_text(text, 'brown')

    def train_on_books3(self, sample_size=10000):
        """Train on Books3 sample."""
        dataset = load_dataset("the_pile", split="train")
        books_data = [
                         text for text in dataset
                         if text['meta']['pile_set_name'] == 'Books3'
                     ][:sample_size]

        for item in tqdm(books_data, desc="Processing Books3"):
            self.process_text(item['text'], 'books3')

    def train(self):
        """Train on all available corpora."""
        # Train on WikiText
        print("Training on WikiText...")
        self.train_on_wikitext()
        self.save_intermediate_results('wikitext')

        # Train on Brown
        print("Training on Brown Corpus...")
        self.train_on_brown()
        self.save_intermediate_results('brown')

        # Train on Books3
        print("Training on Books3...")
        self.train_on_books3()
        self.save_intermediate_results('books3')

    def save_intermediate_results(self, corpus_name):
        """Save intermediate results for a corpus."""
        save_path = self.cache_dir / f'similarities_{corpus_name}.json'
        corpus_similarities = {
            rel: scores[corpus_name]
            for rel, scores in self.similarities.items()
            if corpus_name in scores
        }
        with open(save_path, 'w') as f:
            json.dump(corpus_similarities, f)

    def calculate_corpus_stats(self, corpus_name, scores):
        """Calculate statistics for a corpus."""
        if not scores:
            return None

        scores_array = np.array(scores)
        return {
            'mean': float(np.mean(scores_array)),
            'std': float(np.std(scores_array)),
            'min': float(np.min(scores_array)),
            'max': float(np.max(scores_array)),
            'count': len(scores),
            'quartiles': [
                float(np.percentile(scores_array, 25)),
                float(np.percentile(scores_array, 50)),
                float(np.percentile(scores_array, 75))
            ]
        }

    def save_final_results(self, output_dir='threshold_data'):
        """Calculate and save thresholds and stats."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Calculate thresholds and stats for each relationship
        thresholds = {}
        stats = {}

        for relationship, corpus_scores in self.similarities.items():
            # Combine scores from all corpora
            all_scores = []
            corpus_stats = {}

            for corpus_name, scores in corpus_scores.items():
                corpus_stats[corpus_name] = self.calculate_corpus_stats(corpus_name, scores)
                all_scores.extend(scores)

            if all_scores:
                scores_array = np.array(all_scores)
                mean = np.mean(scores_array)
                std = np.std(scores_array)

                # Calculate combined stats
                corpus_stats['combined'] = self.calculate_corpus_stats('combined', all_scores)
                stats[relationship] = corpus_stats

                # Calculate threshold
                thresholds[relationship] = float(mean - std)

        # Save results
        with open(output_dir / 'thresholds.json', 'w') as f:
            json.dump(thresholds, f, indent=2)

        with open(output_dir / 'stats.json', 'w') as f:
            json.dump(stats, f, indent=2)


if __name__ == '__main__':
    trainer = ThresholdTrainer()
    trainer.train()
    trainer.save_final_results()