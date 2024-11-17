import numpy as np
from spacy.tokens import Token


class CompoundClassifier:
    def __init__(self, nlp):
        self.nlp = nlp

    def _get_contextual_similarity(self, tokens: list[Token]) -> float:
        """Compare contextual embeddings of parts"""
        # Get the transformer component directly
        trf = self.nlp.get_pipe("transformer")

        # Create a doc to get transformer context
        text = " ".join(t.text for t in tokens)
        doc = self.nlp(text)

        # Get transformer output tensors directly
        outputs = trf.predict([doc])
        # Last hidden states typically at index 0
        embeddings = outputs[0]

        # Compare embeddings of tokens (excluding punctuation)
        token_indices = [i for i, t in enumerate(doc) if t.pos_ != 'PUNCT']
        token_embeddings = [embeddings[i] for i in token_indices]

        # Calculate similarities
        similarities = []
        for i, emb1 in enumerate(token_embeddings):
            for j, emb2 in enumerate(token_embeddings[i+1:], i+1):
                cos_sim = np.dot(emb1, emb2) / (
                    np.linalg.norm(emb1) * np.linalg.norm(emb2)
                )
                similarities.append(cos_sim)

        return float(np.mean(similarities)) if similarities else 0.5

    def classify_compound(self, tokens: list[Token]) -> tuple[str, float, dict]:
        similarity = self._get_contextual_similarity(tokens)

        scores = {
            'contextual_similarity': similarity
        }

        if similarity > 0.5:
            return 'compositional', 0.6, scores
        return 'lexical', 0.6, scores
