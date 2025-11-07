"""
Enhanced Content Similarity Metrics

This module provides advanced semantic similarity metrics for comparing
predicted and actual post content, including:
- BERTScore: Token-level contextual embeddings similarity
- Sentence Embeddings: Document-level semantic similarity
- Named Entity Recognition: Entity matching
- Lexical Metrics: Traditional word/character overlap

Designed to dramatically improve content similarity detection beyond
simple word overlap (e.g., "Crooked Hillary" vs "Corrupt Clinton" should
score high despite different words).
"""

import os
import re
from typing import Dict, List, Tuple, Optional
import numpy as np
from loguru import logger

# Lazy imports to avoid loading heavy models at import time
_bert_score = None
_sentence_transformer = None
_spacy_nlp = None


def _get_bert_score():
    """Lazy load BERTScore."""
    global _bert_score
    if _bert_score is None:
        try:
            from bert_score import score as bert_score_fn
            _bert_score = bert_score_fn
            logger.info("BERTScore loaded successfully")
        except ImportError as e:
            logger.warning(f"BERTScore not available: {e}")
            _bert_score = False
    return _bert_score if _bert_score is not False else None


def _get_sentence_transformer():
    """Lazy load Sentence Transformer model."""
    global _sentence_transformer
    if _sentence_transformer is None:
        try:
            from sentence_transformers import SentenceTransformer
            # Use lightweight model optimized for speed
            _sentence_transformer = SentenceTransformer('all-MiniLM-L6-v2')
            logger.info("Sentence Transformer loaded successfully (all-MiniLM-L6-v2)")
        except ImportError as e:
            logger.warning(f"Sentence Transformers not available: {e}")
            _sentence_transformer = False
        except Exception as e:
            logger.error(f"Error loading Sentence Transformer: {e}")
            _sentence_transformer = False
    return _sentence_transformer if _sentence_transformer is not False else None


def _get_spacy_nlp():
    """Lazy load spaCy NER model."""
    global _spacy_nlp
    if _spacy_nlp is None:
        try:
            import spacy
            # Try to load English model
            try:
                _spacy_nlp = spacy.load("en_core_web_sm")
                logger.info("spaCy NER loaded successfully (en_core_web_sm)")
            except OSError:
                # Model not downloaded, try to download it
                logger.warning("spaCy model not found, attempting to download...")
                os.system("python -m spacy download en_core_web_sm")
                _spacy_nlp = spacy.load("en_core_web_sm")
                logger.info("spaCy NER downloaded and loaded successfully")
        except ImportError as e:
            logger.warning(f"spaCy not available: {e}")
            _spacy_nlp = False
        except Exception as e:
            logger.error(f"Error loading spaCy: {e}")
            _spacy_nlp = False
    return _spacy_nlp if _spacy_nlp is not False else None


class SimilarityMetrics:
    """
    Comprehensive content similarity calculator with multiple metrics.
    """

    def __init__(
        self,
        weights: Optional[Dict[str, float]] = None,
        use_bertscore: bool = True,
        use_sentence_embeddings: bool = True,
        use_entity_matching: bool = True,
        bertscore_model: str = "microsoft/deberta-xlarge-mnli"
    ):
        """
        Initialize similarity metrics calculator.

        Args:
            weights: Dict with metric weights (bertscore, sentence, entity, lexical)
                    Default: {bertscore: 0.4, sentence: 0.3, entity: 0.2, lexical: 0.1}
            use_bertscore: Whether to use BERTScore (slower but more accurate)
            use_sentence_embeddings: Whether to use sentence embeddings
            use_entity_matching: Whether to use named entity matching
            bertscore_model: Model to use for BERTScore (default: deberta-xlarge-mnli)
        """
        self.use_bertscore = use_bertscore
        self.use_sentence_embeddings = use_sentence_embeddings
        self.use_entity_matching = use_entity_matching
        self.bertscore_model = bertscore_model

        # Default weights (can be overridden in config)
        self.weights = weights or {
            'bertscore': 0.40,
            'sentence': 0.30,
            'entity': 0.20,
            'lexical': 0.10
        }

        # Normalize weights to sum to 1.0
        total_weight = sum(self.weights.values())
        if total_weight > 0:
            self.weights = {k: v / total_weight for k, v in self.weights.items()}

        logger.info(f"SimilarityMetrics initialized with weights: {self.weights}")

    def calculate_lexical_similarity(
        self,
        predicted: str,
        actual: str
    ) -> Dict[str, float]:
        """
        Calculate traditional lexical similarity metrics.

        Args:
            predicted: Predicted content
            actual: Actual content

        Returns:
            Dict with lexical similarity scores
        """
        # Length similarity
        pred_len = len(predicted)
        actual_len = len(actual)
        length_similarity = 1 - abs(pred_len - actual_len) / max(pred_len, actual_len) if max(pred_len, actual_len) > 0 else 0

        # Word overlap (Jaccard similarity)
        pred_words = set(predicted.lower().split())
        actual_words = set(actual.lower().split())

        if pred_words or actual_words:
            intersection = len(pred_words & actual_words)
            union = len(pred_words | actual_words)
            word_overlap = intersection / union if union > 0 else 0
        else:
            word_overlap = 0

        # Character overlap
        pred_chars = set(predicted.lower())
        actual_chars = set(actual.lower())
        char_overlap = len(pred_chars & actual_chars) / len(pred_chars | actual_chars) if (pred_chars | actual_chars) else 0

        # Composite lexical score (average)
        lexical_score = (length_similarity + word_overlap + char_overlap) / 3

        return {
            'length_similarity': length_similarity,
            'word_overlap': word_overlap,
            'character_overlap': char_overlap,
            'lexical_score': lexical_score
        }

    def calculate_bertscore(
        self,
        predicted: str,
        actual: str
    ) -> Dict[str, float]:
        """
        Calculate BERTScore for semantic similarity.

        BERTScore compares contextual embeddings at token level,
        capturing semantic similarity even with different words.

        Args:
            predicted: Predicted content
            actual: Actual content

        Returns:
            Dict with BERTScore metrics (precision, recall, f1)
        """
        bert_score_fn = _get_bert_score()

        if not bert_score_fn or not self.use_bertscore:
            logger.debug("BERTScore not available, using fallback")
            # Fallback to lexical score if BERTScore unavailable
            lexical = self.calculate_lexical_similarity(predicted, actual)
            return {
                'bertscore_precision': lexical['word_overlap'],
                'bertscore_recall': lexical['word_overlap'],
                'bertscore_f1': lexical['word_overlap']
            }

        try:
            # Calculate BERTScore
            # Returns (P, R, F1) tensors
            P, R, F1 = bert_score_fn(
                [predicted],
                [actual],
                lang='en',
                model_type=self.bertscore_model,
                verbose=False,
                device='cpu'  # Use CPU for compatibility (GPU optional)
            )

            return {
                'bertscore_precision': float(P[0]),
                'bertscore_recall': float(R[0]),
                'bertscore_f1': float(F1[0])
            }

        except Exception as e:
            logger.error(f"BERTScore calculation failed: {e}")
            # Fallback to lexical
            lexical = self.calculate_lexical_similarity(predicted, actual)
            return {
                'bertscore_precision': lexical['word_overlap'],
                'bertscore_recall': lexical['word_overlap'],
                'bertscore_f1': lexical['word_overlap']
            }

    def calculate_sentence_embedding_similarity(
        self,
        predicted: str,
        actual: str
    ) -> Dict[str, float]:
        """
        Calculate sentence embedding cosine similarity.

        Uses Sentence Transformers to get semantic document-level embeddings.

        Args:
            predicted: Predicted content
            actual: Actual content

        Returns:
            Dict with sentence embedding similarity score
        """
        model = _get_sentence_transformer()

        if not model or not self.use_sentence_embeddings:
            logger.debug("Sentence Transformer not available, using fallback")
            # Fallback to lexical
            lexical = self.calculate_lexical_similarity(predicted, actual)
            return {
                'sentence_embedding_similarity': lexical['word_overlap']
            }

        try:
            # Encode both texts
            pred_embedding = model.encode(predicted, convert_to_tensor=False)
            actual_embedding = model.encode(actual, convert_to_tensor=False)

            # Calculate cosine similarity
            cosine_sim = np.dot(pred_embedding, actual_embedding) / (
                np.linalg.norm(pred_embedding) * np.linalg.norm(actual_embedding)
            )

            # Normalize to [0, 1] range (cosine is [-1, 1])
            normalized_sim = (float(cosine_sim) + 1) / 2

            return {
                'sentence_embedding_similarity': normalized_sim
            }

        except Exception as e:
            logger.error(f"Sentence embedding calculation failed: {e}")
            # Fallback to lexical
            lexical = self.calculate_lexical_similarity(predicted, actual)
            return {
                'sentence_embedding_similarity': lexical['word_overlap']
            }

    def extract_entities(self, text: str) -> Dict[str, List[str]]:
        """
        Extract named entities from text.

        Args:
            text: Input text

        Returns:
            Dict mapping entity types to lists of entities
        """
        nlp = _get_spacy_nlp()

        if not nlp or not self.use_entity_matching:
            # Simple fallback: extract capitalized words
            words = text.split()
            capitalized = [w for w in words if w and w[0].isupper() and len(w) > 1]
            return {
                'PERSON': capitalized,
                'ORG': [],
                'GPE': [],
                'EVENT': []
            }

        try:
            doc = nlp(text)
            entities = {
                'PERSON': [],
                'ORG': [],
                'GPE': [],  # Geo-political entity (countries, cities)
                'EVENT': []
            }

            for ent in doc.ents:
                if ent.label_ in entities:
                    entities[ent.label_].append(ent.text)

            return entities

        except Exception as e:
            logger.error(f"Entity extraction failed: {e}")
            return {'PERSON': [], 'ORG': [], 'GPE': [], 'EVENT': []}

    def calculate_entity_similarity(
        self,
        predicted: str,
        actual: str
    ) -> Dict[str, float]:
        """
        Calculate named entity matching similarity.

        Compares entities in predicted vs actual text.
        Useful for Trump posts mentioning specific people/organizations.

        Args:
            predicted: Predicted content
            actual: Actual content

        Returns:
            Dict with entity matching scores
        """
        pred_entities = self.extract_entities(predicted)
        actual_entities = self.extract_entities(actual)

        # Calculate Jaccard similarity for each entity type
        entity_similarities = {}
        for entity_type in ['PERSON', 'ORG', 'GPE', 'EVENT']:
            pred_set = set(e.lower() for e in pred_entities.get(entity_type, []))
            actual_set = set(e.lower() for e in actual_entities.get(entity_type, []))

            if pred_set or actual_set:
                intersection = len(pred_set & actual_set)
                union = len(pred_set | actual_set)
                similarity = intersection / union if union > 0 else 0
            else:
                similarity = 0

            entity_similarities[f'entity_{entity_type.lower()}_similarity'] = similarity

        # Overall entity score (average)
        if entity_similarities:
            entity_score = np.mean(list(entity_similarities.values()))
        else:
            entity_score = 0

        entity_similarities['entity_score'] = entity_score

        return entity_similarities

    def calculate_all_metrics(
        self,
        predicted: str,
        actual: str
    ) -> Dict[str, float]:
        """
        Calculate all similarity metrics and return comprehensive results.

        Args:
            predicted: Predicted content
            actual: Actual content

        Returns:
            Dict with all similarity metrics and composite score
        """
        logger.debug(f"Calculating similarity between:\n  Predicted: {predicted[:100]}...\n  Actual: {actual[:100]}...")

        # Initialize results
        results = {}

        # 1. Lexical metrics (always calculated, fast)
        lexical = self.calculate_lexical_similarity(predicted, actual)
        results.update(lexical)

        # 2. BERTScore (if enabled)
        if self.use_bertscore:
            bertscore = self.calculate_bertscore(predicted, actual)
            results.update(bertscore)
        else:
            results['bertscore_f1'] = lexical['lexical_score']

        # 3. Sentence embeddings (if enabled)
        if self.use_sentence_embeddings:
            sentence_sim = self.calculate_sentence_embedding_similarity(predicted, actual)
            results.update(sentence_sim)
        else:
            results['sentence_embedding_similarity'] = lexical['word_overlap']

        # 4. Entity matching (if enabled)
        if self.use_entity_matching:
            entity_sim = self.calculate_entity_similarity(predicted, actual)
            results.update(entity_sim)
        else:
            results['entity_score'] = lexical['word_overlap']

        # 5. Calculate weighted composite score
        composite = (
            self.weights['bertscore'] * results.get('bertscore_f1', 0) +
            self.weights['sentence'] * results.get('sentence_embedding_similarity', 0) +
            self.weights['entity'] * results.get('entity_score', 0) +
            self.weights['lexical'] * results.get('lexical_score', 0)
        )

        results['composite_similarity'] = composite

        logger.debug(f"Similarity results: Composite={composite:.3f}, BERTScore={results.get('bertscore_f1', 0):.3f}, "
                    f"Sentence={results.get('sentence_embedding_similarity', 0):.3f}, "
                    f"Entity={results.get('entity_score', 0):.3f}, Lexical={results.get('lexical_score', 0):.3f}")

        return results


def test_similarity_metrics():
    """Test the similarity metrics with example pairs."""
    print("\n" + "="*80)
    print("TESTING SIMILARITY METRICS")
    print("="*80 + "\n")

    # Initialize calculator
    calc = SimilarityMetrics()

    # Test cases
    test_cases = [
        {
            'name': 'Identical text',
            'predicted': 'FAKE NEWS! The media is lying about me again!',
            'actual': 'FAKE NEWS! The media is lying about me again!'
        },
        {
            'name': 'Semantic similarity (different words, same meaning)',
            'predicted': 'Crooked Hillary is corrupt and dishonest!',
            'actual': 'Corrupt Clinton is crooked and lying!'
        },
        {
            'name': 'Different entities, similar structure',
            'predicted': 'Biden is destroying our economy!',
            'actual': 'Obama destroyed our jobs!'
        },
        {
            'name': 'Completely different',
            'predicted': 'Just had a great rally in Texas!',
            'actual': 'The stock market is hitting record highs!'
        },
        {
            'name': 'Empty/short texts',
            'predicted': 'MAGA!',
            'actual': 'Make America Great Again!'
        }
    ]

    for i, test in enumerate(test_cases, 1):
        print(f"\nTest Case {i}: {test['name']}")
        print(f"  Predicted: {test['predicted']}")
        print(f"  Actual:    {test['actual']}")

        results = calc.calculate_all_metrics(test['predicted'], test['actual'])

        print(f"\n  Results:")
        print(f"    Composite Score:    {results['composite_similarity']:.3f}")
        print(f"    BERTScore F1:       {results.get('bertscore_f1', 0):.3f}")
        print(f"    Sentence Embedding: {results.get('sentence_embedding_similarity', 0):.3f}")
        print(f"    Entity Score:       {results.get('entity_score', 0):.3f}")
        print(f"    Lexical Score:      {results.get('lexical_score', 0):.3f}")
        print(f"    Word Overlap:       {results.get('word_overlap', 0):.3f}")
        print("-" * 80)

    print("\n" + "="*80)
    print("TESTING COMPLETE")
    print("="*80 + "\n")


if __name__ == "__main__":
    # Run tests
    test_similarity_metrics()
