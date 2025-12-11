"""
Content Generation Model using Claude API with few-shot prompting.
Simple, effective MVP that can be replaced with fine-tuned models later.
"""

import os
import sys
from datetime import datetime, timezone, timedelta
from loguru import logger
import yaml
from anthropic import Anthropic

from src.validation.similarity_metrics import SimilarityMetrics

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data.database import get_session, Post
from context.context_gatherer import RealTimeContextGatherer


class ContentGenerator:
    """
    Generates post content using Claude API with few-shot prompting.
    
    Design: Modular interface that can be swapped with fine-tuned models later.
    """
    
    def __init__(self, config_path="config/config.yaml", api_key=None):
        self.config_path = config_path
        self.config = self._load_config(config_path)
        
        # Initialize Claude client
        self.api_key = api_key or os.getenv('ANTHROPIC_API_KEY')
        if not self.api_key:
            logger.error("ANTHROPIC_API_KEY not set!")
            self.client = None
        else:
            self.client = Anthropic(api_key=self.api_key)
        
        self.example_posts = []
        self.model_version = "claude_few_shot_v1"
        self.similarity_metrics = self._init_similarity_metrics()
        claude_cfg = self.config.get('claude_api', {})
        self.max_calls_per_hour = claude_cfg.get('max_calls_per_hour')
        self.call_history = []
    
    def _load_config(self, config_path):
        """Load configuration"""
        try:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            return config['content_model']
        except Exception as e:
            logger.warning(f"Could not load config: {e}. Using defaults.")
            return {
                'type': 'claude_api',
                'claude_api': {
                    'model': 'claude-sonnet-4-5-20250929',
                    'max_tokens': 280,
                    'temperature': 0.8,
                    'num_examples': 10
                }
            }

    def _load_similarity_config(self):
        """Load similarity metric configuration from validation settings."""
        try:
            with open(self.config_path, 'r') as f:
                full_config = yaml.safe_load(f)
            return full_config.get('validation', {}).get('similarity_metrics', {})
        except Exception as exc:
            logger.warning(f"Could not load similarity config: {exc}")
            return {}

    def _init_similarity_metrics(self):
        """Initialise SimilarityMetrics helper if dependencies are available."""
        similarity_cfg = self._load_similarity_config()
        try:
            return SimilarityMetrics(
                weights=similarity_cfg.get('weights'),
                use_bertscore=similarity_cfg.get('use_bertscore', True),
                use_sentence_embeddings=similarity_cfg.get('use_sentence_embeddings', True),
                use_entity_matching=similarity_cfg.get('use_entity_matching', True),
                bertscore_model=similarity_cfg.get('bertscore_model', "microsoft/deberta-xlarge-mnli")
            )
        except Exception as exc:
            logger.warning(f"Similarity metrics unavailable: {exc}")
            return None

    def _within_call_budget(self) -> bool:
        """Return True if it's safe to call the Anthropic API."""
        if not self.max_calls_per_hour:
            return True
        cutoff = datetime.now(timezone.utc) - timedelta(hours=1)
        self.call_history = [ts for ts in self.call_history if ts >= cutoff]
        return len(self.call_history) < self.max_calls_per_hour

    def _record_call(self):
        if self.max_calls_per_hour:
            self.call_history.append(datetime.now(timezone.utc))

    def _fallback_content(self, predicted_time):
        """Return a low-confidence fallback when rate limit is hit."""
        if not self.example_posts:
            self.load_example_posts()
        sample_text = "Big updates coming soon. Stay tuned!"
        if self.example_posts:
            sample_text = self.example_posts[0]['content']

        return {
            'content': f"[Rate-limit fallback] {sample_text}",
            'model_version': self.model_version,
            'confidence': 0.2,
            'generated_at': datetime.now(),
            'context_used': False,
            'similarity_metrics': {},
            'rate_limit_hit': True,
            'predicted_time_hint': predicted_time.isoformat() if predicted_time else None
        }
    
    def load_example_posts(self, num_examples=None, context=None):
        """Load example posts for few-shot prompting using simple retrieval."""
        if num_examples is None:
            num_examples = self.config.get('claude_api', {}).get('num_examples', 10)
        
        logger.info(f"Loading {num_examples} example posts...")
        
        session = get_session()
        
        # Get diverse sample across time
        posts = session.query(Post)\
            .filter(Post.content != None)\
            .filter(Post.content != '')\
            .order_by(Post.created_at.desc())\
            .limit(num_examples * 3)\
            .all()
        
        session.close()
        
        if not posts:
            logger.warning("No posts found in database!")
            return []
        
        trending_terms = set()
        if context and context.get('trending_keywords'):
            trending_terms = {term.lower() for term in context.get('trending_keywords', [])}
        max_engagement = max(
            (p.favourites_count or 0) + (p.reblogs_count or 0) + (p.replies_count or 0)
            for p in posts
        ) or 1
        now = datetime.now(timezone.utc)

        scored_posts = []
        for post in posts:
            engagement = (post.favourites_count or 0) + (post.reblogs_count or 0) + (post.replies_count or 0)
            engagement_score = engagement / max_engagement

            keyword_score = 0.0
            if trending_terms:
                content_lower = (post.content or "").lower()
                matches = sum(1 for term in trending_terms if term in content_lower)
                keyword_score = matches / max(len(trending_terms), 1)

            created_at = post.created_at or now
            if created_at.tzinfo is None:
                created_at = created_at.replace(tzinfo=timezone.utc)
            recency_hours = max((now - created_at).total_seconds() / 3600, 0.1)
            recency_score = 1 / (1 + recency_hours / 12)

            score = 0.6 * engagement_score + 0.25 * keyword_score + 0.15 * recency_score
            scored_posts.append((score, post))

        scored_posts.sort(key=lambda item: item[0], reverse=True)
        top_posts = [post for _, post in scored_posts[:num_examples]]
        
        self.example_posts = [
            {
                'content': p.content,
                'created_at': p.created_at,
                'engagement': (p.favourites_count or 0) + (p.reblogs_count or 0),
                'score': score
            }
            for score, p in scored_posts[:num_examples]
        ]
        
        logger.info(f"Loaded {len(self.example_posts)} example posts")
        return self.example_posts
    
    def format_examples(self):
        """Format example posts for prompt"""
        if not self.example_posts:
            self.load_example_posts()
        
        examples_text = []
        for i, post in enumerate(self.example_posts, 1):
            examples_text.append(f"Example {i}:\n{post['content']}\n")
        
        return "\n".join(examples_text)
    
    def generate(self, context=None, predicted_time=None):
        """
        Generate post content using Claude API.
        
        Args:
            context: dict with recent news, trends, etc.
            predicted_time: datetime when post is predicted
            
        Returns:
            dict with generated content and metadata
        """
        if self.client is None:
            logger.error("Claude client not initialized!")
            return None

        if not self._within_call_budget():
            logger.warning("Anthropic call budget reached; reusing cached content instead of calling API")
            return self._fallback_content(predicted_time)
        
        # Prepare context information
        time_context = ""
        if predicted_time:
            time_context = f"Time context: {predicted_time.strftime('%A, %B %d, %Y at %I:%M %p')}"

        # Enhanced context from RealTimeContextGatherer
        news_context = ""
        trending_context = ""
        market_context = ""

        if context:
            # News headlines
            if 'news_summary' in context and context['news_summary']:
                news_context = f"Recent news: {context['news_summary']}"
            elif 'recent_news' in context:
                # Fallback to old format
                news_context = f"Recent news: {context['recent_news']}"

            # Trending topics
            if 'trending_keywords' in context and context['trending_keywords']:
                top_trends = ', '.join(context['trending_keywords'][:5])
                trending_context = f"Trending topics: {top_trends}"

            # Market data
            if 'market_sentiment' in context:
                sentiment = context['market_sentiment']
                sp_change = context.get('sp500_change_pct', 0)
                market_context = f"Market: {sentiment} (S&P {sp_change:+.1f}%)"

        # Build few-shot prompt with enhanced context
        if not self.example_posts:
            self.load_example_posts(context=context)
        examples = self.format_examples()

        context_section = "\n".join(filter(None, [time_context, news_context, trending_context, market_context]))

        prompt = f"""You are generating a social media post in the style of Donald Trump's Truth Social posts.

Below are real examples of his posting style:

{examples}

Current context:
{context_section}

Now generate a new post in the same style. Keep it under 280 characters. Make it authentic to the style - direct, capitalized words for emphasis, and characteristic phrasing. The post should be relevant to the current context. Do not include quotes around the post.

Generated post:"""

        try:
            # Call Claude API
            claude_config = self.config.get('claude_api', {})
            
            message = self.client.messages.create(
                model=claude_config.get('model', 'claude-sonnet-4-5-20250929'),
                max_tokens=claude_config.get('max_tokens', 280),
                temperature=claude_config.get('temperature', 0.8),
                messages=[
                    {"role": "user", "content": prompt}
                ]
            )
            
            generated_content = message.content[0].text.strip()
            self._record_call()
            
            # Remove quotes if present
            if generated_content.startswith('"') and generated_content.endswith('"'):
                generated_content = generated_content[1:-1]

            content_confidence = 0.5
            similarity_details = {}
            if self.similarity_metrics and self.example_posts:
                try:
                    similarity_scores = []
                    for example in self.example_posts:
                        metrics = self.similarity_metrics.calculate_all_metrics(
                            generated_content,
                            example['content']
                        )
                        similarity_scores.append(metrics)
                    if similarity_scores:
                        best_metrics = max(similarity_scores, key=lambda m: m['composite_similarity'])
                        content_confidence = best_metrics['composite_similarity']
                        similarity_details = best_metrics
                except Exception as similarity_exc:
                    logger.warning(f"Failed to score generated content: {similarity_exc}")
            
            result = {
                'content': generated_content,
                'model_version': self.model_version,
                'confidence': content_confidence,
                'generated_at': datetime.now(),
                'context_used': bool(context),
                'similarity_metrics': similarity_details
            }
            
            logger.info(f"Generated content: {generated_content[:100]}...")
            
            return result
            
        except Exception as e:
            logger.error(f"Failed to generate content: {e}")
            return None
    
    def generate_multiple(self, n=3, context=None, predicted_time=None):
        """
        Generate multiple candidate posts.
        
        Args:
            n: Number of candidates to generate
            context: Context for generation
            predicted_time: When post is predicted
            
        Returns:
            list of generated posts
        """
        logger.info(f"Generating {n} candidate posts...")
        
        candidates = []
        for i in range(n):
            result = self.generate(context=context, predicted_time=predicted_time)
            if result:
                candidates.append(result)
        
        logger.info(f"Generated {len(candidates)} candidates")
        return candidates
    
    def evaluate_similarity(self, generated_content, actual_content):
        """
        Evaluate similarity between generated and actual content.
        
        For MVP, we'll use simple metrics. Later can add BERTScore.
        
        Args:
            generated_content: Generated post text
            actual_content: Actual post text
            
        Returns:
            dict with similarity metrics
        """
        # Simple word overlap (placeholder for BERTScore)
        gen_words = set(generated_content.lower().split())
        actual_words = set(actual_content.lower().split())
        
        if not gen_words or not actual_words:
            return {'overlap': 0.0, 'length_similarity': 0.0}
        
        overlap = len(gen_words & actual_words) / len(gen_words | actual_words)
        
        length_similarity = 1.0 - abs(len(generated_content) - len(actual_content)) / max(len(generated_content), len(actual_content))
        
        metrics = {
            'word_overlap': overlap,
            'length_similarity': length_similarity,
            'composite_score': (overlap + length_similarity) / 2
        }
        
        return metrics


# Alias for backward compatibility
ContextGatherer = RealTimeContextGatherer


def main():
    """Test the content generator"""
    logger.info("Testing Content Generator...")
    
    # Initialize generator
    generator = ContentGenerator()
    
    # Load example posts
    generator.load_example_posts()
    
    if generator.client:
        # Generate content
        context_gatherer = ContextGatherer()
        context = context_gatherer.get_full_context()
        
        result = generator.generate(
            context=context,
            predicted_time=datetime.now()
        )
        
        if result:
            print("\n" + "="*50)
            print("GENERATED POST")
            print("="*50)
            print(result['content'])
            print("="*50)
            print(f"Model: {result['model_version']}")
            print(f"Confidence: {result['confidence']:.2%}")
            print("="*50)
    else:
        logger.error("Claude API key not set. Cannot generate content.")
        logger.info("Set ANTHROPIC_API_KEY environment variable to test generation.")


if __name__ == "__main__":
    main()
