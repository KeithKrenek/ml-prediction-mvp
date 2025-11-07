"""
Content Generation Model using Claude API with few-shot prompting.
Simple, effective MVP that can be replaced with fine-tuned models later.
"""

import os
import sys
import random
from datetime import datetime
from loguru import logger
import yaml
from anthropic import Anthropic

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
    
    def load_example_posts(self, num_examples=None):
        """Load example posts for few-shot prompting"""
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
        
        # Sample diverse posts (different lengths, engagement)
        sampled = random.sample(posts, min(num_examples, len(posts)))
        
        self.example_posts = [
            {
                'content': p.content,
                'created_at': p.created_at,
                'engagement': p.favourites_count + p.reblogs_count
            }
            for p in sampled
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
            
            # Remove quotes if present
            if generated_content.startswith('"') and generated_content.endswith('"'):
                generated_content = generated_content[1:-1]
            
            result = {
                'content': generated_content,
                'model_version': self.model_version,
                'confidence': 0.7,  # Placeholder confidence score
                'generated_at': datetime.now(),
                'context_used': bool(context)
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
