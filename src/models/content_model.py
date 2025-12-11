"""
Content Generation Model using Claude API with few-shot prompting.

Enhanced with:
- Structured prompt engineering for better context utilization
- Style pattern extraction from historical examples
- Entity-aware content generation
- Recent posting activity analysis
- Multi-factor confidence scoring

Can be replaced with fine-tuned models later.
"""

import os
import sys
import re
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Tuple
from collections import Counter
from loguru import logger
import yaml
from anthropic import Anthropic

from src.validation.similarity_metrics import SimilarityMetrics
from src.models.topic_model import TopicClassifier

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
        self.topic_classifier = self._init_topic_classifier()
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
    
    def _init_topic_classifier(self):
        """Initialize topic classifier for topic-aware example selection."""
        try:
            return TopicClassifier(use_bertopic=False)  # Use lightweight keyword-based
        except Exception as exc:
            logger.warning(f"Topic classifier unavailable: {exc}")
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
    
    def load_example_posts(self, num_examples=None, context=None, use_topic_diversity=True):
        """
        Load example posts for few-shot prompting with topic-aware selection.
        
        Args:
            num_examples: Number of examples to load
            context: Optional context for topic inference
            use_topic_diversity: If True, ensure examples cover diverse topics
            
        Returns:
            List of example post dicts
        """
        if num_examples is None:
            num_examples = self.config.get('claude_api', {}).get('num_examples', 10)
        
        logger.info(f"Loading {num_examples} example posts (topic_diversity={use_topic_diversity})...")
        
        session = get_session()
        
        # Get larger sample for topic-based selection
        sample_size = num_examples * 5 if use_topic_diversity else num_examples * 3
        
        posts = session.query(Post)\
            .filter(Post.content != None)\
            .filter(Post.content != '')\
            .order_by(Post.created_at.desc())\
            .limit(sample_size)\
            .all()
        
        session.close()
        
        if not posts:
            logger.warning("No posts found in database!")
            return []
        
        # Convert to dicts for processing
        now = datetime.now(timezone.utc)
        max_engagement = max(
            (p.favourites_count or 0) + (p.reblogs_count or 0) + (p.replies_count or 0)
            for p in posts
        ) or 1
        
        post_dicts = []
        for post in posts:
            engagement = (post.favourites_count or 0) + (post.reblogs_count or 0) + (post.replies_count or 0)
            engagement_score = engagement / max_engagement
            
            created_at = post.created_at or now
            if created_at.tzinfo is None:
                created_at = created_at.replace(tzinfo=timezone.utc)
            recency_hours = max((now - created_at).total_seconds() / 3600, 0.1)
            recency_score = 1 / (1 + recency_hours / 12)
            
            post_dicts.append({
                'content': post.content,
                'post_id': post.post_id,
                'created_at': post.created_at,
                'engagement': engagement,
                'engagement_score': engagement_score,
                'recency_score': recency_score
            })
        
        # Use topic-aware selection if classifier available
        if use_topic_diversity and self.topic_classifier:
            # Select diverse examples using topic classifier
            selected = self.topic_classifier.select_diverse_examples(
                posts=post_dicts,
                num_examples=num_examples,
                context=context
            )
            
            # Add engagement and recency scoring within each topic group
            trending_terms = set()
            if context and context.get('trending_keywords'):
                trending_terms = {term.lower() for term in context.get('trending_keywords', [])}
            
            final_posts = []
            for post in selected:
                # Calculate keyword relevance score
                keyword_score = 0.0
                if trending_terms:
                    content_lower = (post.get('content') or "").lower()
                    matches = sum(1 for term in trending_terms if term in content_lower)
                    keyword_score = matches / max(len(trending_terms), 1)
                
                # Combined score
                combined_score = (
                    0.4 * post.get('engagement_score', 0) +
                    0.25 * keyword_score +
                    0.15 * post.get('recency_score', 0) +
                    0.2 * post.get('topic_confidence', 0)
                )
                
                final_posts.append({
                    'content': post['content'],
                    'created_at': post.get('created_at'),
                    'engagement': post.get('engagement', 0),
                    'score': combined_score,
                    'topic': post.get('topic', 'unknown'),
                    'topic_confidence': post.get('topic_confidence', 0)
                })
            
            self.example_posts = final_posts
            
            # Log topic distribution
            topic_counts = {}
            for p in final_posts:
                topic = p.get('topic', 'unknown')
                topic_counts[topic] = topic_counts.get(topic, 0) + 1
            logger.info(f"Topic distribution in examples: {topic_counts}")
            
        else:
            # Fallback to original scoring method
            trending_terms = set()
            if context and context.get('trending_keywords'):
                trending_terms = {term.lower() for term in context.get('trending_keywords', [])}
            
            scored_posts = []
            for post in post_dicts:
                keyword_score = 0.0
                if trending_terms:
                    content_lower = (post.get('content') or "").lower()
                    matches = sum(1 for term in trending_terms if term in content_lower)
                    keyword_score = matches / max(len(trending_terms), 1)
                
                score = (
                    0.6 * post.get('engagement_score', 0) +
                    0.25 * keyword_score +
                    0.15 * post.get('recency_score', 0)
                )
                scored_posts.append((score, post))
            
            scored_posts.sort(key=lambda item: item[0], reverse=True)
            
            self.example_posts = [
                {
                    'content': post['content'],
                    'created_at': post.get('created_at'),
                    'engagement': post.get('engagement', 0),
                    'score': score
                }
                for score, post in scored_posts[:num_examples]
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
    
    def _extract_style_patterns(self) -> Dict:
        """
        Extract style patterns from example posts for better generation.
        
        Returns:
            Dict with style patterns including:
            - avg_length: Average post length
            - caps_ratio: Ratio of all-caps words
            - common_phrases: Frequently used phrases
            - punctuation_patterns: Common punctuation usage
            - exclamation_ratio: How often posts end with !
        """
        if not self.example_posts:
            return {}
        
        contents = [p['content'] for p in self.example_posts if p.get('content')]
        if not contents:
            return {}
        
        # Average length
        avg_length = sum(len(c) for c in contents) / len(contents)
        
        # CAPS ratio (words that are all caps with 2+ chars)
        all_words = []
        caps_words = []
        for content in contents:
            words = content.split()
            all_words.extend(words)
            caps_words.extend([w for w in words if w.isupper() and len(w) >= 2])
        
        caps_ratio = len(caps_words) / max(len(all_words), 1)
        
        # Common phrases (2-3 word ngrams)
        phrase_counts = Counter()
        for content in contents:
            words = content.split()
            for n in [2, 3]:
                for i in range(len(words) - n + 1):
                    phrase = ' '.join(words[i:i+n])
                    # Only count phrases that appear meaningful
                    if not all(w.lower() in {'the', 'a', 'an', 'is', 'are', 'to', 'and', 'of', 'in', 'for'} for w in words[i:i+n]):
                        phrase_counts[phrase] += 1
        
        common_phrases = [phrase for phrase, count in phrase_counts.most_common(10) if count >= 2]
        
        # Exclamation ratio
        exclamation_count = sum(1 for c in contents if c.strip().endswith('!'))
        exclamation_ratio = exclamation_count / len(contents)
        
        # Question ratio
        question_count = sum(1 for c in contents if c.strip().endswith('?'))
        question_ratio = question_count / len(contents)
        
        # Common all-caps words
        caps_word_counts = Counter(caps_words)
        top_caps_words = [word for word, count in caps_word_counts.most_common(10) if count >= 2]
        
        return {
            'avg_length': avg_length,
            'caps_ratio': caps_ratio,
            'common_phrases': common_phrases,
            'exclamation_ratio': exclamation_ratio,
            'question_ratio': question_ratio,
            'top_caps_words': top_caps_words
        }
    
    def _extract_entities_from_context(self, context: Dict) -> List[str]:
        """
        Extract key entities (names, organizations) from context for mention in post.
        
        Args:
            context: Context dict with news, trends, etc.
            
        Returns:
            List of entity names to potentially reference
        """
        entities = []
        
        if not context:
            return entities
        
        # Extract from news headlines
        headlines = context.get('top_headlines', [])
        if isinstance(headlines, list):
            for headline in headlines[:5]:
                title = headline.get('title', '') if isinstance(headline, dict) else str(headline)
                # Simple entity extraction: capitalized multi-word phrases
                words = title.split()
                i = 0
                while i < len(words):
                    # Look for sequences of capitalized words (potential names)
                    if words[i] and words[i][0].isupper() and words[i].lower() not in {'the', 'a', 'an', 'in', 'on', 'at', 'for', 'to', 'and', 'or', 'but', 'is', 'are', 'was', 'were'}:
                        entity_parts = [words[i]]
                        j = i + 1
                        while j < len(words) and words[j] and words[j][0].isupper():
                            entity_parts.append(words[j])
                            j += 1
                        if len(entity_parts) >= 1:
                            entity = ' '.join(entity_parts)
                            if len(entity) > 2 and entity not in entities:
                                entities.append(entity)
                        i = j
                    else:
                        i += 1
        
        # Extract from trending keywords
        trending = context.get('trending_keywords', [])
        if isinstance(trending, list):
            for keyword in trending[:5]:
                if keyword and keyword not in entities:
                    entities.append(keyword)
        
        return entities[:10]  # Limit to top 10
    
    def _get_recent_activity_context(self) -> Dict:
        """
        Get recent posting activity for context in prompt.
        
        Returns:
            Dict with recent activity stats
        """
        session = get_session()
        try:
            now = datetime.now(timezone.utc)
            last_24h = now - timedelta(hours=24)
            last_6h = now - timedelta(hours=6)
            last_1h = now - timedelta(hours=1)
            
            # Get recent posts
            recent_posts = session.query(Post)\
                .filter(Post.created_at >= last_24h)\
                .order_by(Post.created_at.desc())\
                .all()
            
            posts_24h = len(recent_posts)
            posts_6h = sum(1 for p in recent_posts if p.created_at and p.created_at.replace(tzinfo=timezone.utc) >= last_6h)
            posts_1h = sum(1 for p in recent_posts if p.created_at and p.created_at.replace(tzinfo=timezone.utc) >= last_1h)
            
            # Get last post time
            last_post_time = None
            hours_since_last = None
            if recent_posts:
                last_post_time = recent_posts[0].created_at
                if last_post_time:
                    if last_post_time.tzinfo is None:
                        last_post_time = last_post_time.replace(tzinfo=timezone.utc)
                    hours_since_last = (now - last_post_time).total_seconds() / 3600
            
            return {
                'posts_24h': posts_24h,
                'posts_6h': posts_6h,
                'posts_1h': posts_1h,
                'hours_since_last': hours_since_last,
                'activity_level': 'high' if posts_24h >= 15 else ('medium' if posts_24h >= 5 else 'low')
            }
        finally:
            session.close()
    
    def _build_enhanced_prompt(
        self,
        context: Optional[Dict],
        predicted_time: Optional[datetime],
        style_patterns: Dict,
        activity_context: Dict
    ) -> str:
        """
        Build an enhanced structured prompt for better content generation.
        
        Args:
            context: External context (news, trends, market)
            predicted_time: When post is predicted
            style_patterns: Extracted style patterns from examples
            activity_context: Recent posting activity
            
        Returns:
            Formatted prompt string
        """
        # Format examples
        examples = self.format_examples()
        
        # Build context section
        context_parts = []
        
        # Time context
        if predicted_time:
            context_parts.append(f"📅 Time: {predicted_time.strftime('%A, %B %d, %Y at %I:%M %p')}")
        
        # News context with entity extraction
        if context:
            # News
            if context.get('news_summary'):
                context_parts.append(f"📰 Recent News: {context['news_summary']}")
            elif context.get('top_headlines'):
                headlines = context['top_headlines'][:3]
                if headlines:
                    headline_texts = []
                    for h in headlines:
                        if isinstance(h, dict):
                            headline_texts.append(h.get('title', ''))
                        else:
                            headline_texts.append(str(h))
                    context_parts.append(f"📰 Headlines: {' | '.join(headline_texts)}")
            
            # Trending
            if context.get('trending_keywords'):
                trends = ', '.join(context['trending_keywords'][:5])
                context_parts.append(f"🔥 Trending: {trends}")
            
            # Market
            if context.get('market_sentiment'):
                sentiment = context['market_sentiment']
                sp_change = context.get('sp500_change_pct')
                if sp_change is not None:
                    market_str = f"📊 Market: {sentiment.upper()} (S&P {sp_change:+.1f}%)"
                else:
                    market_str = f"📊 Market: {sentiment.upper()}"
                context_parts.append(market_str)
            
            # Extracted entities
            entities = self._extract_entities_from_context(context)
            if entities:
                context_parts.append(f"👤 Key Entities: {', '.join(entities[:5])}")
        
        # Activity context
        if activity_context.get('posts_24h') is not None:
            activity_str = f"📱 Recent Activity: {activity_context['posts_24h']} posts in 24h"
            if activity_context.get('hours_since_last') is not None:
                activity_str += f", {activity_context['hours_since_last']:.1f}h since last post"
            context_parts.append(activity_str)
        
        context_section = "\n".join(context_parts) if context_parts else "No specific context available."
        
        # Build style guidelines from patterns
        style_parts = []
        if style_patterns:
            if style_patterns.get('avg_length'):
                style_parts.append(f"• Target length: ~{int(style_patterns['avg_length'])} characters")
            if style_patterns.get('caps_ratio', 0) > 0.05:
                style_parts.append(f"• Use ALL CAPS for emphasis on key words ({style_patterns['caps_ratio']:.0%} of words)")
            if style_patterns.get('top_caps_words'):
                style_parts.append(f"• Common emphasis words: {', '.join(style_patterns['top_caps_words'][:5])}")
            if style_patterns.get('exclamation_ratio', 0) > 0.3:
                style_parts.append("• Often ends with exclamation mark (!)")
            if style_patterns.get('common_phrases'):
                style_parts.append(f"• Characteristic phrases: {', '.join(style_patterns['common_phrases'][:5])}")
        
        style_section = "\n".join(style_parts) if style_parts else "• Direct, assertive tone\n• Use CAPS for emphasis\n• Keep it punchy and memorable"
        
        # Build the full prompt
        prompt = f"""You are generating a social media post in the authentic style of Donald Trump's Truth Social posts.

═══════════════════════════════════════════════════════════
REAL EXAMPLES OF THE POSTING STYLE
═══════════════════════════════════════════════════════════

{examples}

═══════════════════════════════════════════════════════════
CURRENT CONTEXT (use this to make the post relevant)
═══════════════════════════════════════════════════════════

{context_section}

═══════════════════════════════════════════════════════════
STYLE GUIDELINES (match these patterns)
═══════════════════════════════════════════════════════════

{style_section}

═══════════════════════════════════════════════════════════
GENERATION INSTRUCTIONS
═══════════════════════════════════════════════════════════

Generate ONE new post that:
1. Matches the authentic voice and style from the examples
2. Is relevant to the current context (news, trends, or recent events)
3. Uses characteristic capitalization patterns for emphasis
4. Stays under 280 characters
5. Sounds natural, not forced or robotic
6. Does NOT include quotation marks around the output

IMPORTANT: Output ONLY the post text, nothing else.

Generated post:"""
        
        return prompt
    
    def generate(self, context=None, predicted_time=None, use_enhanced_prompt=True):
        """
        Generate post content using Claude API with enhanced prompt engineering.
        
        Args:
            context: dict with recent news, trends, etc.
            predicted_time: datetime when post is predicted
            use_enhanced_prompt: If True, use the new structured prompt
            
        Returns:
            dict with generated content and metadata
        """
        if self.client is None:
            logger.error("Claude client not initialized!")
            return None

        if not self._within_call_budget():
            logger.warning("Anthropic call budget reached; reusing cached content instead of calling API")
            return self._fallback_content(predicted_time)
        
        # Load examples if needed
        if not self.example_posts:
            self.load_example_posts(context=context)
        
        # Build prompt
        if use_enhanced_prompt:
            # Extract style patterns for better prompt
            style_patterns = self._extract_style_patterns()
            
            # Get recent activity context
            activity_context = self._get_recent_activity_context()
            
            # Build enhanced prompt
            prompt = self._build_enhanced_prompt(
                context=context,
                predicted_time=predicted_time,
                style_patterns=style_patterns,
                activity_context=activity_context
            )
        else:
            # Fallback to basic prompt for backwards compatibility
            time_context = ""
            if predicted_time:
                time_context = f"Time context: {predicted_time.strftime('%A, %B %d, %Y at %I:%M %p')}"

            news_context = ""
            trending_context = ""
            market_context = ""

            if context:
                if 'news_summary' in context and context['news_summary']:
                    news_context = f"Recent news: {context['news_summary']}"
                if 'trending_keywords' in context and context['trending_keywords']:
                    top_trends = ', '.join(context['trending_keywords'][:5])
                    trending_context = f"Trending topics: {top_trends}"
                if 'market_sentiment' in context:
                    sentiment = context['market_sentiment']
                    sp_change = context.get('sp500_change_pct', 0)
                    market_context = f"Market: {sentiment} (S&P {sp_change:+.1f}%)"

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
            
            # Clean up output
            generated_content = self._clean_generated_content(generated_content)
            
            # Calculate multi-factor confidence
            confidence_result = self._calculate_content_confidence(
                generated_content, 
                context,
                use_enhanced_prompt
            )
            
            result = {
                'content': generated_content,
                'model_version': 'claude_enhanced_v2' if use_enhanced_prompt else self.model_version,
                'confidence': confidence_result['overall_confidence'],
                'generated_at': datetime.now(),
                'context_used': bool(context),
                'similarity_metrics': confidence_result.get('similarity_details', {}),
                'confidence_breakdown': confidence_result.get('breakdown', {}),
                'prompt_type': 'enhanced' if use_enhanced_prompt else 'basic'
            }
            
            logger.info(f"Generated content ({result['prompt_type']} prompt): {generated_content[:100]}...")
            logger.info(f"Confidence: {result['confidence']:.2%} (style={confidence_result.get('breakdown', {}).get('style_score', 0):.2f}, context={confidence_result.get('breakdown', {}).get('context_score', 0):.2f})")
            
            return result
            
        except Exception as e:
            logger.error(f"Failed to generate content: {e}")
            return None
    
    def _clean_generated_content(self, content: str) -> str:
        """
        Clean up generated content by removing artifacts.
        
        Args:
            content: Raw generated content
            
        Returns:
            Cleaned content
        """
        # Remove surrounding quotes
        if content.startswith('"') and content.endswith('"'):
            content = content[1:-1]
        if content.startswith("'") and content.endswith("'"):
            content = content[1:-1]
        
        # Remove any "Generated post:" prefix that might slip through
        prefixes_to_remove = [
            "Generated post:",
            "Post:",
            "Here's the post:",
            "Here is the post:",
        ]
        for prefix in prefixes_to_remove:
            if content.lower().startswith(prefix.lower()):
                content = content[len(prefix):].strip()
        
        # Ensure it's not too long
        if len(content) > 280:
            # Try to truncate at a sentence boundary
            sentences = content[:280].rsplit('.', 1)
            if len(sentences) > 1 and len(sentences[0]) > 100:
                content = sentences[0] + '.'
            else:
                # Just truncate
                content = content[:277] + '...'
        
        return content.strip()
    
    def _calculate_content_confidence(
        self,
        generated_content: str,
        context: Optional[Dict],
        use_enhanced: bool
    ) -> Dict:
        """
        Calculate multi-factor confidence score for generated content.
        
        Factors:
        1. Style similarity to examples (BERTScore/similarity metrics)
        2. Context relevance (does it mention entities from context?)
        3. Style consistency (length, capitalization patterns)
        
        Args:
            generated_content: The generated text
            context: Context dict used for generation
            use_enhanced: Whether enhanced prompt was used
            
        Returns:
            Dict with overall confidence and breakdown
        """
        scores = {}
        
        # 1. Style similarity (existing metric)
        similarity_score = 0.5
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
                    similarity_score = best_metrics['composite_similarity']
                    similarity_details = best_metrics
            except Exception as exc:
                logger.warning(f"Failed to calculate similarity: {exc}")
        
        scores['similarity_score'] = similarity_score
        
        # 2. Context relevance score
        context_score = 0.5
        if context:
            # Check if generated content mentions entities from context
            entities = self._extract_entities_from_context(context)
            content_lower = generated_content.lower()
            
            if entities:
                matches = sum(1 for e in entities if e.lower() in content_lower)
                context_score = min(1.0, 0.3 + (matches / len(entities)) * 0.7)
            
            # Also check for trending keywords
            trending = context.get('trending_keywords', [])
            if trending:
                trend_matches = sum(1 for t in trending[:5] if t.lower() in content_lower)
                context_score = max(context_score, min(1.0, 0.3 + trend_matches * 0.15))
        
        scores['context_score'] = context_score
        
        # 3. Style consistency score
        style_patterns = self._extract_style_patterns()
        style_score = 0.5
        
        if style_patterns:
            style_factors = []
            
            # Length similarity
            if style_patterns.get('avg_length'):
                length_ratio = len(generated_content) / style_patterns['avg_length']
                length_score = 1.0 - min(1.0, abs(1.0 - length_ratio))
                style_factors.append(length_score)
            
            # Capitalization pattern
            words = generated_content.split()
            caps_words = [w for w in words if w.isupper() and len(w) >= 2]
            actual_caps_ratio = len(caps_words) / max(len(words), 1)
            expected_caps = style_patterns.get('caps_ratio', 0.1)
            caps_score = 1.0 - min(1.0, abs(actual_caps_ratio - expected_caps) * 5)
            style_factors.append(caps_score)
            
            # Exclamation usage
            ends_exclaim = generated_content.strip().endswith('!')
            expected_exclaim = style_patterns.get('exclamation_ratio', 0.5)
            if expected_exclaim > 0.5:  # Usually ends with !
                exclaim_score = 1.0 if ends_exclaim else 0.5
            else:
                exclaim_score = 0.8  # Either way is fine
            style_factors.append(exclaim_score)
            
            style_score = sum(style_factors) / len(style_factors) if style_factors else 0.5
        
        scores['style_score'] = style_score
        
        # Calculate overall confidence (weighted average)
        weights = {
            'similarity_score': 0.4,
            'context_score': 0.3,
            'style_score': 0.3
        }
        
        overall = sum(scores[k] * weights[k] for k in weights)
        
        # Bonus for using enhanced prompt
        if use_enhanced:
            overall = min(0.99, overall * 1.05)
        
        return {
            'overall_confidence': overall,
            'breakdown': scores,
            'similarity_details': similarity_details
        }
    
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
