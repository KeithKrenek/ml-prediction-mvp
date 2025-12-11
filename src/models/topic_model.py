"""
Topic Model for Content Classification

Provides topic-aware example selection for content generation.
Uses keyword-based topic classification (lightweight) with optional
BERTopic support for more sophisticated clustering.

Topics are based on common Trump posting themes:
- Political opponents
- Economy/Jobs
- Immigration/Border
- Media criticism
- Election/Voting
- Foreign policy
- Rally announcements
- Personal/Grievances
- Legal matters
- General announcements
"""

import re
from typing import Dict, List, Optional, Tuple
from collections import Counter, defaultdict
from datetime import datetime, timezone
import numpy as np
from loguru import logger

# Try to import BERTopic for advanced clustering
try:
    from bertopic import BERTopic
    BERTOPIC_AVAILABLE = True
except ImportError:
    BERTOPIC_AVAILABLE = False
    logger.info("BERTopic not available. Using keyword-based topic classification.")


# Predefined topic keywords for lightweight classification
TOPIC_KEYWORDS = {
    'political_opponents': {
        'keywords': [
            'biden', 'obama', 'clinton', 'hillary', 'pelosi', 'schumer',
            'democrats', 'democrat', 'radical left', 'crooked', 'sleepy',
            'corrupt', 'crazy', 'lying', 'cheatin', 'nasty', 'low iq',
            'pocahontas', 'nervous nancy', 'shifty', 'lyin'
        ],
        'weight': 1.0
    },
    'economy_jobs': {
        'keywords': [
            'economy', 'jobs', 'unemployment', 'stock market', 'dow',
            'nasdaq', 'gdp', 'growth', 'trade', 'tariff', 'china trade',
            'manufacturing', 'wages', 'inflation', 'tax', 'business',
            'companies', 'markets', 'record', 'billion', 'trillion'
        ],
        'weight': 1.0
    },
    'immigration_border': {
        'keywords': [
            'border', 'wall', 'immigration', 'illegal', 'caravan',
            'mexico', 'migrants', 'ice', 'deportation', 'aliens',
            'sanctuary', 'invasion', 'drugs', 'fentanyl', 'cartels',
            'smugglers', 'coyotes', 'open borders'
        ],
        'weight': 1.0
    },
    'media_criticism': {
        'keywords': [
            'fake news', 'media', 'cnn', 'msnbc', 'nbc', 'abc', 'cbs',
            'new york times', 'washington post', 'nyt', 'wapo',
            'reporters', 'journalists', 'press', 'ratings', 'failing',
            'enemy of the people', 'dishonest', 'msm', 'mainstream'
        ],
        'weight': 1.0
    },
    'election_voting': {
        'keywords': [
            'election', 'vote', 'voting', 'ballot', 'fraud', 'rigged',
            'stolen', 'dominion', 'mail-in', 'absentee', 'poll',
            '2020', '2024', 'win', 'landslide', 'electoral', 'recount',
            'audit', 'certification', 'swing state'
        ],
        'weight': 1.0
    },
    'foreign_policy': {
        'keywords': [
            'china', 'russia', 'putin', 'xi', 'north korea', 'kim',
            'iran', 'israel', 'ukraine', 'nato', 'afghanistan',
            'military', 'troops', 'war', 'peace', 'nuclear', 'missile',
            'sanctions', 'deal', 'foreign', 'allies', 'enemies'
        ],
        'weight': 1.0
    },
    'rally_events': {
        'keywords': [
            'rally', 'crowd', 'packed', 'thousands', 'arena', 'stadium',
            'supporters', 'fans', 'amazing', 'incredible crowd',
            'standing room', 'overflow', 'love', 'maga', 'kag',
            'make america', 'save america', 'event', 'tonight'
        ],
        'weight': 1.0
    },
    'legal_matters': {
        'keywords': [
            'witch hunt', 'hoax', 'investigation', 'fbi', 'doj',
            'indictment', 'trial', 'judge', 'court', 'lawsuit',
            'attorney', 'lawyer', 'prosecution', 'persecution',
            'justice', 'supreme court', 'unconstitutional', 'illegal'
        ],
        'weight': 1.0
    },
    'personal_grievances': {
        'keywords': [
            'unfair', 'treated', 'never', 'worst', 'terrible', 'sad',
            'disgrace', 'shame', 'attacked', 'persecuted', 'targeted',
            'wronged', 'innocent', 'victim', 'enemies', 'haters'
        ],
        'weight': 0.8
    },
    'announcements': {
        'keywords': [
            'announcing', 'announcement', 'excited', 'proud', 'honor',
            'congratulations', 'congrats', 'welcome', 'endorse',
            'endorsement', 'support', 'backing', 'join', 'together',
            'stay tuned', 'big news', 'breaking'
        ],
        'weight': 0.9
    }
}


class TopicClassifier:
    """
    Lightweight topic classifier for Trump-style posts.
    
    Uses keyword matching with optional BERTopic for more
    sophisticated clustering when available.
    """
    
    def __init__(self, use_bertopic: bool = False):
        """
        Initialize topic classifier.
        
        Args:
            use_bertopic: If True and available, use BERTopic
        """
        self.use_bertopic = use_bertopic and BERTOPIC_AVAILABLE
        self.bertopic_model = None
        self.topic_keywords = TOPIC_KEYWORDS
        
        logger.info(f"TopicClassifier initialized (BERTopic: {self.use_bertopic})")
    
    def classify_post(self, content: str) -> Dict:
        """
        Classify a single post into topics.
        
        Args:
            content: Post text content
            
        Returns:
            Dict with topic scores and primary topic
        """
        if not content:
            return {
                'primary_topic': 'unknown',
                'topic_scores': {},
                'confidence': 0.0
            }
        
        content_lower = content.lower()
        topic_scores = {}
        
        # Score each topic based on keyword matches
        for topic, config in self.topic_keywords.items():
            keywords = config['keywords']
            weight = config.get('weight', 1.0)
            
            # Count keyword matches
            matches = 0
            matched_keywords = []
            for keyword in keywords:
                if keyword in content_lower:
                    matches += 1
                    matched_keywords.append(keyword)
            
            # Normalize by number of keywords and apply weight
            if keywords:
                score = (matches / len(keywords)) * weight
                # Boost if multiple matches
                if matches >= 2:
                    score *= (1 + 0.1 * (matches - 1))
                topic_scores[topic] = min(1.0, score)
        
        # Determine primary topic
        if topic_scores:
            primary_topic = max(topic_scores, key=topic_scores.get)
            confidence = topic_scores[primary_topic]
        else:
            primary_topic = 'general'
            confidence = 0.0
        
        # If no clear topic, mark as general
        if confidence < 0.05:
            primary_topic = 'general'
        
        return {
            'primary_topic': primary_topic,
            'topic_scores': topic_scores,
            'confidence': confidence,
            'top_3_topics': sorted(topic_scores.items(), key=lambda x: x[1], reverse=True)[:3]
        }
    
    def classify_posts(self, posts: List[Dict]) -> List[Dict]:
        """
        Classify multiple posts.
        
        Args:
            posts: List of post dicts with 'content' key
            
        Returns:
            List of classification results
        """
        results = []
        for post in posts:
            content = post.get('content', '')
            classification = self.classify_post(content)
            classification['post'] = post
            results.append(classification)
        
        return results
    
    def get_topic_distribution(self, posts: List[Dict]) -> Dict[str, int]:
        """
        Get distribution of topics across posts.
        
        Args:
            posts: List of post dicts
            
        Returns:
            Dict mapping topic to count
        """
        classifications = self.classify_posts(posts)
        distribution = Counter(c['primary_topic'] for c in classifications)
        return dict(distribution)
    
    def select_diverse_examples(
        self,
        posts: List[Dict],
        num_examples: int = 10,
        target_topics: Optional[List[str]] = None,
        context: Optional[Dict] = None
    ) -> List[Dict]:
        """
        Select diverse examples covering multiple topics.
        
        Ensures examples span different topics for better
        few-shot prompting.
        
        Args:
            posts: List of post dicts to select from
            num_examples: Number of examples to select
            target_topics: If specified, prioritize these topics
            context: Optional context to determine relevant topics
            
        Returns:
            List of selected post dicts with topic info
        """
        if not posts:
            return []
        
        # Classify all posts
        classified = self.classify_posts(posts)
        
        # Determine target topics from context
        if context and not target_topics:
            target_topics = self._infer_topics_from_context(context)
        
        # Group posts by topic
        posts_by_topic = defaultdict(list)
        for c in classified:
            posts_by_topic[c['primary_topic']].append(c)
        
        selected = []
        selected_ids = set()
        
        # If we have target topics, prioritize those
        if target_topics:
            for topic in target_topics:
                topic_posts = posts_by_topic.get(topic, [])
                # Sort by confidence
                topic_posts.sort(key=lambda x: x['confidence'], reverse=True)
                
                for post_info in topic_posts:
                    post = post_info['post']
                    post_id = post.get('post_id', id(post))
                    if post_id not in selected_ids:
                        selected.append({
                            **post,
                            'topic': topic,
                            'topic_confidence': post_info['confidence']
                        })
                        selected_ids.add(post_id)
                        if len(selected) >= num_examples // 2:
                            break
                
                if len(selected) >= num_examples // 2:
                    break
        
        # Fill remaining slots with diverse topics
        all_topics = list(posts_by_topic.keys())
        topic_idx = 0
        
        while len(selected) < num_examples and topic_idx < len(all_topics) * 2:
            topic = all_topics[topic_idx % len(all_topics)]
            topic_posts = posts_by_topic.get(topic, [])
            
            for post_info in topic_posts:
                post = post_info['post']
                post_id = post.get('post_id', id(post))
                if post_id not in selected_ids:
                    selected.append({
                        **post,
                        'topic': topic,
                        'topic_confidence': post_info['confidence']
                    })
                    selected_ids.add(post_id)
                    break
            
            topic_idx += 1
        
        # If still not enough, add remaining posts
        for c in classified:
            if len(selected) >= num_examples:
                break
            post = c['post']
            post_id = post.get('post_id', id(post))
            if post_id not in selected_ids:
                selected.append({
                    **post,
                    'topic': c['primary_topic'],
                    'topic_confidence': c['confidence']
                })
                selected_ids.add(post_id)
        
        logger.info(f"Selected {len(selected)} diverse examples across {len(set(s.get('topic') for s in selected))} topics")
        
        return selected[:num_examples]
    
    def _infer_topics_from_context(self, context: Dict) -> List[str]:
        """
        Infer relevant topics from context.
        
        Args:
            context: Context dict with news, trends, etc.
            
        Returns:
            List of relevant topic names
        """
        relevant_topics = []
        
        # Combine all context text
        context_text = ""
        
        if context.get('news_summary'):
            context_text += " " + context['news_summary']
        
        if context.get('top_headlines'):
            for h in context['top_headlines'][:5]:
                if isinstance(h, dict):
                    context_text += " " + h.get('title', '')
                else:
                    context_text += " " + str(h)
        
        if context.get('trending_keywords'):
            context_text += " " + " ".join(context['trending_keywords'])
        
        if not context_text.strip():
            return []
        
        # Classify the context
        classification = self.classify_post(context_text)
        
        # Return top topics
        top_topics = [topic for topic, score in classification.get('top_3_topics', []) if score > 0.02]
        
        logger.debug(f"Inferred topics from context: {top_topics}")
        
        return top_topics
    
    def get_topic_summary(self, topic: str) -> str:
        """
        Get a human-readable summary of a topic.
        
        Args:
            topic: Topic name
            
        Returns:
            Summary string
        """
        topic_summaries = {
            'political_opponents': 'Posts about political opponents (Biden, Democrats, etc.)',
            'economy_jobs': 'Posts about economy, jobs, and markets',
            'immigration_border': 'Posts about immigration and border security',
            'media_criticism': 'Posts criticizing media and press',
            'election_voting': 'Posts about elections and voting',
            'foreign_policy': 'Posts about foreign policy and international relations',
            'rally_events': 'Posts about rallies and events',
            'legal_matters': 'Posts about legal proceedings and investigations',
            'personal_grievances': 'Posts expressing personal grievances',
            'announcements': 'General announcements and endorsements',
            'general': 'General posts without specific topic'
        }
        return topic_summaries.get(topic, f'Posts about {topic}')


def test_topic_classifier():
    """Test the topic classifier."""
    print("\n" + "="*80)
    print("TOPIC CLASSIFIER TEST")
    print("="*80 + "\n")
    
    classifier = TopicClassifier()
    
    # Test posts
    test_posts = [
        {"content": "Crooked Joe Biden is DESTROYING our country! The worst president in history. SAD!", "post_id": "1"},
        {"content": "Stock Market at RECORD HIGHS! Jobs numbers incredible. The economy is BOOMING!", "post_id": "2"},
        {"content": "Build the WALL! We need to secure our borders NOW. Millions pouring in illegally.", "post_id": "3"},
        {"content": "Fake News CNN is at it again. Ratings in the toilet. FAILING badly!", "post_id": "4"},
        {"content": "Just had an AMAZING rally in Iowa! Thousands of great patriots showed up. Thank you!", "post_id": "5"},
        {"content": "The radical left prosecutors are on a WITCH HUNT. Total persecution!", "post_id": "6"},
        {"content": "China is ripping us off. Time to get TOUGH on trade. America First!", "post_id": "7"},
        {"content": "2020 Election was RIGGED and STOLEN. We all know it. Never forget!", "post_id": "8"},
    ]
    
    print("Classifying test posts:\n")
    
    for post in test_posts:
        result = classifier.classify_post(post['content'])
        print(f"Post: {post['content'][:60]}...")
        print(f"  Primary Topic: {result['primary_topic']}")
        print(f"  Confidence: {result['confidence']:.2%}")
        print(f"  Top 3: {result['top_3_topics']}")
        print()
    
    # Test distribution
    print("="*60)
    print("Topic Distribution:")
    distribution = classifier.get_topic_distribution(test_posts)
    for topic, count in sorted(distribution.items(), key=lambda x: x[1], reverse=True):
        print(f"  {topic}: {count}")
    print()
    
    # Test diverse selection
    print("="*60)
    print("Diverse Example Selection (5 posts):")
    selected = classifier.select_diverse_examples(test_posts, num_examples=5)
    for s in selected:
        print(f"  [{s.get('topic', 'unknown')}] {s['content'][:50]}...")
    print()
    
    # Test with context
    print("="*60)
    print("Selection with Context (economy-related):")
    context = {'trending_keywords': ['economy', 'jobs', 'stock market', 'recession']}
    selected_context = classifier.select_diverse_examples(test_posts, num_examples=4, context=context)
    for s in selected_context:
        print(f"  [{s.get('topic', 'unknown')}] {s['content'][:50]}...")
    print()
    
    print("="*80 + "\n")


if __name__ == "__main__":
    test_topic_classifier()

