# ML Prediction System - Comprehensive Improvement Plan

**Context**: Trump posts 20+ times per day, making current 6-12 hour accuracy windows nearly useless. We need hour-level or better accuracy for timing, and semantic understanding for content.

**Current Performance Issues**:
- Timing: Only 2 features used (engagement, time_since_last)
- Content: Hardcoded 0.7 confidence, no semantic similarity
- Model: Prophet not designed for high-frequency point processes
- Data: Only 40-50 training samples minimum

---

## 🎯 IMPROVEMENT CATEGORIES

### **TIER 1: CRITICAL - High Impact, Medium Effort (1-2 weeks)**
*These address the most severe accuracy bottlenecks*

#### **1.1 Implement Neural Temporal Point Process (NTPP) for Timing**
**Problem**: Prophet treats posts as periodic events with daily/weekly seasonality. With 20+ posts/day, we need to model the conditional intensity of the next post given recent history.

**Solution**: Replace Prophet with Neural Temporal Point Process
- Models inter-event times (time between consecutive posts)
- Captures **burst patterns** (Trump often posts 5-10 times in an hour, then goes silent)
- Uses self-attention to condition on recent post history
- Can predict "time until next post" with confidence intervals

**Implementation**:
- Use existing library: `tick` or `neural_hawkes`
- Features: Recent post times, engagement velocities, hour-of-day embeddings
- Output: Probability distribution over next post time (not just point estimate)
- Training: RNN/Transformer on sequences of post timestamps

**Expected Gain**: 50-70% improvement in MAE (from ~6h to ~2-3h)

**Files to Modify**:
- `src/models/predictor.py` - New `NTPPTimingModel` class
- `src/models/trainer.py` - Add NTPP training pipeline
- `requirements.txt` - Add neural point process library

---

#### **1.2 Implement Semantic Content Similarity (BERTScore + Embeddings)**
**Problem**: Current similarity is word overlap (Jaccard). Doesn't understand meaning. "Crooked Hillary" vs "Corrupt Clinton" scores 0% but are semantically identical.

**Solution**: Multi-layered semantic similarity
- **BERTScore**: Compares contextual embeddings token-by-token (SOTA for generation metrics)
- **Sentence Embeddings**: Use `sentence-transformers` for cosine similarity
- **Entity Recognition**: Extract and match named entities (people, places, organizations)
- **Topic Modeling**: Classify predicted vs actual into topics (immigration, economy, etc.)

**Implementation**:
- BERTScore using `bert-score` library (already has placeholder)
- Sentence embeddings with `all-MiniLM-L6-v2` (fast, 384-dim)
- NER with spaCy or Flair
- Weighted composite: 40% BERTScore + 30% Sentence + 20% Entity + 10% Lexical

**Expected Gain**: 60-80% improvement in content matching (from 30% to 50-70%)

**Files to Modify**:
- `src/validation/validator.py` - Enhance `calculate_similarity_metrics()`
- `requirements.txt` - Add bert-score, sentence-transformers, spacy

---

#### **1.3 Advanced Time-Series Feature Engineering**
**Problem**: Only using 2 features (engagement, time_since_last). Ignoring hour-of-day, day-of-week, trends, velocities.

**Solution**: Rich feature set for timing model
- **Temporal Features**:
  - Hour-of-day (cyclical encoding: sin/cos)
  - Day-of-week (cyclical encoding)
  - Is weekend, is business hours
  - Time since midnight, time until midnight

- **Historical Pattern Features**:
  - Posts in last 1h, 3h, 6h, 12h, 24h (sliding windows)
  - Average inter-post time over last 10 posts
  - Std dev of inter-post times (measures regularity)
  - Max posts in 1-hour window over last 7 days

- **Engagement Velocity Features**:
  - Engagement rate on last post (per hour since posted)
  - Rolling average engagement over last 5 posts
  - Ratio of current engagement to average (outlier detection)

- **Context-Driven Features**:
  - Is there breaking news? (sentiment spike in headlines)
  - Stock market movement magnitude today
  - Number of trending political topics
  - Time since last major news event

**Implementation**:
- Create `FeatureEngineer` class in `src/features/engineering.py`
- Compute features on-demand during prediction and training
- Store feature importance scores to prune low-signal features

**Expected Gain**: 30-50% timing improvement when combined with NTPP

**Files to Create**:
- `src/features/engineering.py` - New feature engineering module
- `src/features/temporal.py` - Temporal feature extractors
- `src/features/context.py` - Context-based features

---

#### **1.4 Implement Post Clustering & Topic Classification**
**Problem**: Current content prediction uses 10 random examples. No understanding of Trump's topic patterns or which topics cluster together.

**Solution**: Categorize posts into topics and predict topic probabilities
- **Topic Model**: LDA or BERTopic on historical posts
- **Classification**: Train classifier to predict topic distribution
- **Examples Selection**: Instead of random 10, select 2-3 from each likely topic
- **Prompt Engineering**: "Given recent posts about {topics}, predict next post"

**Topics Examples**:
- Immigration/Border
- Economy/Jobs
- Political Opponents
- Election/Fraud Claims
- Foreign Policy
- Media Criticism
- Rally Announcements
- Personal/Grievances

**Implementation**:
- Offline: Run BERTopic on all historical posts, assign topics
- Store topic labels in database (add `topic` column to `posts` table)
- At prediction time: Predict topic probabilities, sample examples proportionally
- Enhance prompt with topic context

**Expected Gain**: 40-60% content accuracy improvement

**Files to Modify**:
- `src/data/database.py` - Add `topic` and `topic_probability` columns
- `src/models/predictor.py` - Add topic prediction to content pipeline
- Create `src/models/topic_model.py` - Topic modeling module

---

### **TIER 2: HIGH IMPACT - Medium-High Effort (2-4 weeks)**
*Significant improvements requiring more infrastructure*

#### **2.1 Multi-Model Ensemble for Timing**
**Problem**: Single model (Prophet) can't capture all patterns. Trump's posting is multi-modal: scheduled rallies, reactive (news), random.

**Solution**: Ensemble of specialized models
- **Model 1: NTPP** - Captures burst patterns and inter-event dynamics
- **Model 2: XGBoost** - Uses tabular features (hour, day, engagement, context)
- **Model 3: ARIMA** - Captures long-term trends and seasonality
- **Ensemble**: Weighted average or meta-model that learns optimal weights

**Gating Mechanism**: Predict which model to trust based on context
- If recent posts <1h ago → Use NTPP (burst mode)
- If breaking news detected → Use XGBoost with context features
- If regular pattern → Use ARIMA

**Expected Gain**: 20-30% additional improvement over single NTPP

**Files to Create**:
- `src/models/ensemble.py` - Ensemble coordinator
- `src/models/xgboost_timing.py` - XGBoost timing model
- `src/models/arima_timing.py` - ARIMA baseline

---

#### **2.2 Fine-Tune Small Language Model for Content Generation**
**Problem**: Claude API is expensive, slow, and doesn't learn from mistakes. No feedback loop.

**Solution**: Fine-tune a small open-source LLM (e.g., Llama-3-8B, Mistral-7B)
- **Training Data**: Historical posts as training examples
- **Fine-Tuning**: LoRA or QLoRA for parameter-efficient training
- **Prompt**: Include context (news, trends, recent posts)
- **Advantage**: Can run locally, improve over time, much faster

**Hybrid Approach**: Use Claude for initial training data labeling, then fine-tune smaller model

**Cost Savings**: ~$0.015 per prediction (Claude) → ~$0.001 (self-hosted)

**Expected Gain**: 20-40% content accuracy improvement + 93% cost reduction

**Files to Create**:
- `src/models/finetuned_llm.py` - Fine-tuned content generator
- `scripts/finetune_content_model.py` - Training script
- `configs/llm_config.yaml` - Model configuration

---

#### **2.3 Implement Real-Time Confidence Calibration**
**Problem**: All content predictions have hardcoded 0.7 confidence. No way to know which predictions to trust.

**Solution**: Dynamic confidence scoring
- **Timing Confidence**: Std dev of NTPP distribution (wider = less confident)
- **Content Confidence**:
  - Model perplexity (lower = more confident)
  - Similarity to training data (higher = more confident)
  - Topic clarity (single dominant topic = more confident)
  - Context freshness score (stale context = less confident)

- **Calibration**: Use validation set to calibrate confidence → actual accuracy
  - Train isotonic regression: predicted_confidence → true_accuracy
  - Output calibrated probabilities

**Use Case**: Only report predictions with >0.6 calibrated confidence

**Expected Gain**: Reduces false alarms by 50-70%

**Files to Modify**:
- `src/models/predictor.py` - Add `calculate_confidence()` methods
- Create `src/models/calibration.py` - Confidence calibration module

---

#### **2.4 Add Visual Content Analysis for Image Posts**
**Problem**: Many Trump posts contain images/memes with no text. Current system ignores these completely.

**Solution**: Multi-modal understanding
- **OCR**: Extract text from images (Tesseract or cloud OCR)
- **Image Classification**: Detect image type (rally photo, meme, infographic, etc.)
- **Object Detection**: Identify people, flags, locations in images
- **CLIP Embeddings**: Get semantic embedding of image content
- **Combined Similarity**: Text + Image embeddings for content matching

**Implementation**:
- Download images from posts (store URLs in DB)
- Run OCR + CLIP on images
- Store embeddings in vector database (pgvector extension)
- At validation: Compare image embeddings of predicted vs actual

**Expected Gain**: Increases coverage from ~60% (text-only) to ~95% (text + images)

**Files to Create**:
- `src/data/image_processor.py` - Image download and processing
- `src/validation/image_similarity.py` - Visual similarity metrics
- Update `src/data/database.py` - Add image_url, image_embedding columns

---

### **TIER 3: MEDIUM IMPACT - Lower Priority (4-8 weeks)**
*Nice-to-haves and infrastructure improvements*

#### **3.1 Real-Time Event Detection & Triggers**
**Problem**: Trump often posts in response to breaking news. Current system doesn't detect triggers.

**Solution**: Event monitoring and reactive prediction
- **News Monitoring**: Real-time RSS feeds, Google News alerts
- **Trigger Detection**: Sentiment spikes, specific keywords (Biden, impeachment, etc.)
- **Reactive Predictions**: If trigger detected, predict post within 1-2 hours
- **Bayesian Update**: Adjust timing predictions based on trigger probability

**Expected Gain**: 30-50% improvement on reactive posts (but only ~40% of posts)

---

#### **3.2 Add Previous Post Continuation Detection**
**Problem**: Trump often posts threads (1/2, 2/2). Current system treats each post independently.

**Solution**: Thread detection and continuation modeling
- **Thread Detection**: Identify when posts are continuations
- **Context Propagation**: Include previous thread posts in content prediction
- **Timing Adjustment**: Threads usually posted 1-5 minutes apart

**Expected Gain**: 60-80% accuracy improvement on thread posts (~15% of all posts)

---

#### **3.3 Implement Active Learning for Labeling**
**Problem**: Need more training data but manual labeling is expensive.

**Solution**: Active learning loop
- Predict posts
- Identify low-confidence predictions
- Request human labeling for uncertain cases
- Retrain with new labels
- Focus labeling budget on most informative examples

**Expected Gain**: 2-3x more effective training data usage

---

#### **3.4 A/B Testing Framework for Models**
**Problem**: No safe way to test new models in production.

**Solution**: Shadow deployment and A/B testing
- Run multiple models in parallel
- Compare predictions to actual outcomes
- Gradual rollout (10% → 50% → 100%)
- Automatic rollback if metrics degrade

---

#### **3.5 Anomaly Detection for Prediction Quality**
**Problem**: Bad predictions go unnoticed until validation runs (hours later).

**Solution**: Real-time anomaly detection
- Flag predictions with unusual features (very high/low confidence, outlier timing, etc.)
- Alert on model degradation
- Auto-disable prediction if anomaly rate exceeds threshold

---

## 📊 PRIORITIZATION MATRIX

### **Impact vs Effort**

| Priority | Improvement | Impact | Effort | Timing Gain | Content Gain | Timeline |
|----------|-------------|--------|--------|-------------|--------------|----------|
| **P0** | Neural Temporal Point Process | 🔥🔥🔥 | ⚙️⚙️ | 50-70% | - | 1-2 weeks |
| **P1** | Semantic Similarity (BERTScore) | 🔥🔥🔥 | ⚙️⚙️ | - | 60-80% | 1 week |
| **P2** | Advanced Feature Engineering | 🔥🔥🔥 | ⚙️⚙️ | 30-50% | - | 1 week |
| **P3** | Post Clustering & Topics | 🔥🔥 | ⚙️⚙️ | - | 40-60% | 1-2 weeks |
| **P4** | Confidence Calibration | 🔥🔥 | ⚙️⚙️ | Reduces FP | Reduces FP | 1 week |
| **P5** | Multi-Model Ensemble | 🔥🔥 | ⚙️⚙️⚙️ | 20-30% | - | 2-3 weeks |
| **P6** | Fine-Tune LLM | 🔥🔥 | ⚙️⚙️⚙️ | - | 20-40% | 3-4 weeks |
| **P7** | Visual Content Analysis | 🔥🔥 | ⚙️⚙️⚙️ | - | +35% coverage | 2-3 weeks |
| **P8** | Real-Time Event Detection | 🔥 | ⚙️⚙️ | 30-50% reactive | - | 2-4 weeks |
| **P9** | Thread Detection | 🔥 | ⚙️ | Thread timing | Thread content | 1 week |
| **P10** | Active Learning | 🔥 | ⚙️⚙️ | Data efficiency | Data efficiency | 3-4 weeks |
| **P11** | A/B Testing Framework | 🔥 | ⚙️⚙️ | Infrastructure | Infrastructure | 2-3 weeks |

Legend:
- 🔥 Impact: 1=Low, 2=Medium, 3=High
- ⚙️ Effort: 1=Low (1 week), 2=Medium (2-3 weeks), 3=High (4+ weeks)

---

## 🎯 RECOMMENDED IMPLEMENTATION SEQUENCE

### **Phase 1: Quick Wins (Weeks 1-3)**
Focus on improvements that can be implemented quickly with high impact:

1. **Week 1**: Semantic Similarity (P1)
   - Implement BERTScore + sentence embeddings
   - Add entity matching
   - **Deliverable**: 60-80% content accuracy improvement

2. **Week 2**: Advanced Feature Engineering (P2)
   - Build feature engineering module
   - Add temporal, engagement, context features
   - **Deliverable**: Feature set ready for new models

3. **Week 3**: Confidence Calibration (P4)
   - Implement dynamic confidence scoring
   - Calibrate on validation set
   - **Deliverable**: Reliable confidence scores

**Expected Total Gain**: 60-80% content improvement, 10-20% timing improvement

---

### **Phase 2: Core Model Improvements (Weeks 4-7)**
Replace foundational models with better architectures:

4. **Weeks 4-5**: Neural Temporal Point Process (P0)
   - Implement NTPP model
   - Train on historical data
   - A/B test against Prophet
   - **Deliverable**: 50-70% timing MAE improvement

5. **Weeks 6-7**: Post Clustering & Topics (P3)
   - Train topic model on historical posts
   - Integrate topics into content prediction
   - **Deliverable**: 40-60% content accuracy improvement

**Expected Total Gain**: 50-70% timing improvement, 100-140% content improvement (cumulative)

---

### **Phase 3: Advanced Capabilities (Weeks 8-12)**
Add sophisticated features for edge cases:

6. **Weeks 8-10**: Visual Content Analysis (P7)
   - Implement image processing pipeline
   - Add multi-modal similarity
   - **Deliverable**: 95% coverage (vs 60% text-only)

7. **Weeks 10-12**: Multi-Model Ensemble (P5)
   - Implement XGBoost and ARIMA models
   - Build ensemble coordinator
   - **Deliverable**: Additional 20-30% timing improvement

**Expected Total Gain**: 70-100% timing improvement, 100-140% content improvement, 95% coverage

---

### **Phase 4: Production Optimization (Weeks 13-16)**
Operationalize and optimize:

8. **Weeks 13-14**: Real-Time Event Detection (P8)
9. **Weeks 15-16**: A/B Testing Framework (P11)

---

## 📈 PROJECTED ACCURACY IMPROVEMENTS

### **Current State**
- Timing MAE: ~6-8 hours (estimated based on 6h/12h thresholds)
- Content Similarity: ~30-40% (based on word overlap)
- Coverage: ~60% (text-only posts)

### **After Phase 1 (Week 3)**
- Timing MAE: ~5-6 hours (-10-20%)
- Content Similarity: ~50-70% (+60-80%)
- Coverage: ~60%

### **After Phase 2 (Week 7)**
- Timing MAE: ~2-3 hours (-50-70%)
- Content Similarity: ~70-85% (+100-140%)
- Coverage: ~60%

### **After Phase 3 (Week 12)**
- Timing MAE: ~1-2 hours (-70-100%)
- Content Similarity: ~75-90% (+120-160%)
- Coverage: ~95% (+58%)

### **With 20+ Posts/Day Context**
- Current "Within 6h" accuracy: ~50% (almost random - many posts in 6h window)
- After improvements: "Within 1h" accuracy: ~60-70% (actually useful!)

---

## 🔧 TECHNICAL DEBT & PREREQUISITES

### **Infrastructure Needed**
1. **Vector Database**: For storing embeddings (pgvector PostgreSQL extension)
2. **GPU Access**: For NTPP training and fine-tuning (can use cloud)
3. **Increased Storage**: Image storage (~100GB for historical images)
4. **API Rate Limits**: NewsAPI upgrade or alternative sources

### **Data Quality Improvements**
1. **Deduplication**: Remove duplicate/retweeted posts
2. **Outlier Detection**: Flag and handle anomalous posts
3. **Missing Data Handling**: Imputation strategies for incomplete posts
4. **Validation Set**: Proper train/val/test split (currently no validation set)

---

## 🎓 LEARNING & EXPERIMENTATION

### **Experiments to Run**
1. **Ablation Studies**: Which features contribute most to accuracy?
2. **Hyperparameter Tuning**: Grid search for optimal model configs
3. **Error Analysis**: What types of posts are hardest to predict?
4. **Seasonal Patterns**: How does accuracy vary by time/day/month?

### **Metrics to Track**
- Per-topic accuracy (which topics are easiest/hardest?)
- Per-time-of-day accuracy (morning vs evening posts)
- Accuracy vs post frequency (does accuracy degrade during bursts?)
- Confidence calibration curves

---

## 💡 INNOVATION IDEAS (EXPLORATORY)

### **Radical Approaches** (Higher Risk, Higher Reward)
1. **Reinforcement Learning**: Treat prediction as sequential decision-making
2. **Causal Inference**: Model causal relationships (news → post)
3. **Graph Neural Networks**: Model Trump's interaction network
4. **Diffusion Models**: Generate post content as denoising process
5. **Multi-Task Learning**: Jointly predict timing, content, engagement

---

## 📋 DECISION FRAMEWORK

### **How to Choose What to Implement First?**

**Optimize for**:
1. **Impact**: Largest expected accuracy gain
2. **Risk**: Lowest technical risk
3. **Dependencies**: Can be implemented independently
4. **Learning**: Provides insights for next steps

**My Recommendation**: Start with **P1 (Semantic Similarity)** because:
- Highest confidence in success (well-established technique)
- No model retraining needed (drop-in replacement)
- Immediate feedback (can test on existing predictions)
- Unlocks better validation for all future improvements
- 1 week effort, 60-80% content gain

**Then move to P0 (NTPP)** for timing improvements.

---

## 🚀 NEXT STEPS

1. **Review this plan** and select priority improvements
2. **Set up infrastructure** (vector DB, GPU, storage)
3. **Implement Phase 1** improvements (Weeks 1-3)
4. **Measure impact** on validation set
5. **Iterate** based on results

Ready to begin implementation whenever you are! Let me know which improvement you'd like to tackle first.
