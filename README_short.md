# Bot or Not: Twitter Bot Detection

A machine learning system for detecting bot accounts on Twitter, combining user metadata with NLP analysis of posts. Built for the Bot or Not hackathon as a 2-person team (Egyptians), this project classifies English and French accounts using XGBoost/LightGBM ensembles.

## Key Technical Approach

**Features:**
- User metadata: tweet counts, follower ratios, username/description analysis
- Post analysis: TF-IDF text features, temporal patterns, language-specific signals (French accents, repetition detection)
- Ensemble modeling with hyperparameter tuning and threshold optimization

**Results:**
- Strong holdout performance (F1 ~0.85 English, ~0.82 French)
- Placed respectably but learned a crucial lesson about evaluation metrics

## The Big Lesson

We optimized for F1 score but the competition penalized false positives 5x more than false negatives. Solid technical work, but we didn't align our metrics with the actual scoring. Next time: understand the evaluation criteria first.

## Quick Start

```bash
pip install -r requirements.txt
# Place datasets in data/ folder
python src/bot_detector.py
```

Generates submission files for English/French test sets.

## Repo Structure

- `src/bot_detector.py`: Main detector implementation
- `notebooks/bot_detector.ipynb`: Exploration notebook  
- `data/`: Dataset files (not included)
- `explore_data.py`: Data utilities

This was my first hackathon—great technical challenge, even better learning experience about real ML deployment.