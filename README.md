# Bot or Not: Twitter Bot Detection

This is a bot detector I built for Twitter accounts—combines user stats with text analysis to spot the fakes. Built for the Bot or Not hackathon, this project classifies English and French Twitter accounts as human or bot using XGBoost and LightGBM ensemble models.

## Why This Project Was Interesting

This was my first hackathon experience, and it taught me a lot about real-world ML deployment. The challenge was straightforward on paper—detect bots from user metadata and posts—but the competition scoring revealed some harsh realities about model evaluation. We had a solid F1 score during development, but the final leaderboard penalized false positives way more than we expected. It was a good reminder that metrics matter, and understanding the business impact (or in this case, competition rules) is just as important as technical performance.

## Problem Statement

Given Twitter user profiles and their recent posts, classify accounts as either human or bot. The datasets include:
- User metadata (tweet counts, follower/following ratios, descriptions, usernames)
- Post content and timestamps
- Separate English and French datasets with different bot behaviors

The goal was to build a detector that generalizes across languages and handles the class imbalance typical of bot detection tasks.

## Team Context / My Role

My buddy and I teamed up for this—I'm not sure who convinced who to do a hackathon, but here we are. We were a 2-person team called "Egyptians." I led most of the machine learning development, focusing on feature engineering and model selection. My partner handled data preprocessing and some of the French language specifics. It was a great collaboration—we split the work based on our strengths and learned a ton from each other's approaches.

## Technical Approach

The solution combines traditional tabular ML with NLP techniques:

### Data Processing / Feature Engineering

**User-Level Features:**
- Basic counts: tweet_count, followers_count, following_count
- Text analysis on usernames: length, presence of numbers/underscores
- Description features: length, presence of URLs/mentions, whether description exists
- Location data: binary flag for having a location

**Post-Level Features:**
- Content statistics: average post length, URL/mention/hashtag percentages
- Temporal patterns: posting span, first/last post dates
- TF-IDF on aggregated post text (top 20 features per user)
  I spent way too long debugging why the TF-IDF features weren't merging properly with the user data.
- Language-specific features for French data:
  - Accent detection (bots often lack proper French accents)
  - French word recognition
  - Content repetition rates (bots spam similar content)
  - Hashtag abuse patterns

**Data Handling:**
- JSON parsing for nested user/post structures
- Multi-dataset combination with language tagging
- Missing value imputation and normalization

### Modeling

**Architecture:**
- Ensemble of XGBoost and LightGBM classifiers
- Class weighting to handle imbalanced bot/human ratios
- Hyperparameter tuning via RandomizedSearchCV (100 iterations)
- Custom threshold optimization for F1 score maximization

**Key Decisions:**
- Chose gradient boosting over neural networks for interpretability and speed
- Used ensemble averaging instead of stacking to reduce overfitting
- Implemented separate evaluation for English/French holdouts to catch language-specific issues
- Added threshold tuning because default 0.5 wasn't optimal for imbalanced data

## Results

**Holdout Performance:**
- English dataset: F1 ~0.85, precision/recall balanced
- French dataset: F1 ~0.82, struggled more with accent-based features
- Cross-validation on combined data: ~87% accuracy

**Competition Outcome:**
We placed respectably but not at the top. The model performed well on our metrics, but the competition scoring heavily penalized false positives. This taught us that evaluation metrics need to match the real-world cost structure.

## What Went Wrong / Key Lesson

Our biggest mistake was optimizing for F1 score without understanding the competition's scoring function. We spent weeks tuning hyperparameters to maximize recall and precision equally, but the final evaluation weighted false positives much more heavily than false negatives. Honestly, it stung a bit when we saw the results.

In hindsight, it makes sense—false positives (flagging real humans as bots) are more damaging than false negatives (missing some bots). But we didn't realize this until after submission. It was a valuable lesson in aligning technical metrics with business objectives. Next time, I'd spend more time reverse-engineering the evaluation criteria and less time on marginal F1 improvements.

## How to Run

### Setup
```bash
pip install -r requirements.txt
```

### Data Preparation
Place dataset files in the `data/` folder:
- `dataset.posts&users.1-6.json` (training data)
- `dataset.bots.1-6.txt` (bot labels)
- `dataset.posts&users.7.json` (English test)
- `dataset.posts&users.8.json` (French test)

### Training & Evaluation
```bash
python src/bot_detector.py
```

This will:
1. Train on labeled datasets 1-6
2. Evaluate on English/French holdouts
3. Generate final predictions on test sets
4. Save submission files

### Output Files
- `submission.csv`: All predictions with user IDs
- `bot_detector_output.txt`: Bot user IDs only
- `Egyptians.detections.en.txt`: English test predictions
- `Egyptians.detections.fr.txt`: French test predictions

## Repo Structure

```
├── src/
│   └── bot_detector.py          # Main detector class and training script
├── notebooks/
│   └── bot_detector.ipynb       # Jupyter notebook for exploration
├── data/                        # Dataset files (not included)
├── explore_data.py              # Data exploration utilities
├── test_synthetic.py            # Testing and validation scripts
├── translate_datasets.py        # French-to-English translation (optional)
├── requirements.txt             # Python dependencies
└── README.md                    # This file
```

## Future Improvements

**Technical:**
- Add transformer-based text embeddings for better NLP features
- Implement cross-language transfer learning
- Experiment with anomaly detection for unsupervised bot finding
- Add temporal features (posting patterns over time)

**Evaluation:**
- Always clarify scoring metrics upfront
- Implement multiple evaluation schemes during development
- Add cost-sensitive learning for imbalanced objectives

**Code Quality:**
- Refactor into proper Python modules
- Add comprehensive unit tests
- Implement proper logging and experiment tracking

This project showed me that hackathons are as much about problem understanding as they are about technical implementation. The ML work was solid, but we learned that the "human" part of machine learning—understanding requirements and evaluation—is just as critical.
- `extract_data()` builds `users` and `posts` DataFrames.
- `extract_features()` creates user and post-level features, including:
  - numeric account/profile features
  - description and username text features
  - post statistics such as URL rate, mention count, hashtag use
  - French-specific features such as accent usage and repeated content
  - aggregated TF-IDF features from post text

### Model and prediction
- The pipeline uses a tuned `xgboost` model plus a `lightgbm` ensemble.
- `find_optimal_threshold()` chooses the best decision threshold for F1 score.
- English and French final submission files are generated separately from the final test dataset paths configured in `src/bot_detector.py`.

## Submission instructions

Final submission files are produced in the repository root as:
- `Egyptians.detections.en.txt` for English
- `Egyptians.detections.fr.txt` for French

Each file contains one bot `id` per line, matching the required `dataset.bots.txt` format.

If you only submit one language, leave the other `test_*_files` list empty in `src/bot_detector.py`.

### Email content example
```
To: bot.or.not.competition.adm@gmail.com
Subject: Bot or Not Submission

Hi,

Please find our final submission below.

Team name: Egyptians
GitHub Repo: https://github.com/gmikhaie/Bot-or-Not-UACS-Competition
We are submitting for both English and French.
Attached are:

Egyptians.detections.en.txt
Egyptians.detections.fr.txt

Best,
Egyptians Team
```

## Notes
- The README explains how the repository works and how the code handles both English and French datasets.
- The final submission files are already generated and present in the repository root.

Overall, I'm proud of what we built. It wasn't perfect, but it was real ML work on a real problem. If you're reading this for a job, hit me up—I'd love to chat about building better systems.
