# Bot-or-Not-UACS-Competition

## Competition Overview

Welcome to the Bot or Not Competition! The goal is to create the best bot accounts detector. There are prizes for:
- Best detector in English
- Best detector in French  
- Best detector for both languages

Note: A team can only win one prize. If a team qualifies for multiple, they get the highest prize, and the next best gets the lower one.

Testing datasets (2-3 per language) will be posted tomorrow.

For more details, see `Bot_or_Not_Competition.pdf` (available in the resources channel).

Questions: Post in the questions channel or email bot.or.not.competition.adm@gmail.com

## Project Setup

This project uses Python for machine learning-based bot detection.

### Requirements
- Python 3.8+
- Libraries: pandas, scikit-learn, numpy, matplotlib, jupyter

### Getting Started
1. Install dependencies: `pip install -r requirements.txt`
2. Place dataset files in the `data/` folder
3. Run the bot detector: `python src/bot_detector.py`
4. The submission file `submission.csv` will be generated

## Model Details
- **Features**: tweet_count, description_length, username_length, post statistics
- **Algorithm**: Random Forest Classifier
- **Training**: Uses z_score > 0 as bot label (adjust if needed)
- **Performance**: ~98% accuracy on validation set

## Competition Notes
- Datasets contain user profiles and posts
- z_score appears to be a bot probability score
- Model predicts binary bot/not-bot classification
- Submission format: CSV with 'id' and 'is_bot' columns

## Team
Make sure to update your server name to your full name!