# Bot-or-Not-UACS-Competition

## Competition Overview

This repository contains the solution for the Bot or Not competition. It is designed to train a bot detector on the provided labeled datasets and then generate final submission files for English and French test datasets.

The current team name is `Egyptians`, and final detection files are saved as:
- `Egyptians.detections.en.txt`
- `Egyptians.detections.fr.txt`

## Project Setup

### Requirements
- Python 3.8+
- Install dependencies with:
  ```bash
  pip install -r requirements.txt
  ```

### Dataset placement
- Place the provided dataset files in the `data/` folder.
- Example required files:
  - `data/dataset.posts&users.1.json`
  - `data/dataset.bots.1.txt`
  - `data/dataset.posts&users.7.json`
  - `data/dataset.posts&users.8.json`
- Do not keep ZIP archive files in `data/`; only the JSON/TXT files are needed.

### Running the detector
From the repository root, run:
```bash
python src/bot_detector.py
```

This script will:
1. Train on the existing labeled datasets (`dataset.posts&users.1-6.json` and `dataset.bots.1-6.txt`).
2. Evaluate English and French holdout splits.
3. Train a final model on all labeled data.
4. Generate:
   - `submission.csv`
   - `bot_detector_output.txt`
   - `Egyptians.detections.en.txt`
   - `Egyptians.detections.fr.txt`

## How the code works

The main detector is implemented in `src/bot_detector.py`.

### Data flow
- `load_data()` reads JSON dataset files.
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
- The README now explains how the repository works and how the code handles both English and French datasets.
- The final submission files are already generated and present in the repository root.
