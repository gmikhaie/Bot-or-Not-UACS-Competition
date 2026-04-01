import pandas as pd
import numpy as np
import warnings
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, classification_report
from sklearn.feature_extraction.text import TfidfVectorizer
import xgboost as xgb
import json
import os

# Reduce verbose xgboost messages, especially in loops
warnings.filterwarnings('ignore', category=UserWarning, module='xgboost')

class BotDetector:
    def __init__(self):
        self.model = None
        self.feature_cols = None

    def preprocess_data(self, df):
        """Preprocess the input dataframe"""
        # Handle missing values
        df = df.dropna()

        # gNormalize text
        if 'text' in df.columns:
            df['text'] = df['text'].str.lower().str.strip()
            df['text'] = df['text'].str.replace(r'[^\w\s]', '', regex=True)

        # Encode languae
        if 'language' in df.columns:
            df['language'] = df['language'].map({'english': 0, 'french': 1})

        return df

    def extract_features(self, users_df, posts_df=None):
        """Extract features from user and post data"""
        # Basic user features
        features = users_df[['id', 'tweet_count']].copy()

        # Optional features from users table
        if 'z_score' in users_df.columns:
            features['z_score'] = users_df['z_score']

        if 'followers_count' in users_df.columns:
            features['followers_count'] = users_df['followers_count']

        if 'following_count' in users_df.columns:
            features['following_count'] = users_df['following_count']

        # Text features from description
        if 'description' in users_df.columns:
            desc = users_df['description'].fillna('')
            features['description_length'] = desc.str.len()
            features['has_description'] = (desc != '').astype(int)
            features['description_has_url'] = desc.str.contains('http', regex=False).astype(int)
            features['description_has_mention'] = desc.str.contains('@').astype(int)

        # Username features
        if 'username' in users_df.columns:
            username = users_df['username'].fillna('')
            features['username_length'] = username.str.len()
            features['username_has_numbers'] = username.str.contains(r'\d').astype(int)
            features['username_has_underscore'] = username.str.contains('_').astype(int)

        # Location features
        if 'location' in users_df.columns:
            features['has_location'] = users_df['location'].notna().astype(int)

        # If posts data is available, add post-based features
        if posts_df is not None and 'author_id' in posts_df.columns:
            posts_df = posts_df.copy()
            posts_df['text'] = posts_df['text'].fillna('')
            posts_df['text_len'] = posts_df['text'].str.len()
            posts_df['has_url'] = posts_df['text'].str.contains('http', regex=False).astype(int)
            posts_df['mention_count'] = posts_df['text'].str.count('@')
            posts_df['has_hashtag'] = posts_df['text'].str.contains('#', regex=False).astype(int)

            post_stats = posts_df.groupby('author_id').agg(
                post_count=('text', 'count'),
                avg_post_length=('text_len', 'mean'),
                std_post_length=('text_len', 'std'),
                pct_posts_with_url=('has_url', 'mean'),
                avg_mentions=('mention_count', 'mean'),
                pct_hashtag_posts=('has_hashtag', 'mean'),
                first_post=('created_at', 'min'),
                last_post=('created_at', 'max')
            ).reset_index()

            post_stats['posting_span_days'] = (
                pd.to_datetime(post_stats['last_post']) - pd.to_datetime(post_stats['first_post'])
            ).dt.days.fillna(0)

            post_stats = post_stats.drop(columns=['first_post', 'last_post'])

            # Merge with users
            features = features.merge(post_stats, left_on='id', right_on='author_id', how='left')
            features = features.drop('author_id', axis=1)


        # Fill missing values and ensure numeric
        features = features.fillna(0)
        # Convert to numeric types
        for col in features.columns:
            if features[col].dtype == 'object':
                features[col] = pd.to_numeric(features[col], errors='coerce').fillna(0)
        return features

    def train(self, X, y):
        """Train the model"""
        self.model = xgb.XGBClassifier(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            random_state=42,
            verbosity=0,
            eval_metric='logloss'
        )
        self.model.fit(X, y)

    def tune_hyperparameters(self, X, y):
        """Tune XGBoost hyperparameters using grid search"""
        param_grid = {
            'max_depth': [3, 6, 9],
            'learning_rate': [0.01, 0.1, 0.2],
            'n_estimators': [50, 100, 200],
            'subsample': [0.8, 1.0]
        }
        xgb_model = xgb.XGBClassifier(random_state=42, verbosity=0, eval_metric='logloss')
        grid_search = GridSearchCV(estimator=xgb_model, param_grid=param_grid, cv=3, scoring='accuracy', n_jobs=-1, verbose=1)
        grid_search.fit(X, y)
        self.model = grid_search.best_estimator_
        print(f"Best params: {grid_search.best_params_}")
        print(f"Best CV score: {grid_search.best_score_:.3f}")
        return self.model

    def predict(self, X):
        """Make predictions"""
        return self.model.predict(X)

    def load_data(self, filepath):
        """Load data from JSON file"""
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return data

    def extract_data(self, data):
        """Extract users and posts from JSON data"""
        users_df = pd.DataFrame(data['users'])
        posts_df = pd.DataFrame(data['posts'])
        return users_df, posts_df

    def load_multiple_datasets(self, filepaths):
        """Load and combine multiple datasets"""
        all_users = []
        all_posts = []
        for filepath in filepaths:
            data = self.load_data(filepath)
            users_df, posts_df = self.extract_data(data)
            # Add language based on dataset number
            if any(num in filepath for num in ['4', '5', '6']):
                language = 'french'
            else:
                language = 'english'
            posts_df['language'] = language
            all_users.append(users_df)
            all_posts.append(posts_df)
        
        combined_users = pd.concat(all_users, ignore_index=True)
        combined_posts = pd.concat(all_posts, ignore_index=True)
        return combined_users, combined_posts

    def prepare_features(self, users_df, posts_df=None, target_col=None):
        """Prepare features and target"""
        features = self.extract_features(users_df, posts_df)
        
        # Separate id and features
        ids = features['id']
        X = features.drop('id', axis=1)
        self.feature_cols = X.columns.tolist()
        
        y = None
        if target_col and target_col in users_df.columns:
            y = users_df[target_col]
        
        return X, y, ids

    def evaluate(self, train_files, test_files, label):
        """Train on train_files and evaluate on test_files."""
        # load & label data
        train_users, train_posts = [], []
        for f in train_files:
            users, posts = self.extract_data(self.load_data(f))
            if 'dataset.bots' in f:
                pass
            # add language marker
            posts['language'] = 'french' if any(x in f for x in ['4', '5', '6']) else 'english'
            bot_path = f.replace('dataset.posts&users', 'dataset.bots').replace('.json', '.txt')
            bot_ids = self.load_bot_ids(bot_path)
            if bot_ids:
                users = self.apply_bot_labels(users, bot_ids)
            train_users.append(users)
            train_posts.append(posts)
        train_users = pd.concat(train_users, ignore_index=True)
        train_posts = pd.concat(train_posts, ignore_index=True)

        test_users, test_posts = [], []
        for f in test_files:
            users, posts = self.extract_data(self.load_data(f))
            posts['language'] = 'french' if any(x in f for x in ['4', '5', '6']) else 'english'
            bot_path = f.replace('dataset.posts&users', 'dataset.bots').replace('.json', '.txt')
            bot_ids = self.load_bot_ids(bot_path)
            users = self.apply_bot_labels(users, bot_ids)
            test_users.append(users)
            test_posts.append(posts)
        test_users = pd.concat(test_users, ignore_index=True)
        test_posts = pd.concat(test_posts, ignore_index=True)

        # train
        X_train, y_train, _ = self.prepare_features(train_users, train_posts, target_col='is_bot')
        self.tune_hyperparameters(X_train, y_train)

        # evaluate
        X_test, y_test, _ = self.prepare_features(test_users, test_posts, target_col='is_bot')
        preds = self.predict(X_test)

        report = classification_report(y_test, preds, zero_division=0, output_dict=True)
        print(f"\\n=== {label} Evaluation ===")
        print(f"Train {len(train_users)} users | Test {len(test_users)} users")
        print(f"Accuracy: {accuracy_score(y_test, preds):.3f}")
        print(f"Precision (bot): {report['1']['precision']:.3f}")
        print(f"Recall (bot): {report['1']['recall']:.3f}")
        print(f"F1 (bot): {report['1']['f1-score']:.3f}\n")

        return report

    def pseudo_label(self, users_df, z_score_threshold=0.0):
        """Create weak labels from z_score in absence of true labels"""
        if 'z_score' not in users_df.columns:
            raise ValueError('z_score not found in users_df; cannot create pseudo-labels')
        return (users_df['z_score'] > z_score_threshold).astype(int)

    def load_bot_ids(self, filepath):
        """Load known bot ids from dataset.bots.txt file"""
        if not os.path.exists(filepath):
            return set()
        with open(filepath, 'r', encoding='utf-8') as f:
            return {line.strip() for line in f if line.strip()}

    def apply_bot_labels(self, users_df, bot_ids):
        """Assign 'is_bot' labels from discovered bot ID set"""
        labels = users_df['id'].isin(bot_ids).astype(int)
        users_df = users_df.copy()
        users_df['is_bot'] = labels
        return users_df

# Example usage
if __name__ == "__main__":
    from sklearn.model_selection import StratifiedKFold

    detector = BotDetector()
    data_dir = '../data/'
    if not os.path.exists(data_dir):
        data_dir = 'data/'
    all_files = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith('.json')]

    # Holdout English test: train on 2-6, test on 1
    english_train = [f for f in all_files if '1' not in f]
    english_test = [f for f in all_files if '1' in f]
    detector.evaluate(english_train, english_test, 'English holdout')

    # Holdout French test: train on 1-3,5-6, test on 4
    french_train = [f for f in all_files if '4' not in f]
    french_test = [f for f in all_files if '4' in f]
    detector.evaluate(french_train, french_test, 'French holdout')

    # Combined cross-validation across all datasets
    all_users, all_posts = [], []
    for f in all_files:
        users, posts = detector.extract_data(detector.load_data(f))
        language = 'french' if any(x in f for x in ['4', '5', '6']) else 'english'
        posts['language'] = language
        bot_path = f.replace('dataset.posts&users', 'dataset.bots').replace('.json', '.txt')
        bot_ids = detector.load_bot_ids(bot_path)
        users = detector.apply_bot_labels(users, bot_ids)
        all_users.append(users)
        all_posts.append(posts)
    all_users = pd.concat(all_users, ignore_index=True)
    all_posts = pd.concat(all_posts, ignore_index=True)

    X_all, y_all, _ = detector.prepare_features(all_users, all_posts, target_col='is_bot')
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = []

    for train_idx, test_idx in skf.split(X_all, y_all):
        X_train, X_test = X_all.iloc[train_idx], X_all.iloc[test_idx]
        y_train, y_test = y_all.iloc[train_idx], y_all.iloc[test_idx]

        model = xgb.XGBClassifier(learning_rate=0.1, max_depth=6, n_estimators=200, subsample=1.0, random_state=42, verbosity=0, eval_metric='logloss')
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        cv_scores.append(accuracy_score(y_test, preds))

    print(f"\n=== Combined 5-fold CV Accuracy ===")
    print(f"Mean: {np.mean(cv_scores):.3f}, Std: {np.std(cv_scores):.3f}")

    # Train final model on all data for predictions
    detector.train(X_all, y_all)
    ids_all = all_users['id']
    predictions = detector.predict(X_all)

    submission = pd.DataFrame({
        'id': ids_all,
        'is_bot': predictions
    })

    # Save submission CSV
    submission.to_csv('submission.csv', index=False)
    print("Submission saved to submission.csv")

    # Save bot detector output (one user ID per line for flagged bots)
    bot_ids = ids_all[predictions == 1]
    with open('bot_detector_output.txt', 'w', encoding='utf-8') as f:
        for bot_id in bot_ids:
            f.write(f"{bot_id}\n")
    print("Bot detector output saved to bot_detector_output.txt")

    print(f"Predicted {predictions.sum()} bots out of {len(predictions)} users")

    # Optional: Predict on separate test datasets (if provided)
    test_files = []  # Add paths to test JSON files here, e.g. ['data/test1.json', 'data/test2.json']
    if test_files:
        print("\n=== Predicting on Test Datasets ===")
        test_users, test_posts = [], []
        for f in test_files:
            users, posts = detector.extract_data(detector.load_data(f))
            posts['language'] = 'french' if any(x in f for x in ['4', '5', '6']) else 'english'
            test_users.append(users)
            test_posts.append(posts)
        test_users = pd.concat(test_users, ignore_index=True)
        test_posts = pd.concat(test_posts, ignore_index=True)

        X_test, _, ids_test = detector.prepare_features(test_users, test_posts, target_col=None)
        test_predictions = detector.predict(X_test)

        test_submission = pd.DataFrame({
            'id': ids_test,
            'is_bot': test_predictions
        })

        # Save test submission
        test_submission.to_csv('test_submission.csv', index=False)
        print("Test submission saved to test_submission.csv")

        # Save test bot detector output
        test_bot_ids = ids_test[test_predictions == 1]
        with open('test_bot_detector_output.txt', 'w', encoding='utf-8') as f:
            for bot_id in test_bot_ids:
                f.write(f"{bot_id}\n")
        print("Test bot detector output saved to test_bot_detector_output.txt")

        print(f"Predicted {test_predictions.sum()} bots out of {len(test_predictions)} test users")