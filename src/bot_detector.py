import pandas as pd
import numpy as np
import warnings
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.feature_extraction.text import TfidfVectorizer
import xgboost as xgb
import lightgbm as lgb
import json
import os

# Reduce verbose xgboost messages, especially in loops
warnings.filterwarnings('ignore', category=UserWarning, module='xgboost')

class BotDetector:
    def __init__(self):
        self.model = None
        self.feature_cols = None
        self.tfidf = TfidfVectorizer(max_features=20, stop_words='english', ngram_range=(1, 2))
        self.tfidf_fitted = False
        self.lgb_model = None

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
            features['username_has_numbers'] = username.str.contains(r'\d', regex=True).astype(int)
            features['username_has_underscore'] = username.str.contains('_', regex=False).astype(int)

        # Location features
        if 'location' in users_df.columns:
            features['has_location'] = users_df['location'].notna().astype(int)
        
        # French-specific text features (for French language detection)
        is_french = (posts_df is not None and 'language' in posts_df.columns and 
                    (posts_df['language'] == 'french').any()) if posts_df is not None else False

        # If posts data is available, add post-based features
        if posts_df is not None and 'author_id' in posts_df.columns:
            posts_df = posts_df.copy()
            posts_df['text'] = posts_df['text'].fillna('')
            posts_df['text_len'] = posts_df['text'].str.len()
            posts_df['has_url'] = posts_df['text'].str.contains('http', regex=False).astype(int)
            posts_df['mention_count'] = posts_df['text'].str.count('@')
            posts_df['has_hashtag'] = posts_df['text'].str.contains('#', regex=False).astype(int)
            
            # French-specific features
            if is_french:
                posts_df['has_accents'] = posts_df['text'].str.contains(r'[àâäæéèêëïîôöœùûüç]', regex=True).astype(int)
                posts_df['accent_count'] = posts_df['text'].str.findall(r'[àâäæéèêëïîôöœùûüç]').str.len()
                posts_df['has_french_words'] = posts_df['text'].str.contains(
                    r'(?:le|la|les|de|des|un|une|et|ou|mais|avec|pour|sur|dans|que|qui|ça|c\'est)\b', 
                    regex=True, case=False
                ).astype(int)
                
                # Repeat content detection (strong French bot signal)
                posts_df['text_hash'] = posts_df['text'].apply(lambda x: hash(x) if isinstance(x, str) else 0)

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
            
            # Add French feature aggregations if present
            if is_french:
                french_agg = posts_df.groupby('author_id').agg(
                    pct_posts_with_accents=('has_accents', 'mean'),
                    avg_accent_count=('accent_count', 'mean'),
                    pct_posts_with_french_words=('has_french_words', 'mean'),
                    unique_posts=('text_hash', 'nunique'),  # Number of unique posts
                    total_posts=('text', 'count')
                ).reset_index()
                # Calculate content repetition rate
                french_agg['content_repetition_rate'] = 1 - (french_agg['unique_posts'] / french_agg['total_posts'])
                french_agg = french_agg.drop(columns=['unique_posts', 'total_posts'])
                post_stats = post_stats.merge(french_agg, on='author_id', how='left')

            post_stats['posting_span_days'] = (
                pd.to_datetime(post_stats['last_post']) - pd.to_datetime(post_stats['first_post'])
            ).dt.days.fillna(0)

            post_stats = post_stats.drop(columns=['first_post', 'last_post'])
            
            # Add hashtag abuse detection (strong French bot signal)
            post_stats['hashtag_abuse'] = (post_stats['pct_hashtag_posts'] > 0.35).astype(int)
            post_stats['hashtag_url_mismatch'] = ((post_stats['pct_hashtag_posts'] > 0.3) & (post_stats['pct_posts_with_url'] < 0.4)).astype(int)
            # Add TF-IDF text feature aggregation by author (up to 20 sparse textual features)
            author_text = posts_df.groupby('author_id')['text'].apply(' '.join).reset_index()
            if len(author_text) > 0:
                if not self.tfidf_fitted:
                    self.tfidf.fit(author_text['text'].fillna(''))
                    self.tfidf_fitted = True
                tfidf_matrix = self.tfidf.transform(author_text['text'].fillna(''))
                tfidf_feature_names = [f"tfidf_{i}_{w.replace(' ', '_')}" for i, w in enumerate(self.tfidf.get_feature_names_out())]
                tfidf_df = pd.DataFrame(tfidf_matrix.toarray(), columns=tfidf_feature_names, index=author_text['author_id'])
                tfidf_df = tfidf_df.reset_index().rename(columns={'index':'author_id'})
                post_stats = post_stats.merge(tfidf_df, on='author_id', how='left')
            # Merge with users
            features = features.merge(post_stats, left_on='id', right_on='author_id', how='left')
            features = features.drop('author_id', axis=1)


        # Fill missing values and ensure numeric
        features = features.fillna(0)
        # Fill any missing French feature columns with 0
        french_cols = ['pct_posts_with_accents', 'avg_accent_count', 'pct_posts_with_french_words', 'content_repetition_rate', 'hashtag_abuse', 'hashtag_url_mismatch']
        for col in french_cols:
            if col not in features.columns:
                features[col] = 0
        # Convert to numeric types
        for col in features.columns:
            if features[col].dtype == 'object':
                features[col] = pd.to_numeric(features[col], errors='coerce').fillna(0)
        return features

    def train(self, X, y):
        """Train the model"""
        # Calculate class weights to handle imbalanced data
        scale_pos_weight = (len(y) - y.sum()) / max(y.sum(), 1)
        
        self.model = xgb.XGBClassifier(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            random_state=42,
            verbosity=0,
            eval_metric='logloss',
            scale_pos_weight=scale_pos_weight  # Weight for imbalanced classes
        )
        self.model.fit(X, y)

    def tune_hyperparameters(self, X, y):
        """Tune XGBoost hyperparameters using randomized search for speed"""
        # Calculate class weights to handle imbalanced data
        scale_pos_weight = (len(y) - y.sum()) / max(y.sum(), 1)
        
        # Expanded parameter space for better tuning
        param_dist = {
            'max_depth': [3, 4, 5, 6, 7, 8, 9, 10],
            'learning_rate': [0.02, 0.03, 0.05, 0.1, 0.15, 0.2],
            'n_estimators': [100, 150, 200, 250, 300, 400],
            'subsample': [0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
            'colsample_bytree': [0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
            'min_child_weight': [1, 2, 3, 5],
            'gamma': [0, 0.1, 0.5, 1]  # L2 regularization on leaf weights
        }
        
        xgb_model = xgb.XGBClassifier(
            random_state=42, 
            verbosity=0, 
            eval_metric='logloss',
            scale_pos_weight=scale_pos_weight
        )
        
        # Increased from 50 to 100 iterations for better coverage
        grid_search = RandomizedSearchCV(
            estimator=xgb_model, 
            param_distributions=param_dist,
            n_iter=100,  # Doubled from 50
            cv=2,
            scoring='f1',
            n_jobs=-1, 
            verbose=0,
            random_state=42
        )
        grid_search.fit(X, y)
        self.model = grid_search.best_estimator_
        print(f"Best params: {grid_search.best_params_}")
        print(f"Best CV F1 score: {grid_search.best_score_:.3f}")
        
        # Train LightGBM model as ensemble
        self.lgb_model = lgb.LGBMClassifier(
            n_estimators=300,
            max_depth=5,
            learning_rate=0.1,
            num_leaves=31,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            verbose=-1,
            scale_pos_weight=scale_pos_weight
        )
        self.lgb_model.fit(X, y)
        
        return self.model

    def predict(self, X):
        """Make predictions"""
        return self.model.predict(X)
    
    def predict_proba(self, X):
        """Get probability predictions"""
        return self.model.predict_proba(X)
    
    def predict_with_threshold(self, X, threshold=0.5):
        """Make predictions with custom threshold for F1 optimization"""
        proba = self.model.predict_proba(X)[:, 1]  # Probability of being a bot
        return (proba >= threshold).astype(int)
    
    def predict_ensemble(self, X, threshold=0.5):
        """Ensemble prediction averaging XGBoost and LightGBM"""
        xgb_proba = self.model.predict_proba(X)[:, 1]
        if self.lgb_model is not None:
            lgb_proba = self.lgb_model.predict_proba(X)[:, 1]
            # Average the two models
            ensemble_proba = (xgb_proba + lgb_proba) / 2
        else:
            ensemble_proba = xgb_proba
        return (ensemble_proba >= threshold).astype(int)
    
    def find_optimal_threshold(self, X, y):
        """Find optimal threshold to maximize F1 score"""
        if self.lgb_model is not None:
            xgb_proba = self.model.predict_proba(X)[:, 1]
            lgb_proba = self.lgb_model.predict_proba(X)[:, 1]
            proba = (xgb_proba + lgb_proba) / 2
        else:
            proba = self.model.predict_proba(X)[:, 1]
        
        best_f1 = 0
        best_threshold = 0.5
        for threshold in np.arange(0.3, 0.8, 0.05):
            preds = (proba >= threshold).astype(int)
            f1 = f1_score(y, preds, zero_division=0)
            if f1 > best_f1:
                best_f1 = f1
                best_threshold = threshold
        
        return best_threshold

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
        
        # Ensure consistent column order
        french_cols = ['pct_posts_with_accents', 'avg_accent_count', 'pct_posts_with_french_words', 'content_repetition_rate', 'hashtag_abuse', 'hashtag_url_mismatch']
        other_cols = [col for col in X.columns if col not in french_cols]
        ordered_cols = other_cols + [col for col in french_cols if col in X.columns]
        X = X[ordered_cols]
        
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
        
        # Find optimal threshold for F1 (especially helps with French data)
        optimal_threshold = self.find_optimal_threshold(X_train, y_train)
        
        # Use ensemble predictions with optimal threshold
        preds = self.predict_ensemble(X_test, threshold=optimal_threshold)

        report = classification_report(y_test, preds, zero_division=0, output_dict=True)
        print(f"\\n=== {label} Evaluation ===")
        print(f"Train {len(train_users)} users | Test {len(test_users)} users")
        print(f"Optimal threshold: {optimal_threshold:.3f}")
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

    # -----------------------------
    # Dataset folder configuration
    # -----------------------------
    # REQUIRED FOR SUBMISSION: ensure dataset files are in this folder
    # Place your downloaded dataset JSON files here. Example paths:
    #  - /workspaces/Bot-or-Not-UACS-Competition/data/dataset.posts&users.1.json
    #  - /workspaces/Bot-or-Not-UACS-Competition/data/dataset.bots.1.txt
    # Adjust data_dir if your datasets are located elsewhere.
    data_dir = '../data/'  # first choice, if running from src/
    if not os.path.exists(data_dir):
        data_dir = 'data/'   # fallback, if already at repository root

    if not os.path.exists(data_dir):
        raise FileNotFoundError(f"Data directory not found: {data_dir}. place dataset files in data/ or ../data/")

    # REQUIRED FOR SUBMISSION: set your team name here (will prefix submission files)
    team_name = "Egyptians"  # Change this to your chosen team name

    all_files = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith('.json') and not ('7' in f or '8' in f)]

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
    skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)  # Reduced from 5 to 3 folds
    cv_scores = []

    for train_idx, test_idx in skf.split(X_all, y_all):
        X_train, X_test = X_all.iloc[train_idx], X_all.iloc[test_idx]
        y_train, y_test = y_all.iloc[train_idx], y_all.iloc[test_idx]

        # Calculate class weights
        scale_pos_weight = (len(y_train) - y_train.sum()) / max(y_train.sum(), 1)
        
        model = xgb.XGBClassifier(
            learning_rate=0.1, 
            max_depth=6, 
            n_estimators=200, 
            subsample=1.0, 
            random_state=42, 
            verbosity=0, 
            eval_metric='logloss',
            scale_pos_weight=scale_pos_weight
        )
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        cv_scores.append(accuracy_score(y_test, preds))

    print(f"\n=== Combined 3-fold CV Accuracy ===")
    print(f"Mean: {np.mean(cv_scores):.3f}, Std: {np.std(cv_scores):.3f}")

    # Train final model on all data for predictions
    detector.train(X_all, y_all)
    ids_all = all_users['id']
    
    # Find optimal threshold on all data and use ensemble for final predictions
    optimal_threshold = detector.find_optimal_threshold(X_all, y_all)
    predictions = detector.predict_ensemble(X_all, threshold=optimal_threshold)

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
    # REQUIRED FOR SUBMISSION: add paths to final evaluation test JSON files here
    # Example: test_en_files = ['data/final_test_en.json']
    # Example: test_fr_files = ['data/final_test_fr.json']
    test_en_files = ['data/dataset.posts&users.7.json']  # English test datasets
    test_fr_files = ['data/dataset.posts&users.8.json']  # French test datasets

    if test_en_files:
        print("\n=== Predicting on English Test Datasets ===")
        test_users, test_posts = [], []
        for f in test_en_files:
            users, posts = detector.extract_data(detector.load_data(f))
            posts['language'] = 'english'
            test_users.append(users)
            test_posts.append(posts)
        test_users = pd.concat(test_users, ignore_index=True)
        test_posts = pd.concat(test_posts, ignore_index=True)

        X_test, _, ids_test = detector.prepare_features(test_users, test_posts, target_col=None)
        test_predictions = detector.predict_ensemble(X_test, threshold=optimal_threshold)

        # Save submission file in required format
        en_output_file = f"{team_name}.detections.en.txt"
        with open(en_output_file, 'w', encoding='utf-8') as f:
            for user_id in ids_test[test_predictions == 1]:
                f.write(f"{user_id}\n")
        print(f"English detections saved to {en_output_file}")
        print(f"Predicted {test_predictions.sum()} bots out of {len(test_predictions)} English test users")

    if test_fr_files:
        print("\n=== Predicting on French Test Datasets ===")
        test_users, test_posts = [], []
        for f in test_fr_files:
            users, posts = detector.extract_data(detector.load_data(f))
            posts['language'] = 'french'
            test_users.append(users)
            test_posts.append(posts)
        test_users = pd.concat(test_users, ignore_index=True)
        test_posts = pd.concat(test_posts, ignore_index=True)

        X_test, _, ids_test = detector.prepare_features(test_users, test_posts, target_col=None)
        test_predictions = detector.predict_ensemble(X_test, threshold=optimal_threshold)

        # Save submission file in required format
        fr_output_file = f"{team_name}.detections.fr.txt"
        with open(fr_output_file, 'w', encoding='utf-8') as f:
            for user_id in ids_test[test_predictions == 1]:
                f.write(f"{user_id}\n")
        print(f"French detections saved to {fr_output_file}")
        print(f"Predicted {test_predictions.sum()} bots out of {len(test_predictions)} French test users")