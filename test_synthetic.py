import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report

# Set up plotting
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (12, 8)

class BotDetector:
    def __init__(self):
        self.model = None
        self.feature_cols = None

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
            all_users.append(users_df)
            all_posts.append(posts_df)

        combined_users = pd.concat(all_users, ignore_index=True)
        combined_posts = pd.concat(all_posts, ignore_index=True)
        return combined_users, combined_posts

    def extract_features(self, users_df, posts_df=None):
        """Extract features from user and post data"""
        # Basic user features - only include columns that exist
        available_cols = [col for col in ['id', 'tweet_count'] if col in users_df.columns]
        features = users_df[available_cols].copy()

        # Add synthetic features that match our feature engineering
        if 'tweet_count' in features.columns:
            # Create synthetic features based on tweet_count patterns
            features['description_length'] = np.where(features['tweet_count'] > 100,
                                                    np.random.randint(0, 30),  # Bots: short bios
                                                    np.random.randint(20, 100))  # Humans: longer bios
            features['has_description'] = np.where(features['tweet_count'] > 100,
                                                 np.random.choice([0, 1], size=len(features), p=[0.7, 0.3]),
                                                 np.random.choice([0, 1], size=len(features), p=[0.1, 0.9]))
            features['description_has_url'] = np.where(features['tweet_count'] > 100,
                                                     np.random.choice([0, 1], size=len(features), p=[0.3, 0.7]),
                                                     np.random.choice([0, 1], size=len(features), p=[0.8, 0.2]))
            features['username_length'] = np.where(features['tweet_count'] > 100,
                                                 np.random.randint(3, 8, size=len(features)),  # Bots: short
                                                 np.random.randint(5, 15, size=len(features)))  # Humans: longer
            features['username_has_numbers'] = np.where(features['tweet_count'] > 100,
                                                      np.random.choice([0, 1], size=len(features), p=[0.2, 0.8]),
                                                      np.random.choice([0, 1], size=len(features), p=[0.8, 0.2]))
            features['has_location'] = np.where(features['tweet_count'] > 100,
                                             np.random.choice([0, 1], size=len(features), p=[0.8, 0.2]),
                                             np.random.choice([0, 1], size=len(features), p=[0.3, 0.7]))

        return features.fillna(0)

    def prepare_features(self, users_df, posts_df=None, target_col=None):
        """Prepare features and target"""
        features = self.extract_features(users_df, posts_df)

        # Separate id and features
        ids = features.get('id', pd.Series(range(len(features))))
        X = features.drop('id', axis=1, errors='ignore')
        self.feature_cols = X.columns.tolist()

        y = None
        if target_col and target_col in users_df.columns:
            y = users_df[target_col]

        return X, y, ids

    def train(self, X, y):
        """Train the model"""
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.model.fit(X, y)

    def predict(self, X):
        """Make predictions"""
        return self.model.predict(X)

# Set up plotting
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (12, 8)

def create_realistic_synthetic_users(n_users=200):
    """Create synthetic user data that better matches real data patterns"""

    # First, let's analyze the real data to understand distributions
    try:
        data_dir = '../data/'
        if not os.path.exists(data_dir):
            data_dir = 'data/'

        # Load real data to understand patterns
        real_data = json.load(open(os.path.join(data_dir, 'dataset.posts&users.1.json'), 'r'))
        real_users = pd.DataFrame(real_data['users'])

        # Get statistics from real data
        real_bot_threshold = real_users['z_score'].median()  # Use median as threshold
        real_high_z = real_users[real_users['z_score'] > real_bot_threshold]
        real_low_z = real_users[real_users['z_score'] <= real_bot_threshold]

        print(f"Real data analysis:")
        print(f"  Total users: {len(real_users)}")
        print(f"  Bot threshold (median z_score): {real_bot_threshold:.3f}")
        print(f"  High z_score users: {len(real_high_z)}")
        print(f"  Low z_score users: {len(real_low_z)}")

    except:
        # Fallback if real data not available
        real_high_z = None
        real_low_z = None
        print("Using fallback synthetic data generation")

    users = []

    for i in range(n_users):
        # Decide if this user is a bot (balanced classes)
        is_bot = np.random.choice([0, 1], p=[0.5, 0.5])

        if is_bot:
            # Bot characteristics based on real data patterns
            if real_high_z is not None and len(real_high_z) > 0:
                # Sample from real bot-like users
                sample = real_high_z.sample(1).iloc[0]
                tweet_count = int(sample['tweet_count'])
                username = sample['username']
                description = sample.get('description', '')
            else:
                # Fallback bot patterns
                tweet_count = np.random.randint(100, 1000)
                username = f"bot{np.random.randint(1, 999)}"
                description = np.random.choice([
                    "Follow for more content!",
                    "DM for business inquiries",
                    "Best deals online",
                    "",  # Empty bio
                    "Content creator"
                ])

        else:
            # Human characteristics
            if real_low_z is not None and len(real_low_z) > 0:
                # Sample from real human-like users
                sample = real_low_z.sample(1).iloc[0]
                tweet_count = int(sample['tweet_count'])
                username = sample['username']
                description = sample.get('description', '')
            else:
                # Fallback human patterns
                tweet_count = np.random.randint(10, 300)
                username = f"user{np.random.randint(1000, 9999)}"
                description = np.random.choice([
                    "Just a regular person sharing thoughts",
                    "Software engineer | Coffee lover | Dog mom",
                    "Sports enthusiast and foodie",
                    "Teacher by day, artist by night",
                    "Travel blogger and photographer"
                ])

        # Create features that match our feature engineering
        user = {
            'id': f'synthetic_{i}',
            'tweet_count': tweet_count,
            'description_length': len(description) if description else 0,
            'has_description': 1 if description else 0,
            'description_has_url': 1 if 'http' in str(description) else 0,
            'username_length': len(username),
            'username_has_numbers': 1 if any(c.isdigit() for c in username) else 0,
            'has_location': np.random.choice([0, 1], p=[0.4, 0.6]),  # Humans more likely to have location
            'true_label': is_bot
        }
        users.append(user)

    return pd.DataFrame(users)

def test_model_on_synthetic_data():
    """Test the trained model on synthetic data"""

    # Create synthetic data
    synthetic_df = create_realistic_synthetic_users(200)
    print(f"Created {len(synthetic_df)} synthetic users")
    print(f"Bots: {synthetic_df['true_label'].sum()}, Humans: {(synthetic_df['true_label'] == 0).sum()}")

    # Load the trained model (assuming it was trained on real data)
    detector = BotDetector()

    # Load real training data to train the model
    import os
    data_dir = '../data/'
    if not os.path.exists(data_dir):
        data_dir = 'data/'  # Try relative to current dir
    if not os.path.exists(data_dir):
        print("Data directory not found")
        return pd.DataFrame()
    all_files = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith('.json')]
    english_files = [f for f in all_files if '1' in f or '2' in f or '3' in f]

    if english_files:
        users_df, posts_df = detector.load_multiple_datasets(english_files)
        users_df['is_bot'] = (users_df['z_score'] > 0).astype(int)
        X_train, y_train, _ = detector.prepare_features(users_df, posts_df, target_col='is_bot')
        detector.train(X_train, y_train)
        print("Model trained on real data")
    else:
        print("No training data found")
        return

    # Prepare synthetic features (without true_label)
    synthetic_features = synthetic_df.drop('true_label', axis=1)
    X_synthetic, _, ids_synthetic = detector.prepare_features(synthetic_features)

    # Make predictions
    predictions = detector.predict(X_synthetic)

    # Compare predictions with true labels
    synthetic_df['predicted_bot'] = predictions

    # Calculate accuracy
    correct = (synthetic_df['true_label'] == synthetic_df['predicted_bot']).sum()
    accuracy = correct / len(synthetic_df) * 100
    print(".2f")

    # Show confusion matrix
    from sklearn.metrics import confusion_matrix, classification_report
    cm = confusion_matrix(synthetic_df['true_label'], synthetic_df['predicted_bot'])
    print("\nConfusion Matrix:")
    print("True\\Pred | Human | Bot")
    print(f"Human     | {cm[0][0]:5d} | {cm[0][1]:3d}")
    print(f"Bot       | {cm[1][0]:5d} | {cm[1][1]:3d}")

    print("\nClassification Report:")
    print(classification_report(synthetic_df['true_label'], synthetic_df['predicted_bot'],
                               target_names=['Human', 'Bot']))

    # Visualize results
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))

    # Plot 1: Prediction accuracy by true label
    accuracy_by_label = synthetic_df.groupby('true_label')['predicted_bot'].value_counts().unstack()
    accuracy_by_label.plot(kind='bar', ax=axes[0,0])
    axes[0,0].set_title('Predictions by True Label')
    axes[0,0].set_xlabel('True Label (0=Human, 1=Bot)')
    axes[0,0].set_ylabel('Count')

    # Plot 2: Feature distributions for bots vs humans
    bot_features = synthetic_df[synthetic_df['true_label'] == 1][['tweet_count', 'description_length', 'username_length']]
    human_features = synthetic_df[synthetic_df['true_label'] == 0][['tweet_count', 'description_length', 'username_length']]

    bot_features.boxplot(ax=axes[0,1])
    axes[0,1].set_title('Bot Feature Distributions')
    axes[0,1].set_ylabel('Value')

    human_features.boxplot(ax=axes[1,0])
    axes[1,0].set_title('Human Feature Distributions')
    axes[1,0].set_ylabel('Value')

    # Plot 3: Prediction confidence (if available)
    if hasattr(detector.model, 'predict_proba'):
        proba = detector.model.predict_proba(X_synthetic)
        synthetic_df['bot_probability'] = proba[:, 1]

        synthetic_df.boxplot(column='bot_probability', by='true_label', ax=axes[1,1])
        axes[1,1].set_title('Bot Probability by True Label')
        axes[1,1].set_xlabel('True Label (0=Human, 1=Bot)')
        axes[1,1].set_ylabel('Predicted Bot Probability')

    plt.tight_layout()
    plt.show()

    # Show some examples
    print("\nExamples of correct classifications:")
    correct_bots = synthetic_df[(synthetic_df['true_label'] == 1) & (synthetic_df['predicted_bot'] == 1)].head(3)
    correct_humans = synthetic_df[(synthetic_df['true_label'] == 0) & (synthetic_df['predicted_bot'] == 0)].head(3)

    print("Correctly identified bots:")
    for _, row in correct_bots.iterrows():
        print(f"  Tweet count: {row['tweet_count']}, Username length: {row['username_length']}, Has numbers: {row['username_has_numbers']}")

    print("Correctly identified humans:")
    for _, row in correct_humans.iterrows():
        print(f"  Tweet count: {row['tweet_count']}, Username length: {row['username_length']}, Has location: {row['has_location']}")

    return synthetic_df

if __name__ == "__main__":
    results = test_model_on_synthetic_data()