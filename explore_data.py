import json
import pandas as pd
import os

data_dir = 'data/'

# Load one English file
data = json.load(open(os.path.join(data_dir, 'dataset.posts&users.1.json'), 'r', encoding='utf-8'))

print("Keys in JSON:", data.keys())
print("Metadata:", data['metadata'])
print("Number of posts:", len(data['posts']))
print("Number of users:", len(data['users']))

users_df = pd.DataFrame(data['users'])
posts_df = pd.DataFrame(data['posts'])

print("Users columns:", users_df.columns.tolist())
print("Users head:")
print(users_df.head())
print("Posts columns:", posts_df.columns.tolist())
print("Posts head:")
print(posts_df.head())

# Check if there's a label
if 'is_bot' in users_df.columns:
    print("Has is_bot label")
else:
    print("No is_bot label found")

# Check z_score distribution
print("Z-score stats:")
print(users_df['z_score'].describe())