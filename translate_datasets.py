import json
import os
from deep_translator import GoogleTranslator

def translate_datasets():
    data_dir = 'data/'
    translator = GoogleTranslator(source='fr', target='en')
    
    for filename in os.listdir(data_dir):
        if filename.endswith('.json') and any(num in filename for num in ['4', '5', '6']):  # French datasets
            filepath = os.path.join(data_dir, filename)
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Translate posts
            for post in data['posts']:
                if post['text']:
                    try:
                        post['text'] = translator.translate(post['text'][:500])
                    except:
                        pass  # Keep original if fails
            
            # Save translated version
            translated_path = filepath.replace('.json', '_translated.json')
            with open(translated_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            print(f"Translated {filename} -> {os.path.basename(translated_path)}")

if __name__ == "__main__":
    translate_datasets()