import json

from processor_layer.loader import DataLoader
from processor_layer.processor import DataProcessor
from feature_layer.tfidf import TFIDF

def main():
    # đường dẫn dữ liệu (có thể thay đổi)
    path = r"data/data.xlsx" 

    data = DataLoader(path)
    df = data.load_data()
    print(df.head(5))
    with open(r'data/vietnamese-stopwords.txt', 'r', encoding='utf-8') as f:
        stopwords = [line.strip() for line in f if line.strip()]
    with open(r'data/emoji_dict.json','r', encoding='utf-8') as f:
        emoji_dict = json.load(f)
    with open(r'data/teencode_dict.json', 'r', encoding='utf-8') as f:
        teencode_dict = json.load(f)
if __name__ == "__main__":
    main()
