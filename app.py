import os
import joblib
import json
import pandas as pd
import numpy as np
from config import Config
from processor_layer.processor import DataProcessor

class SentimentApp:
    def __init__(self):
        print("đợi xí")
        self.load_resources()
        print("ok r nè")

    def load_resources(self):
        """Load các tài nguyên cần thiết: Model, Vectorizer, Dicts"""
        # 1. Load các từ điển xử lý văn bản
        with open(Config.STOPWORDS_FILE, 'r', encoding='utf-8') as f:
            self.stopwords = [line.strip() for line in f if line.strip()]
        with open(Config.EMOJI_FILE, 'r', encoding='utf-8') as f:
            self.emoji_dict = json.load(f)
        with open(Config.TEENCODE_FILE, 'r', encoding='utf-8') as f:
            self.teencode_dict = json.load(f)

        # 2. Load TF-IDF Vectorizer đã train
        vect_path = os.path.join(Config.MODEL_DIR, "tfidf_vectorizer.pkl")
        if not os.path.exists(vect_path):
            raise FileNotFoundError(f"Không thấy {vect_path}. CHẠY chạy main.py trước")
        self.vectorizer = joblib.load(vect_path)

        # 3. Load Model ( chọn SVM  thường tốt nhất)
        #  có thể đổi thành 'logistic_sentiment.pkl' hoặc 'naive_sentiment.pkl'
        

        target_model_name = "svm_sentiment.pkl" # Đặt tên file vào biến này
        model_path = os.path.join(Config.MODEL_DIR, target_model_name)
        
        if not os.path.exists(model_path):
             raise FileNotFoundError(f"Không tìm thấy {model_path}. Hãy chạy main.py trước!")
        
        saved_data = joblib.load(model_path)
        self.model = saved_data['model'] 
        
        # Sửa dòng print bị lỗi thành:
        print(f"--- Đã load model thành công: {target_model_name} ---")

    def predict(self, text):
        """Dự đoán cảm xúc cho một câu"""
        # B1: Tạo DataFrame giả để dùng lại class DataProcessor
        df_temp = pd.DataFrame({'Sentence': [text], 'Emotion': ['Unknown']})
        
        # B2: Tiền xử lý
        processor = DataProcessor(
            df_temp, 'Sentence', 'Emotion', 
            self.stopwords, self.emoji_dict, self.teencode_dict
        )
        processor.preprocess()
        
        
        # Lấy text đã xử lý (truy cập trực tiếp vào cột 'processed')
        # Vì processor._df là DataFrame,  lấy dòng đầu tiên của cột 'processed'
        processed_text = processor._df['processed'].iloc[0]
        
        
        # B3: Vector hóa (Transform)
        # Lưu ý: Phải bỏ vào list lồng nhau [[w1, w2...]]
        X_vec = self.vectorizer.transform([processed_text])

        # B4: Dự đoán
        label_pred = self.model.predict(X_vec)[0]
        
        # Nếu model có hỗ trợ tính xác suất
        try:
            proba = np.max(self.model.predict_proba(X_vec)) * 100
            return label_pred, proba
        except:
            return label_pred, None

# --- CHẠY ỨNG DỤNG ---
if __name__ == "__main__":
    app = SentimentApp()
    
    print("="*50)
    print("Nói j đok ik kưng")
    print("Nhập 'exit' hoặc 'quit' để thoát.")
    print("="*50)

    while True:
        text = input("\n hai ba say it!!: ")
        if text.lower() in ['exit', 'quit']:
            break
        
        if not text.strip():
            continue

        label, conf = app.predict(text)
        
        print(f"--> Cảm xúc: {label}")
        if conf:
            print(f"--> Độ tin cậy: {conf:.2f}%")