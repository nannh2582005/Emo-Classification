# main.py
import json
import os
from config import Config
from processor_layer.loader import DataLoader
from processor_layer.processor import DataProcessor
from feature_layer.tfidf import TFIDF
from model_layer.logistics import LogisticRegressionModel # Import module mới
from model_layer.naive import NaiveBayesModel
from model_layer.svm import SVMModel
from visualization.naive import NaiveVisualization


def main():
    # 1. THIẾT LẬP ĐƯỜNG DẪN
    data_path = r"data/data.xlsx" 
    stopwords_path = r"data/vietnamese-stopwords.txt"
    emoji_path = r"data/emoji_dict.json"
    teencode_path = r"data/teencode_dict.json"
    
    # 2. LOAD DATA
    print(" BƯỚC 1: load dữ liệu")
    loader = DataLoader(data_path)
    df = loader.load_data()
    # loader.review_data()

    # Load từ điển phụ trợ
    with open(stopwords_path, 'r', encoding='utf-8') as f:
        stopwords = [line.strip() for line in f if line.strip()]
    with open(emoji_path, 'r', encoding='utf-8') as f:
        emoji_dict = json.load(f)
    with open(teencode_path, 'r', encoding='utf-8') as f:
        teencode_dict = json.load(f)

    # 3. TIỀN XỬ LÝ (PREPROCESSING)
    print("\n BƯỚC 2: tiền xử lí")
    processor = DataProcessor(df, "Sentence", "Emotion", stopwords, emoji_dict, teencode_dict)
    processor.preprocess()
    
    # Lấy danh sách tên nhãn để in báo cáo cho đẹp (map từ id ngược lại tên)
    # id2label = {0: 'Tích cực', 1: 'Tiêu cực', 2: 'Trung tính'} (Ví dụ)
    label_names = [processor._id2label[i] for i in range(len(processor._id2label))]

    # 4. TRÍCH ĐẶC TRƯNG (FEATURE EXTRACTION)
    print("\n BƯỚC 3: vec to hoá (TF-IDF) ")
    feature = TFIDF(processor.df["processed"])
    X = feature.fit_transform(processor.df["processed"])
    y = processor.df["label_id"]
    
    print(f"Kích thước vector: {X.shape}")

    # 5. HUẤN LUYỆN VÀ ĐÁNH GIÁ MÔ HÌNH (TRAIN MODEL)
    print("\n Bước 4: đánh giá")
    
    # Khởi tạo mô hình
    print("\nHuấn luyện Logistic Regression")
    log_reg = LogisticRegressionModel(random_state=42)
    
    # Chia tập dữ liệu 
    log_reg.split_data(X, y, test_size=0.2)
    
    # Train 
    log_reg.train()
    
    # Đánh giá ( kieenimplement method trong class để dang gọi)
    log_reg.evaluate(target_names=label_names)
    
    # Lưu mô hình 
    log_reg.save_model("models/logistic_sentiment.pkl")

    print("\nHuấn luyện Naive Bayes")
    # Khởi tạo mô hình
    nb = NaiveBayesModel(alpha=1.0, random_state=42)

    # Chia tập dữ liệu
    nb.split_data(X, y, test_size=0.2)

    # Train
    nb.train()

    # Đánh giá
    nb.evaluate(target_names=label_names)

    # Lưu mô hình
    nb.save_model("models/naive_sentiment.pkl")
    
    print("\nHuấn luyện SVM")
    svm = SVMModel(X, y, config=Config)

    # Chia dữ liệu
    svm.split_data()

    # Train
    svm.train()

    # Evaluate
    svm.evaluate(verbose=True)

    # Save model
    svm.save("svm_sentiment.pkl")

    print("Đã lưu mô hình SVM!")

    # 6. TRỰC QUAN MÔ HÌNH (VISUALIZE MODEL)
    # Naive Bayes

    # Tạo lớp trực quan hóa
    viz = NaiveVisualization()  

    # Vẽ và lưu hình
    viz.visualize(nb, target_names=label_names)




if __name__ == "__main__":
    main()