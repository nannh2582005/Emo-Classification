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
from visualization.svm import SVMVisualization
from optimize.svm_optimize import SVMOptimizer

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
    log_reg = LogisticRegressionModel(random_state=42)
    
    # Chia tập dữ liệu 
    log_reg.split_data(X, y, test_size=0.2)
    
    # Train 
    log_reg.train()
    
    # Đánh giá ( Kiên implement method trong class để dang gọi)
    log_reg.evaluate(target_names=label_names)
    
    # Lưu mô hình ( Kiên implement method)
    log_reg.save_model("models/logistic_sentiment.pkl")

    # Khởi tạo mô hình Naive Bayes
    nb = NaiveBayesModel(alpha=1.0, random_state=42)

    # Chia tập dữ liệu
    nb.split_data(X, y, test_size=0.2)

    # Huấn luyện mô hình Naive Bayes
    nb.train()

    # Đánh giá mô hình Naive Bayes
    nb.evaluate(target_names=label_names)

    # Lưu mô hình Naive Bayes
    nb.save_model("models/naive_sentiment.pkl")

    # 6. TRỰC QUAN MÔ HÌNH (VISUALIZE MODEL)
    # Naive Bayes

    # Tạo lớp trực quan hóa
    viz = NaiveVisualization()  

    # Vẽ và lưu hình
    viz.visualize(nb, target_names=label_names)

    print("\n=== Tối ưu và train SVM ===")
    svm_optimizer = SVMOptimizer(X, y, config=Config)

    # Tìm tham số tốt nhất
    best_params, best_score = svm_optimizer.optimize(cv=5)
    print("SVM best params:", best_params)
    print("SVM best score:", best_score)

    # Train mô hình SVM với tham số tối ưu và lấy đối tượng SVMModel
    svm_model = svm_optimizer.train_best_model()

    # Evaluate và visualize
    svm_model.evaluate(verbose=True)

    # Save
    svm_model.save("svm_sentiment.pkl")

    svm_viz = SVMVisualization(save_dir="images")
    svm_viz.visualize(svm_model, target_names=label_names)


if __name__ == "__main__":
    main()
