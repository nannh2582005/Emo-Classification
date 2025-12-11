# main.py
import json
import os
from config import Config
from src.processor_layer.loader import DataLoader
from src.processor_layer.processor import DataProcessor
from src.feature_layer.tfidf import TFIDF
from src.model_layer.logistics import LogisticRegressionModel 
from src.model_layer.naive import NaiveBayesModel
from src.model_layer.svm import SVMModel
from src.visualization_layer.naive import NaiveVisualization
from src.visualization_layer.svm import SVMVisualization
from src.visualization_layer.logistics import LogisticVisualization
from src.optimize_layer.svm_optimize import SVMOptimizer
from src.optimize_layer.naive_optimize import NaiveBayesOptimizer
from src.optimize_layer.logistic_optimize import LogisticOptimizer


def main():

    # LOAD DATA
    print("\nload dữ liệu:")
    loader = DataLoader(Config.DATA_FILE)
    df = loader.load_data()
    # loader.review_data()

    # Load từ điển phụ trợ
    with open(Config.STOPWORDS_FILE, 'r', encoding='utf-8') as f:
        stopwords = [line.strip() for line in f if line.strip()]
    with open(Config.EMOJI_FILE, 'r', encoding='utf-8') as f:
        emoji_dict = json.load(f)
    with open(Config.TEENCODE_FILE, 'r', encoding='utf-8') as f:
        teencode_dict = json.load(f)

    # TIỀN XỬ LÝ (PREPROCESSING)
    print("\nTiền xử lí:")
    processor = DataProcessor(df, Config.TEXT_COLUMN, Config.LABEL_COLUMN, stopwords, emoji_dict, teencode_dict)
    processor.preprocess()
    
    # Lấy danh sách tên nhãn để in báo cáo cho đẹp (map từ id ngược lại tên)
    # id2label = {0: 'Tích cực', 1: 'Tiêu cực', 2: 'Trung tính'} (Ví dụ)
    label_names = [processor._id2label[i] for i in range(len(processor._id2label))]

    # TRÍCH ĐẶC TRƯNG (FEATURE EXTRACTION)
    print("\nVector hóa (TF-IDF):")
    feature = TFIDF(processor.df["processed"])
    X = feature.fit_transform(processor.df["processed"])
    y = processor.df["label_id"]
    
    print(f"Kích thước vector: {X.shape}")

    # HUẤN LUYỆN VÀ ĐÁNH GIÁ MÔ HÌNH (TRAIN MODEL)
    print("\nHuấn luyện mô hình và đánh giá:")
    
    
    
    # LogisticRegressionModel
    # Khởi tạo mô hình 
    
    
    
    log_reg = LogisticRegressionModel(X, y, config=Config)
    
    # Chia tập dữ liệu 
    log_reg.split_data()
    
   
    # B1: Tối ưu tham số
    log_optimizer = LogisticOptimizer(X, y, config=Config)
    best_params_log, best_score_log = log_optimizer.optimize(cv=5)
    print(f"Logistic Best Params: {best_params_log}")
    print(f"Logistic Best Score: {best_score_log:.4f}")

    # B2: Train model với tham số tốt nhất
    log_reg = log_optimizer.train_best_model()
    
    # Đánh giá
    log_reg.evaluate(target_names=label_names)
    
    # Lưu mô hình
    log_reg.save_model("models_saver/logistic_sentiment.pkl")


    log_viz = LogisticVisualization(save_dir="images")
    log_viz.visualize(log_reg, target_names=label_names)
    # =====================================================
    print("\n=== Tối ưu và train Naive Bayes ===")

    # Khởi tạo optimizer
    nb_optimizer = NaiveBayesOptimizer(X, y, config=Config)

    # Tìm tham số tối ưu
    nb_best_params, nb_best_score = nb_optimizer.optimize(cv=5)
    print("Naive Bayes best params:", nb_best_params)
    print("Naive Bayes best score:", nb_best_score)

    # Train mô hình NB với tham số tối ưu
    nb_model = nb_optimizer.train_best_model()

    # Evaluate mô hình
    nb_model.evaluate(verbose=True)

    # Save model
    nb_model.save("naive_sentiment.pkl")

    # Visualization
    nb_viz = NaiveVisualization(save_dir="images")
    nb_viz.visualize(nb_model, target_names=label_names)

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
    
    
    
    ####
    # QUAN TRỌNG: lưu TF-IDF Vectorizer để chạy app
    import joblib
    print(" lưu TF-IDF Vectorizer để dùng cho app")
    # tfidf.vectorizer là đối tượng của sklearn
    joblib.dump(feature.vectorizer, "models_saver/tfidf_vectorizer.pkl") 
    print("lưu vectorizer")


if __name__ == "__main__":
    main()
