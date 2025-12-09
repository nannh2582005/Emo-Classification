# model_layer/naive.py

import os
import joblib
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


class NaiveBayesModel:
    def __init__(self, alpha=1.0, random_state=42):
        """
        Tạo mô hình Multinomial Naive Bayes
        alpha: hệ số smoothing (Laplace / Lidstone)
        random_state: để cố định việc chia train/test
        """
        self.alpha = alpha
        self.random_state = random_state

        self.model = MultinomialNB(alpha=self.alpha)

        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None

    def split_data(self, X, y, test_size=0.2):
        """
        Chia dữ liệu train/test.
        Dùng stratify=y để giữ tỉ lệ nhãn giữa train và test.
        """
        print(f"chia dữ liệu với tỉ lệ Test: {test_size*100:.0f}%...")
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y,
            test_size=test_size,
            random_state=self.random_state,
            stratify=y
        )
        print(f"  Train ({self.X_train.shape[0]}), Test ({self.X_test.shape[0]})")
        return self.X_train, self.X_test, self.y_train, self.y_test

    def train(self):
        """Huấn luyện mô hình Naive Bayes"""
        if self.X_train is None:
            raise ValueError("chạy split_data trước khi train!")
        self.model.fit(self.X_train, self.y_train)
        print("Huấn luyện mô hình Naive Bayes xong.")

    def predict(self, X):
        """Dự đoán nhãn cho dữ liệu mới"""
        return self.model.predict(X)

    def evaluate(self, target_names=None):
        """Đánh giá mô hình trên tập Test"""
        if self.X_test is None:
            raise ValueError("chưa gọi split_data, không có X_test để đánh giá.")

        print("\nĐánh giá mô hình Naive Bayes")
        y_pred = self.model.predict(self.X_test)

        acc = accuracy_score(self.y_test, y_pred)
        print(f"Accuracy: {acc:.4f}")

        print("\nBáo cáo phân loại:")
        print(classification_report(self.y_test, y_pred, target_names=target_names))

        print("\nMa trận nhầm lẫn (Confusion Matrix):")
        print(confusion_matrix(self.y_test, y_pred))

    def save_model(self, filepath="models/naive_bayes_model.pkl"):
        """Lưu mô hình ra file"""
        directory = os.path.dirname(filepath)
        if directory and not os.path.exists(directory):
            os.makedirs(directory)

        joblib.dump(self.model, filepath)
        print(f"Đã lưu mô hình Naive Bayes tại: {filepath}")

    def load_model(self, filepath="models/naive_bayes_model.pkl"):
        """Nạp mô hình từ file"""
        if os.path.exists(filepath):
            self.model = joblib.load(filepath)
            print(f"Đã nạp mô hình Naive Bayes từ: {filepath}")
        else:
            raise FileNotFoundError(f"Không tìm thấy file mô hình: {filepath}")
