# model_layer/naive.py

import os
import joblib
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from logs.logger import setup_logger
from config import Config
from .base import DataModel


class NaiveBayesModel(DataModel):
    """Naive Bayes cho phân loại cảm xúc."""

    def __init__(self, X, y, config: Config = Config):
        super().__init__(X, y, test_size=config.TEST_SIZE, random_state=config.RANDOM_STATE)
        self.config = config

        # Tham số mô hình
        self.alpha = config.NAIVE_DEFAULT.get("alpha", 1.0)

        # Khởi tạo model
        self._model = MultinomialNB(alpha=self.alpha)

        # Logger
        self.logger = setup_logger(self.__class__.__name__)
        self.logger.info("Khởi tạo NaiveBayesModel với alpha=%s", self.alpha)

    @property
    def X_test(self):
        return self._X_test

    @property
    def y_test(self):
        return self._y_test

    def train(self):
        """Train mô hình Naive Bayes."""
        self.logger.info("Bắt đầu train mô hình Naive Bayes")

        # chia data 
        self.split_data()
        self.logger.info(
            "Train size: %d | Test size: %d",
            len(self._y_train), len(self._y_test)
        )

        # train model
        self._model.fit(self._X_train, self._y_train)
        self.logger.info("Huấn luyện mô hình Naive Bayes hoàn tất!")

    def predict(self, X):
        """Dự đoán nhãn cho dữ liệu mới."""
        n_samples = X.shape[0]
        self.logger.info("Dự đoán %d mẫu.", n_samples)
        return self._model.predict(X)

    def evaluate(self, verbose=True):
        """Đánh giá mô hình trên tập test."""
        self.logger.info("Bắt đầu đánh giá mô hình Naive Bayes")

        y_pred = self._model.predict(self._X_test)

        acc = accuracy_score(self._y_test, y_pred)
        self.logger.info("Accuracy: %.4f", acc)

        report = classification_report(self._y_test, y_pred, output_dict=True)

        # log F1 mỗi lớp
        f1_dict = {k: v.get("f1-score") for k, v in report.items() if isinstance(v, dict)}
        self.logger.info("F1 mỗi lớp: %s", f1_dict)

        if verbose:
            print("=== Naive Bayes Classification Report ===")
            print(classification_report(self._y_test, y_pred))
            print("\nConfusion Matrix:")
            print(confusion_matrix(self._y_test, y_pred))

        return report

    def save(self, filename="naive_bayes.pkl"):
        """Lưu mô hình xuống thư mục models."""
        if not os.path.exists(self.config.MODEL_DIR):
            os.makedirs(self.config.MODEL_DIR)

        path = os.path.join(self.config.MODEL_DIR, filename)

        joblib.dump({
            "model": self._model,
            "alpha": self.alpha,
            "config": self.config.NAIVE_DEFAULT
        }, path)

        self.logger.info("Đã lưu mô hình vào: %s", path)
        return path

    def load_model(self, filepath="models_saver/naive_bayes.pkl"):
        """Load mô hình Naive Bayes đã lưu."""
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Không tìm thấy file mô hình: {filepath}")

        data = joblib.load(filepath)

        self._model = data["model"]
        self.alpha = data.get("alpha", 1.0)

        self.logger.info("Đã load mô hình từ %s | alpha=%s", filepath, self.alpha)
        print(f"Đã nạp mô hình từ: {filepath}")
