import joblib
import os
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from config import Config
from logs.logger import setup_logger

class LogisticRegressionModel:
    def __init__(self, X, y, config: Config = Config):
        """
        Khởi tạo mô hình Logistic Regression.
        Nhận X, y ngay từ đầu để đồng bộ với cấu trúc Optimizer.
        """
        self.logger = setup_logger(self.__class__.__name__)
        self.X = X
        self.y = y
        self.config = config
        
        # Các thuộc tính dữ liệu sau khi split
        self._X_train = None
        self._X_test = None
        self._y_train = None
        self._y_test = None
        
        # Model sklearn thực tế
        self._model = None 

    def split_data(self):
        """Chia dữ liệu train/test"""
        self.logger.info(f" chia dữ liệu với tỉ lệ Test: {self.config.TEST_SIZE*100}%...")
        
        self._X_train, self._X_test, self._y_train, self._y_test = train_test_split(
            self.X, self.y, 
            test_size=self.config.TEST_SIZE, 
            random_state=self.config.RANDOM_STATE, 
            stratify=self.y 
        )
        self.logger.info(f"Đã chia Train ({self._X_train.shape[0]}), Test ({self._X_test.shape[0]})")

    def train(self):
        """Huấn luyện mô hình mặc định (nếu không dùng Optimizer)"""
        if self._X_train is None:
            self.split_data()
        
        self.logger.info("Logistic Regression ...")
        self._model = LogisticRegression(
            max_iter=1000,
            solver='lbfgs',
            random_state=self.config.RANDOM_STATE
        )
        self._model.fit(self._X_train, self._y_train)
        self.logger.info("xong.")

    def evaluate(self, target_names=None):
        """Đánh giá mô hình"""
        if self._model is None:
            self.logger.error("Chưa có model. Vui lòng train trước.")
            return

        self.logger.info(" ĐÁNH GIÁ MÔ HÌNH (LOGISTIC REGRESSION):")
        y_pred = self._model.predict(self._X_test)
        
        acc = accuracy_score(self._y_test, y_pred)
        self.logger.info(f"Độ chính xác (Accuracy): {acc:.4f}")
        
        print("\nBáo cáo chi tiết:")
        print(classification_report(self._y_test, y_pred, target_names=target_names))
        
        print("\nMa trận nhầm lẫn:")
        print(confusion_matrix(self._y_test, y_pred))

    def save_model(self, filename="logistic_model.pkl"):
        """Lưu mô hình"""
        filepath = os.path.join(self.config.MODEL_DIR, filename)
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
            
        joblib.dump(self._model, filepath)
        self.logger.info(f"lưu mô hình tại: {filepath}")