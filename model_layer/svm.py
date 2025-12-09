from .base import DataModel
from sklearn.svm import SVC
from sklearn.metrics import classification_report
from imblearn.over_sampling import RandomOverSampler
from config import Config
from logs.logger import setup_logger
import joblib
import os

class SVMModel(DataModel):
    """SVM cho phân loại cảm xúc với lựa chọn cân bằng dữ liệu"""
    def __init__(self, X, y, config: Config = Config):
        super().__init__(X, y, test_size = config.TEST_SIZE, random_state = config.RANDOM_STATE)
        self.config = config
        self._kernel = config.SVM_DEFAULT['kernel']
        self._C = config.SVM_DEFAULT['C']
        self.logger = setup_logger(self.__class__.__name__)
        self.logger.info("Khởi tạo SVMModel với kernel=%s, C=%s", self._kernel, self._C)

    @property
    def X_test(self):
        return self._X_test
    @property 
    def y_test(self):
        return self._y_test
    
    def balance_data(self):
        """cân bằng dữ liệu bằng ramdom over sampling"""
        self.logger.info("Bắt đầu cân bằng dữ liệu với RandomOverSampler")
        ros = RandomOverSampler(random_state=self._random_state)
        self._X_train, self._y_train = ros.fit_resample(self._X_train, self._y_train)
        self.logger.info("Cân bằng dữ liệu xong. Số mẫu mới: %d", len(self._y_train))

    def train(self):
        """huấn luyện mô hình bằng phương pháp Support Vector Machine"""
        self.logger.info("Bắt đầu train mô hình SVM")
        # chia tập dữ liệu
        self.split_data() 
        self.logger.info("Tập train size: %d | test size: %d", len(self._y_train), len(self._y_test))
        # cân bằng dữ liệu
        self.balance_data()
        self._model = SVC(kernel=self._kernel, C=self._C,
                              probability=True, random_state=self._random_state)

        self._model.fit(self._X_train, self._y_train)
        self.logger.info("Huấn luyện mô hình hoàn tất!")

    def predict(self, X):
        """dự đoán"""
        n_samples = X.shape[0]  # FIX lỗi
        self.logger.info("Dự đoán %d mẫu.", n_samples)
        return self._model.predict(X)


    def evaluate(self, verbose=True):
        """đánh giá"""
        self.logger.info("Bắt đầu đánh giá mô hình")
        y_pred = self._model.predict(self._X_test)
        report = classification_report(self._y_test, y_pred, output_dict=True)

        self.logger.info("Đánh giá mô hình xong.")
        self.logger.info("F1 mỗi lớp: %s", {k: v.get("f1-score") 
                                for k, v in report.items() if k.isdigit()})
        
        if verbose:
            print("=== Classification Report ===")
            print(classification_report(self._y_test, y_pred))
    
        self._last_report = report
        return report
    
    def save(self, filename="svm_model.pkl"):
        """Lưu mô hình xuống thư mục models."""
        if not os.path.exists(self.config.MODEL_DIR):
            os.makedirs(self.config.MODEL_DIR)

        path = os.path.join(self.config.MODEL_DIR, filename)
        joblib.dump({
            "model": self._model,
            "kernel": self._kernel,
            "C": self._C,
            "config": self.config.SVM_DEFAULT
        }, path)

        self.logger.info("Đã lưu mô hình vào: %s", path)

        return path
    
    def load_model(self, filepath="models/svm_model.pkl"):
        """Load mô hình đã lưu."""
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Không tìm thấy file mô hình: {filepath}")

        data = joblib.load(filepath)
        self._model = data["model"]
        self._kernel = data["kernel"]
        self._C = data["C"]

        self.logger.info("Đã load mô hình từ %s | kernel=%s, C=%s", filepath, self._kernel, self._C)
        print(f"Đã nạp mô hình từ: {filepath}")