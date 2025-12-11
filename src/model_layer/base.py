from abc import ABC, abstractmethod
from sklearn.model_selection import train_test_split

class DataModel(ABC):
    def __init__(self, X, y, test_size=0.2, random_state=42):
        self._X = X
        self._y = y
        self._test_size = test_size
        self._random_state = random_state
        
        self._X_train = None
        self._X_test = None
        self._y_train = None
        self._y_test = None
        self._model = None  # class con sẽ gán model vào đây

    def split_data(self):
        """Chia dữ liệu thành train/test"""
        self._X_train, self._X_test, self._y_train, self._y_test = train_test_split(
            self._X, self._y, test_size=self._test_size, random_state=self._random_state
        )

    @abstractmethod
    def train(self):
        """Huấn luyện model (class con triển khai)"""
        pass

    @abstractmethod
    def predict(self, X):
        """Dự đoán dữ liệu mới (class con triển khai)"""
        pass

    @abstractmethod
    def evaluate(self):
        """Đánh giá model trên dữ liệu test (class con triển khai)"""
        pass

