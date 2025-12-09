from sklearn.model_selection import GridSearchCV
from abc import ABC, abstractmethod
from logs.logger import setup_logger

class ModelOptimizer(ABC):
    """Class cha cho tối ưu tham số của các mô hình ML"""

    def __init__(self, X, y):
        self.X = X
        self.y = y
        self.best_params_ = None
        self.best_score_ = None
        self.logger = setup_logger(self.__class__.__name__)
        self.logger.info("Khởi tạo ModelOptimizer")

    @property
    @abstractmethod
    def model_class(self):
        """Trả về class của mô hình (ví dụ SVC, LogisticRegression, ...). Con phải override."""
        pass

    @property
    @abstractmethod
    def param_grid(self):
        """Trả về grid tham số cho GridSearchCV. Con phải override."""
        pass

    def optimize(self, cv=5, scoring='f1_macro'):
        """Tối ưu tham số với GridSearchCV"""
        self.logger.info("Bắt đầu tối ưu tham số cho %s", self.model_class.__name__)
        grid = GridSearchCV(self.model_class(), self.param_grid, cv=cv, scoring=scoring, n_jobs=-1)
        grid.fit(self.X, self.y)
        self.best_params_ = grid.best_params_
        self.best_score_ = grid.best_score_
        self.logger.info("Tối ưu xong. Best score: %.4f, Best params: %s", self.best_score_, self.best_params_)
        return self.best_params_, self.best_score_

    def train_best_model(self):
        """Huấn luyện mô hình với tham số tốt nhất"""
        if self.best_params_ is None:
            raise ValueError("Chưa tối ưu tham số. Chạy optimize() trước!")
        self.logger.info("Huấn luyện mô hình với tham số tối ưu")
        model = self.model_class(**self.best_params_)
        model.fit(self.X, self.y)
        return model
