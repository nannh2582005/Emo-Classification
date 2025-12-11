from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.svm import SVC
from src.model_layer.svm import SVMModel
from config import Config
from logs.logger import setup_logger

class SVMOptimizer:
    """
    Lớp tối ưu tham số cho SVM.
    - X, y: dữ liệu
    - cv: số fold cho GridSearchCV
    - test_size, random_state: chia dữ liệu
    """
    def __init__(self, X, y, config: Config = Config):
        self.logger = setup_logger(self.__class__.__name__)
        self.X = X
        self.y = y
        self.config = config
        self.random_state = config.RANDOM_STATE
        self.test_size = config.TEST_SIZE
        self.best_params_ = None
        self.best_score_ = None
        self.logger.info("Khởi tạo SVMOptimizer")

    def optimize(self, cv: int = 5):
        """Tìm tham số tốt nhất bằng GridSearchCV"""
        self.logger.info("Bắt đầu tối ưu tham số SVM với %d-fold CV", cv)
        param_grid = {
            'kernel': ['linear', 'rbf', 'poly'],
            'C': [0.1, 1, 10],
            'gamma': ['scale', 'auto']
        }

        svc = SVC(probability=True, random_state=self.random_state)
        grid = GridSearchCV(svc, param_grid, cv=cv, scoring='f1_macro', n_jobs=1, verbose=3)
        grid.fit(self.X, self.y)

        self.best_params_ = grid.best_params_
        self.best_score_ = grid.best_score_

        self.logger.info("Tối ưu xong. Best params: %s, Best score: %.4f", self.best_params_, self.best_score_)
        return self.best_params_, self.best_score_

    def train_best_model(self):
        """Trả về đối tượng SVMModel đã train với tham số tốt nhất"""
        if self.best_params_ is None:
            raise ValueError("Chưa tối ưu. Hãy gọi optimize() trước.")

        self.logger.info("Huấn luyện mô hình SVM với tham số tốt nhất")
        # Tạo SVMModel
        svm_model = SVMModel(self.X, self.y, config=self.config)

        # Gán tham số tốt nhất
        svm_model._kernel = self.best_params_.get('kernel', 'linear')
        svm_model._C = self.best_params_.get('C', 1)
        # Nếu gamma tồn tại
        if 'gamma' in self.best_params_:
            svm_model._gamma = self.best_params_['gamma']
        else:
            svm_model._gamma = 'scale'

        # Chia dữ liệu, train
        svm_model.split_data()
        svm_model.balance_data()
        svm_model._model = SVC(
            kernel=svm_model._kernel,
            C=svm_model._C,
            gamma=svm_model._gamma,
            probability=True,
            random_state=svm_model._random_state
        )
        svm_model._model.fit(svm_model._X_train, svm_model._y_train)

        self.logger.info("Huấn luyện SVMModel hoàn tất")
        return svm_model
