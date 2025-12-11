from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from src.model_layer.logistics import LogisticRegressionModel
from config import Config
from logs.logger import setup_logger

class LogisticOptimizer:
    """
    Lớp tối ưu tham số cho Logistic Regression.
    - X, y: dữ liệu
    - cv: số fold cho GridSearchCV
    """

    def __init__(self, X, y, config: Config = Config):
        self.logger = setup_logger(self.__class__.__name__)
        self.X = X
        self.y = y
        self.config = config
        self.best_params_ = None
        self.best_score_ = None

        self.logger.info("Khởi tạo LogisticOptimizer")

    def optimize(self, cv: int = 5):
        """
        Tìm tham số tốt nhất cho Logistic Regression bằng GridSearchCV
        """
        self.logger.info("Bắt đầu tối ưu tham số Logistic Regression với %d-fold CV", cv)

        # Lưới tham số cần dò tìm
        param_grid = {
            'C': [0.1, 1, 10, 100],          # Độ mạnh của Regularization
            'solver': ['lbfgs', 'liblinear'] # Thuật toán tối ưu
        }

        # Khởi tạo model cơ sở để tune
        lr = LogisticRegression(max_iter=1000, random_state=self.config.RANDOM_STATE)

        grid = GridSearchCV(
            lr,
            param_grid,
            cv=cv,
            scoring='f1_macro', # Hoặc 'accuracy' tùy bài toán
            n_jobs=1,
            verbose=1
        )

        grid.fit(self.X, self.y)

        self.best_params_ = grid.best_params_
        self.best_score_ = grid.best_score_

        self.logger.info(
            "Tối ưu xong. Best params: %s, Best score: %.4f",
            self.best_params_, self.best_score_
        )

        return self.best_params_, self.best_score_

    def train_best_model(self):
        """
        Trả về đối tượng LogisticRegressionModel đã train với tham số tốt nhất.
        """
        if self.best_params_ is None:
            raise ValueError("Chưa tối ưu. Hãy gọi optimize() trước.")

        self.logger.info("Logistic Regression  tham số tối ưu")

        # Tạo wrapper model
        log_model = LogisticRegressionModel(self.X, self.y, config=self.config)

        # Lấy tham số tối ưu
        best_C = self.best_params_.get("C", 1.0)
        best_solver = self.best_params_.get("solver", 'lbfgs')

        # Chia dữ liệu (gọi method của class model)
        log_model.split_data()
        
        # Khởi tạo model sklearn với tham số tốt nhất
        log_model._model = LogisticRegression(
            C=best_C,
            solver=best_solver,
            max_iter=1000,
            random_state=self.config.RANDOM_STATE
        )

        # Train trên tập train đã chia
        log_model._model.fit(log_model._X_train, log_model._y_train)

        self.logger.info("train LogisticRegressionModel xong")

        return log_model