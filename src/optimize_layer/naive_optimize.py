from sklearn.model_selection import GridSearchCV
from sklearn.naive_bayes import MultinomialNB
from src.model_layer.naive import NaiveBayesModel
from config import Config
from logs.logger import setup_logger


class NaiveBayesOptimizer:
    """
    Lớp tối ưu tham số cho Multinomial Naive Bayes.
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

        self.logger.info("Khởi tạo NaiveBayesOptimizer")

    def optimize(self, cv: int = 5):
        """
        Tìm tham số tốt nhất cho MultinomialNB bằng GridSearchCV
        """
        self.logger.info("Bắt đầu tối ưu tham số Naive Bayes với %d-fold CV", cv)

        param_grid = {
            'alpha': [0.1, 0.3, 0.5, 1.0, 2.0]
        }

        nb = MultinomialNB()
        grid = GridSearchCV(
            nb,
            param_grid,
            cv=cv,
            scoring='f1_macro',
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
        Trả về đối tượng NaiveBayesModel đã train với tham số tốt nhất.
        """
        if self.best_params_ is None:
            raise ValueError("Chưa tối ưu. Hãy gọi optimize() trước.")

        self.logger.info("Huấn luyện mô hình Naive Bayes với tham số tối ưu")

        # Tạo model NB
        nb_model = NaiveBayesModel(self.X, self.y, config=self.config)

        # Gán alpha tối ưu
        nb_model.alpha = self.best_params_.get("alpha", 1.0)

        # Train mô hình với alpha tốt nhất
        nb_model.split_data()
        nb_model._model = MultinomialNB(alpha=nb_model.alpha)
        nb_model._model.fit(nb_model._X_train, nb_model._y_train)

        self.logger.info("Huấn luyện NaiveBayesModel hoàn tất")

        return nb_model
