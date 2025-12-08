from .base import DataModel
from sklearn.svm import SVC
from sklearn.metrics import classification_report
from imblearn.over_sampling import RandomOverSampler

class SVMModel(DataModel):
    """SVM cho phân loại cảm xúc với lựa chọn cân bằng dữ liệu"""
    def __init__(self, X, y, test_size=0.2, random_state=42, 
                 kernel='linear', C=1):
        super().__init__(X, y, test_size, random_state)
        self._kernel = kernel
        self._C = C

    @property
    def X_test(self):
        return self._X_test
    @property 
    def y_test(self):
        return self._y_test
    
    def balance_data(self):
        """cân bằng dữ liệu bằng ramdom over sampling"""
        ros = RandomOverSampler(random_state=self._random_state)
        self._X_train, self._y_train = ros.fit_resample(self._X_train, self._y_train)

    def train(self):
        """huấn luyện mô hình bằng phương pháp Support Vector Machine"""
        # chia tập dữ liệu
        self.split_data() 
        # cân bằng dữ liệu
        self.balance_data()
        self._model = SVC(kernel=self._kernel, C=self._C,
                              probability=True, random_state=self._random_state)

        self._model.fit(self._X_train, self._y_train)

    def predict(self, X):
        """dự đoán"""
        return self._model.predict(X)

    def evaluate(self, verbose=True):
        """đánh giá"""
        y_pred = self._model.predict(self._X_test)
        report = classification_report(self._y_test, y_pred, output_dict=True)
    
        if verbose:
            print("=== Classification Report ===")
            print(classification_report(self._y_test, y_pred))
    
        self._last_report = report
        return report