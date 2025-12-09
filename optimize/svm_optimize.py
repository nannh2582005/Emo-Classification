from sklearn.svm import SVC

class SVMOptimizer(ModelOptimizer):
    @property
    def model_class(self):
        return SVC

    @property
    def param_grid(self):
        return {
            'C': [0.1, 1, 10],
            'kernel': ['linear', 'rbf', 'poly'],
            'gamma': ['scale', 'auto']
        }
