# visualization/logistics.py
from .base import VisualizationBase
from logs.logger import setup_logger

class LogisticVisualization(VisualizationBase):
    def __init__(self, save_dir: str = "images"):
        super().__init__(save_dir)
        self.logger = setup_logger(self.__class__.__name__)
        self.logger.info("Khởi tạo LogisticVisualization, thư mục lưu hình: %s", save_dir)

    def visualize(self, model, target_names=None):
        """
        Trực quan hóa mô hình Logistic Regression bằng confusion matrix.
        model: đã train logic
        target_names: danh sách tên nhãn
        """
        self.logger.info(" trực quan hóa  Logistic Regression")

        # Lấy giá trị thực và dự đoán
       
        
        y_true = model._y_test 
        
        
        y_pred = model._model.predict(model._X_test)
        

        self.logger.info("Dữ liệu đánh giá: %d mẫu", len(y_true))

        # Nếu không có target_names thì tạo mặc định
        if target_names is None:
            target_names = [str(i) for i in sorted(set(y_true))]
            self.logger.warning(
                "Không truyền target_names, sử dụng danh sách mặc định: %s",
                target_names
            )

        # Vẽ confusion matrix
        self.logger.info("confusion matrix cho Logistic Regression...")
        fig = self.plot_confusion_matrix(
            y_true=y_true,
            y_pred=y_pred,
            labels=target_names,
            title="Confusion Matrix - Logistic Regression"
        )

        # Lưu ảnh
        filename = "logistic_confusion_matrix.png"
        self.save_figure(fig, filename)