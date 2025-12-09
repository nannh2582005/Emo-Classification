from .base import VisualizationBase
from logs.logger import setup_logger

class NaiveVisualization(VisualizationBase):
    def __init__(self, save_dir: str = "images"):
        super().__init__(save_dir)
        self.logger = setup_logger(self.__class__.__name__)
        self.logger.info("Khởi tạo NaiveVisualization, thư mục lưu hình: %s", save_dir)

    def visualize(self, model, target_names=None):
        """
        Trực quan hóa mô hình Naive Bayes bằng confusion matrix.

        model: đối tượng NaiveBayesModel đã train xong
        target_names: danh sách tên nhãn
        """
        self.logger.info("Bắt đầu trực quan hóa mô hình Naive Bayes")

        # Lấy dữ liệu thật và dự đoán
        y_true = model.y_test
        y_pred = model.predict(model.X_test)

        self.logger.info("Dữ liệu đánh giá: %d mẫu", len(y_true))

        if target_names is None:
            target_names = [str(i) for i in sorted(set(y_true))]
            self.logger.warning("Không truyền target_names, dùng danh sách mặc định: %s", target_names)

        # Vẽ confusion matrix
        self.logger.info("Vẽ confusion matrix cho Naive Bayes...")
        fig = self.plot_confusion_matrix(
            y_true=y_true,
            y_pred=y_pred,
            labels=target_names,
            title="Confusion Matrix - Naive Bayes"
        )

        # Lưu file hình
        filename = "naive_confusion_matrix.png"
        self.save_figure(fig, filename)
        self.logger.info("Đã lưu hình trực quan hóa Naive Bayes tại: %s", filename)

        print("Hoàn tất trực quan hóa Naive Bayes")