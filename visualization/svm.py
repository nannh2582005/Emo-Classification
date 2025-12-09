from .base import VisualizationBase
from logs.logger import setup_logger


class SVMVisualization(VisualizationBase):
    def __init__(self, save_dir: str = "images"):
        super().__init__(save_dir)
        self.logger = setup_logger(self.__class__.__name__)
        self.logger.info("Khởi tạo SVMVisualization, thư mục lưu hình: %s", save_dir)

    def visualize(self, model, target_names=None):
        """
        Trực quan hóa mô hình SVM bằng confusion matrix.

        model: đối tượng SVMModel đã train xong
        target_names: danh sách tên nhãn
        """
        self.logger.info("Bắt đầu trực quan hóa mô hình SVM")

        # Lấy giá trị thực và dự đoán
        y_true = model.y_test
        y_pred = model.predict(model.X_test)

        self.logger.info("Dữ liệu đánh giá: %d mẫu", len(y_true))

        # Nếu không có labels thì tạo mặc định
        if target_names is None:
            target_names = [str(i) for i in sorted(set(y_true))]
            self.logger.warning("Không truyền target_names, sử dụng danh sách mặc định: %s", target_names)

        # Tạo confusion matrix
        self.logger.info("Vẽ confusion matrix cho SVM...")
        fig = self.plot_confusion_matrix(
            y_true=y_true,
            y_pred=y_pred,
            labels=target_names,
            title="Confusion Matrix - SVM"
        )

        # Lưu file hình
        filename = "svm_confusion_matrix.png"
        self.save_figure(fig, filename)
        self.logger.info("Đã lưu hình trực quan hóa SVM tại: %s", filename)

        print("Hoàn tất trực quan hóa SVM!")
