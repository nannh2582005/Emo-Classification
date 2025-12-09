from abc import ABC, abstractmethod
from config import Config
from logs.logger import setup_logger

class DataFeature(ABC):
    """lớp trừu tượng tạo đặc trưng cho mô hình
    Dữ liệu đầu vào là dữ liệu đã được token hóa thông qua lớp DataProcessor"""
    def __init__(self, texts: list):  
        """đầu vào là danh sách texts đã được xử lý để chuẩn bị cho mã hóa"""
        self.logger = setup_logger("feature") # tạo log riêng cho feature
        self.logger.info("Khởi tạo DataFeature")
        # thuộc tính lưu số lượng token, vocab 
        self.__token_count = 0 # privated attribute để tránh truy cập và thay đổi từ ngoài
        self.__vocab = set() # privated attribute để tránh truy cập và thay đổi từ ngoài
        # tính toán số lượng token và vocab của toàn bộ dữ liệu
        self.logger.info("Bắt đầu tính token_count và vocab")
        self._compute_statistics(texts)
        self.logger.info(
            f"Hoàn thành thống kê: Tổng token = {self.token_size}, "
            f"Vocab size = {self.vocab_size}"
        )

    @property
    def token_size(self) -> int:
        """lấy số lượng token của cả mô hình"""
        return self.__token_count

    @property
    def vocab_size(self) -> int:
        """lấy số lượng vocab của cả mô hình"""
        return len(self.__vocab)

    def _compute_statistics(self, texts: list):
        """tính toán tổng token và vocab của corpus"""
        for text in texts:
            self.__token_count += len(text)
            self.__vocab.update(text)
        self.logger.debug("Cập nhật thống kê cho DataFeature")

    @abstractmethod
    def fit(self, X):
        """học dữ liệu"""
        pass
    @abstractmethod
    def transform(self, X):
        """vector hóa"""
        pass
    def fit_transform(self, X):
        """kết hợp học và vector hóa"""
        self.logger.info("Gọi fit_transform trong DataFeature")
        self.fit(X)
        self.logger.info("fit() hoàn thành — bắt đầu transform()")
        return self.transform(X)
