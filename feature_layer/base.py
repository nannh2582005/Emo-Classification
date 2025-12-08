from abc import ABC, abstractmethod

class DataFeature(ABC):
    """lớp trừu tượng tạo đặc trưng cho mô hình
    Dữ liệu đầu vào là dữ liệu đã được token hóa thông qua lớp DataProcessor"""
    def __init__(self, texts: list):  
        """đầu vào là danh sách texts đã được xử lý để chuẩn bị cho mã hóa"""
        # thuộc tính lưu số lượng token, vocab 
        self.__token_count = 0 # privated attribute để tránh truy cập và thay đổi từ ngoài
        self.__vocab = set() # privated attribute để tránh truy cập và thay đổi từ ngoài
        # tính toán số lượng token và vocab của toàn bộ dữ liệu
        self._compute_statistics(texts)
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
        self.fit(X)
        return self.transform(X)