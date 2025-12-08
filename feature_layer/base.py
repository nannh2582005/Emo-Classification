from abc import ABC, abstractmethod

class DataFeature(ABC):
    """class tạo feature cho mô hình huấn luyện:
    - one-hot
    - tfidf
    - word2vec
    - lưu lại số lượng token, vocab và kích thước của vocab
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
    def token_vector(self):
        """phương thức vector hóa token bằng các phương pháp khác nhau mà mỗi class con cần triển khai"""
        pass