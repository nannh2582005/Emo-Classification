import os
import pandas as pd
from logs.logger import setup_logger

class DataLoader:
    """
    Docstring for DataLoader
    đọc dữ liệu từ nhiều dịnh dạng khác nhau: csv, xlxs, json 
    """
    def __init__(self, file_path: str):
        """hàm khởi tạo"""
        self.logger = setup_logger('data_loader')
        self.file_path = file_path # đường dẫn đến file dữ liệu (csv, xlsx, json)
        self._df = None
        self.logger.info(f"Tải dữ liệu từ: {file_path}")

    def load_data(self):
        """nhập dữ liệu từ nhiều file khác nhau"""
        
        # kiểm tra file có tồn tại hay không
        if not os.path.exists(self.file_path):
            self.logger.error(f"File không tồn tại: {self.file_path}")
            raise FileNotFoundError("File không tồn tại!")
        
        self.logger.info(f"Đang đọc dữ liệu từ file {self.file_path}")

        # lấy phần mở rộng của file 
        ext = os.path.splitext(self.file_path)[-1].lower() 
        try: 
            if ext == '.csv':
                self._df = pd.read_csv(self.file_path, encoding='utf8')
            elif ext in ['.xlsx', '.xls']:
                self._df = pd.read_excel(self.file_path)
            elif ext == '.json':
                self._df = pd.read_json(self.file_path)
            else:
                self.logger.error("Định dạng file không hỗ trợ.")
                raise ValueError("Không hỗ trợ định dạng này, vui lòng nhập định dạng [csv, xlsx, json]")
            self.logger.info(f"Tải dữ liệu thành công! Số dòng: {len(self._df)}")
        except Exception as e:
            self.logger.exception("Lỗi khi đọc dữ liệu:")
            raise e
        
        return self._df

    @property
    def data(self):
        """truy cập dữ liệu"""
        if self._df is None:
            self.logger.warning("Truy cập dữ liệu khi chưa load!")
        return self._df
    
    @property
    def review_data(self):
        """xem trước 5 dòng dữ liệu"""
        if self._df is None:
            self.logger.warning("Gọi review_data khi chưa load!")
            return None

        self.logger.info("Xem trước 5 dòng dữ liệu.")
        return self._df.head(5)
