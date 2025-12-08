import os
import pandas as pd

class DataLoader:
    """
    Docstring for DataLoader
    đọc dữ liệu từ nhiều dịnh dạng khác nhau: csv, xlxs, json 
    """
    def __init__(self, file_path: str):
        """hàm khởi tạo"""
        self.file_path = file_path # đường dẫn đến file dữ liệu (csv, xlsx, json)
        self._df = None

    def load_data(self):
        """nhập dữ liệu từ nhiều file khác nhau"""
        
        # kiểm tra file có tồn tại hay không
        if not os.path.exists(self.file_path):
            raise FileNotFoundError("File không tồn tại!")
        
        # lấy phần mở rộng của file 
        ext = os.path.splitext(self.file_path)[-1].lower() 
        if ext == '.csv':
            self._df = pd.read_csv(self.file_path, encoding='utf8')
        elif ext in ['.xlsx', '.xls']:
            self._df = pd.read_excel(self.file_path)
        elif ext == '.json':
            self._df = pd.read_json(self.file_path)
        else:
            raise ValueError("Không hỗ trợ định dạng này, vui lòng nhập định dạng [csv, xlsx, json]")
        return self._df
    
    @property
    def data(self):
        """truy cập dữ liệu"""
        return self._df
    
    def review_data(self, n = 5):
        """xem trước 5 dòng dữ liệu, có thể điều chỉnh tham số n"""
        return self._df.head(n)