# thư viện cần thiết 
from pyvi import ViTokenizer
import pandas as pd
import re

class DataProcessor:
    """
    lớp tiền xử lý dữ liệu:
    - cleaning: đưa về chữ thường, loại bỏ các ký tự đặc biệt, xóa bỏ khoảng trắng thừa, chuẩn hóa unicode
    - tokenization, loại bỏ dấu câu và loại bỏ các stopword
    - xử lý các từ teencode - xem xét thêm 
    - mã hóa nhãn 
    """
    def __init__(self, df: pd.DataFrame, text_column: str, label_column: str, stopwords: list = None):
        self.df = df.copy() # tránh làm mất dữ liệu gốc
        self.label_column = label_column    # cột nhãn
        self.text_column = text_column  # cột text
        self.label2id = self.create_label_dict()   # tạo dict nhãn -> id
        self.id2label = {v: k for k, v in self.label2id.items()}  # dict id -> nhãn
        self.df["label_id"] = self.encode_labels()  # thêm cột mã hóa
        self.stopwords = set(stopwords) if stopwords else set()

    def clean_text(self):
        '''chuyển chữ thường, loại bỏ các ký hiệu đặc biệt và khoảng trắng thừa'''
        def clean(s):
            s = s.lower()
            s = re.sub(r'[^\w\s]', ' ', s)  # loại bỏ các ký hiệu đặc biệt 
            s = re.sub(r'\s+', ' ', s).strip()  # loại bỏ khoảng trắng thừa
            return s
        self.df[self.text_column] = self.df[self.text_column].apply(clean)  # clean với cột text
        self.df[self.label_column] = self.df[self.label_column].apply(clean)    # clean với cột label

    def tokenization(self):
        '''tách từ bằng ViTokenizer, áp dụng với từng chuỗi trong dữ liệu
        kết quả trả về có dạng: hôm_nay'''
        self.df[self.text_column] = self.df[self.text_column].apply(ViTokenizer.tokenize)

    def remove_stopwords(self):
        '''loại bỏ stopwords'''
        def remove(tokens):
            return ' '.join([t for t in tokens.split() if t not in self.stopwords]) # lọc stopwords
        # áp dụng cho từng cột trong dữ liệu 
        self.df[self.text_column] = self.df[self.text_column].apply(remove) 
 
    def create_label_dict(self):    
        '''Lấy danh sách nhãn và gán id: 0, 1, 2, 3, 4, 5, 6'''
        unique_labels = self.df[self.label_column].unique()
        return {label: idx for idx, label in enumerate(unique_labels)}
    
    def encode_labels(self):
        '''đổi nhãn text thành số'''
        return self.df[self.label_column].map(self.label2id)

    def preprocess(self):
        '''chạy các bước tiền xử lý'''
        self.clean_text()
        self.tokenization()
        self.remove_stopwords()