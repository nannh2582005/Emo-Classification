# thư viện cần thiết 
from pyvi import ViTokenizer
import pandas as pd
import re
import unicodedata
class DataProcessor:
    """
    lớp tiền xử lý dữ liệu:
    - cleaning: đưa về chữ thường, loại bỏ các ký tự đặc biệt, xóa bỏ khoảng trắng thừa, chuẩn hóa unicode
    - tokenization, loại bỏ dấu câu và loại bỏ các stopword
    - xử lý các từ teencode - xem xét thêm 
    - mã hóa nhãn 
    """
    def __init__(self, df: pd.DataFrame, text_column: str, label_column: str, stopwords: list = None):
        self._df = df.copy() # tránh làm mất dữ liệu gốc
        self._label_column = label_column    # cột nhãn
        self._text_column = text_column  # cột text
        self._stopwords = set(stopwords) if stopwords else set()
        self._processed_col = 'processed'
        # tạo nhãn 
        self._label2id = self._create_label_dict()   # tạo dict nhãn -> id
        self._id2label = {v: k for k, v in self._label2id.items()}  # dict id -> nhãn
        self._df["label_id"] = self._encode_labels()  # thêm cột mã hóa
        
    @staticmethod
    def _normalize_unicode(s:str):
        """chuẩn text về unicode"""
        return unicodedata.normalize('NFKC',s)
    
    @staticmethod
    def _clean_text(s: str):
        '''chuyển chữ thường, loại bỏ các ký hiệu đặc biệt và khoảng trắng thừa'''
        s = s.lower()
        s = re.sub(r'https?://\S+', ' ', s)
        s = re.sub(r'[!\"#$%&\'()*+,-./:;<=>?@\[\\\]^_`{|}~]', ' ', s)  # loại bỏ các ký hiệu đặc biệt nhưng giữ lại emoji 
        s = re.sub(r'\s+', ' ', s).strip()  # loại bỏ khoảng trắng thừa
        return s
    
    @staticmethod
    def _tokenization(s:str):
        '''tách từ bằng ViTokenizer, áp dụng với từng chuỗi trong dữ liệu'''
        return ViTokenizer.tokenize(s)

    def _remove_stopwords(self, tokens: str):
        '''loại bỏ stopwords'''
        return [t for t in tokens if t not in self._stopwords] # lọc stopwords
        # áp dụng cho từng cột trong dữ liệu 
 
    def _create_label_dict(self):    
        '''Lấy danh sách nhãn và gán id: 0, 1, 2, 3, 4, 5, 6'''
        unique_labels = self._df[self._label_column].unique()
        return {label: idx for idx, label in enumerate(unique_labels)}
    
    def _encode_labels(self):
        '''đổi nhãn text thành số'''
        return self._df[self._label_column].map(self._label2id)
    
    def preprocess(self) -> pd.DataFrame:
        '''chạy các bước tiền xử lý'''
        self._df[self._processed_col] = (
            self._df[self._text_column].astype(str)
            .apply(self._normalize_unicode)
            .apply(self._clean_text)
            .apply(self._tokenization)
            .apply(lambda s: s.split())
            .apply(self._remove_stopwords)
            )
        return self._df
    
    @property
    def df(self):
        return self._df