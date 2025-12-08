# thư viện cần thiết 
from pyvi import ViTokenizer
import pandas as pd
import re
import unicodedata
import json

class DataProcessor:
    """
    lớp tiền xử lý dữ liệu:
    - cleaning: đưa về chữ thường, loại bỏ các ký tự đặc biệt, xóa bỏ khoảng trắng thừa, chuẩn hóa unicode
    - tokenization, loại bỏ dấu câu và loại bỏ các stopword
    - gom nhãn dữ liệu 
    - xử lý các từ teencode 
    - chuyển emoji thành từ mang cảm xúc
    - mã hóa nhãn 
    """
    def __init__(self, df: pd.DataFrame,
                  text_column: str, 
                  label_column: str, 
                  stopwords: list = None,
                  emoji: dict = None,
                  teencode: dict = None):
        self._df = df.copy() # tránh làm mất dữ liệu gốc
        self._label_column = label_column    # cột nhãn
        self._text_column = text_column  # cột text
        self._stopwords = set(stopwords) if stopwords else set() # lưu stopwords
        self._emoji_dict = emoji # lưu emoji với key là emoji và value là nhãn tương ứng
        self._teencode_dict = teencode # lưu những từ viết tắt với key là teencode và value là nghĩa của từ
        self._processed_col = 'processed'

        self._merge_label()
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
        if not s:
            return ' '
        s = s.lower()
        s = re.sub(r'https?://\S+', ' ', s)  # bỏ link
        # Giữ lại chữ, số, khoảng trắng và emoji
        # Thay các ký tự không phải chữ/emoji bằng khoảng trắng
        s = re.sub(r'[!\"#$%&\'()*+,-./:;<=>?@\[\\\]^_`{|}~]', ' ', s)
        s = re.sub(r'\s+', ' ', s).strip()
        return s
    def _normalize_repeated_chars(self, text: str):
        text = re.sub(r'(.)\1{2,}', r'\1\1', text)
        text = re.sub(r'(\)+)', lambda m: ')' * min(len(m.group(1)), 2), text)
        text = re.sub(r'(=)(\)+)', lambda m: '=' + ')' * min(len(m.group(2)), 2), text)
        return text

    @staticmethod
    def _tokenization(s:str):
        '''tách từ bằng ViTokenizer, áp dụng với từng chuỗi trong dữ liệu'''
        return ViTokenizer.tokenize(s)

    def _remove_stopwords(self, tokens: list[str]):
        '''loại bỏ stopwords'''
        return [t for t in tokens if t not in self._stopwords] # lọc stopwords
        # áp dụng cho từng cột trong dữ liệu 
    
    def _merge_label(self):
        """Gộp nhãn dữ liệu thành Tích cực / Tiêu cực / Trung tính"""
        def mapper(label):
            if label in ['Enjoyment', 'Surprise']:
                return 'Tích cực'
            elif label in ['Disgust', 'Fear', 'Sadness', 'Anger']:
                return 'Tiêu cực'
            else:
                return 'Trung tính'
        self._df['merged_label'] = self._df[self._label_column].apply(mapper)

    def _replace_teencode(self, text: str):
        for k, v in self._teencode_dict.items():
            # escape để tránh lỗi regex với ký tự đặc biệt
            pattern = rf"\b {k} \b"
            text = re.sub(pattern, f" {v} ", text)
        return text

    def _replace_emoji(self, text: str):
        for emo, meaning in self._emoji_dict.items():
            text = text.replace(emo, f" {meaning} ")
        return text

    def _create_label_dict(self):    
        '''Lấy danh sách nhãn và gán id: 0, 1, 2. Tương ứng:
        - Tích cực 0
        - Tiêu cực 1
        - Trung tính 2
        '''
        unique_labels = self._df['merged_label'].unique()
        return {label: idx for idx, label in enumerate(unique_labels)}
    
    def _encode_labels(self):
        '''đổi nhãn text thành số'''
        return self._df['merged_label'].map(self._label2id)
    
    def preprocess(self):
        '''chạy các bước tiền xử lý'''
        self._df[self._processed_col] = (
            self._df[self._text_column].astype(str)
            .apply(self._normalize_unicode)
            .apply(self._replace_teencode)
            .apply(self._replace_emoji)
            .apply(self._normalize_repeated_chars)
            .apply(self._clean_text)
            .apply(self._tokenization)
            .apply(lambda s: s.split())
            .apply(self._remove_stopwords)
            )
        return self._df
    
    @property
    def df(self):
        return self._df