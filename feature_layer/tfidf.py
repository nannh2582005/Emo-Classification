from sklearn.feature_extraction.text import TfidfVectorizer
from .base import DataFeature
class TFIDF(DataFeature):
    """vector hóa bằng phương pháp TFIDF
    X: list những danh sách token, ví dụ: [["tôi", "yêu", "việt_nam"], ["hôm_nay", "trời", "đẹp"]]
    sử dụng tham số để nhận đầu vào là danh sách đã tokenize"""
    def __init__(self, texts: list[list[str]], max_features: int = 5000, ngram_range = (1,2)):
        super().__init__(texts)
        # tham số để nhận pretokenized list
        self.vectorizer = TfidfVectorizer(
            tokenizer=lambda x: x,
            preprocessor=lambda x: x,
            token_pattern=None,
            max_features=max_features,  
            ngram_range=ngram_range
        )
    
    def fit(self, X: list[list[str]]):
        self.vectorizer.fit(X)

    def transform(self, X):
        return self.vectorizer.transform(X)