from sklearn.feature_extraction.text import TfidfVectorizer
from .base import DataFeature
from logs.logger import setup_logger
from config import Config
def identity_tokenizer(text):
    return text
#để chạy app do thư viện joblib (dùng để lưu file .pkl) không thể lưu được các hàm vô danh (lambda).

class TFIDF(DataFeature):
    def __init__(self, texts: list[list[str]]):
        
        super().__init__(texts)
        self.logger = setup_logger("tfidf")
        # lấy tham số từ config 
        max_features = Config.TFIDF_MAX_FEATURES
        ngram_range = Config.TFIDF_NGRAM
        
        self.logger.info(
            f"Khởi tạo TFIDF với max_features={max_features}, "
            f"ngram_range={ngram_range}"
        )

        self.vectorizer = TfidfVectorizer(
            tokenizer=identity_tokenizer,    # Thay lambda x: x bằng hàm có tên
            preprocessor=identity_tokenizer,
            token_pattern=None,
            max_features=max_features,
            ngram_range=ngram_range,
            min_df=3,
            max_df=0.95,
            sublinear_tf=True,
            norm='l2'
        )
        self.logger.info("TFIDF đã được khởi tạo thành công.")

    def fit(self, X):
        self.logger.info("Bắt đầu fit TFIDF")
        try: 
            self.vectorizer.fit(X)
            self.logger.info("Fit TFIDF thành công.")
        except Exception as e:
            self.logger.exception("Lỗi khi fit TFIDF:")
            raise e

    def transform(self, X):
        """biến token thành TFIDF"""
        self.logger.info("Bắt đầu transform TFIDF...")
        try: 
            vect =  self.vectorizer.transform(X)
            self.logger.info(
                f"Transform hoàn thành. Shape = {vect.shape}"
            )
            return vect
        except Exception as e:
            self.logger.exception("Lỗi khi transform TFIDF:")
            raise e
