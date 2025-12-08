from sklearn.feature_extraction.text import TfidfVectorizer
from .base import DataFeature

class TFIDF(DataFeature):
    def __init__(self, texts: list[list[str]], 
                 max_features: int = 10000, 
                 ngram_range=(1, 2),
                 min_df=3,
                 max_df=0.95):
        
        super().__init__(texts)

        self.vectorizer = TfidfVectorizer(
            tokenizer=lambda x: x,
            preprocessor=lambda x: x,
            token_pattern=None,
            max_features=max_features,
            ngram_range=ngram_range,
            min_df=min_df,
            max_df=max_df,
            sublinear_tf=True,
            norm='l2'
        )

    def fit(self, X):
        self.vectorizer.fit(X)

    def transform(self, X):
        return self.vectorizer.transform(X)