class DataEncoder:
    def __init__(self, df, text_column, embed_dim=50):
        self.df = df
        self.text_column = text_column
        self.embed_dim = embed_dim

        self.df["tokens"] = self.df[self.text_column].apply(self.tokenize)
        self.word2index = self.build_vocab(self.df["tokens"])                
        self.index2word = {v: k for k, v in self.word2index.items()}

        self.embed_matrix = self.create_embedding_matrix()                   

    # tách từ
    def tokenize(self, text):
        return str(text).split()

    # tạo từ điển: từ → index
    def build_vocab(self, token_series):
        counter = Counter()
        for tokens in token_series:
            counter.update(tokens)
        return {word: idx for idx, (word, _) in enumerate(counter.items())}

    # tạo ma trận embedding ngẫu nhiên
    def create_embedding_matrix(self):
        vocab_size = len(self.word2index)
        # random embedding để mô hình train tiếp
        return np.random.uniform(
            low=-0.01, high=0.01,
            size=(vocab_size, self.embed_dim)
        )

    # vector TF-IDF cho toàn dataset
    def compute_idf(self):
        N = len(self.df)
        df_count = np.zeros(len(self.word2index))

        for tokens in self.df["tokens"]:
            for t in set(tokens):
                if t in self.word2index:
                    df_count[self.word2index[t]] += 1

        return np.log((N + 1) / (df_count + 1)) + 1

    # TF-IDF cho 1 câu
    def tfidf_vectorize(self, tokens, idf):
        tf = np.zeros(len(self.word2index))
        for token in tokens:
            if token in self.word2index:
                tf[self.word2index[token]] += 1

        if tf.sum() > 0:
            tf = tf / tf.sum()

        return tf * idf

    # TF-IDF cho toàn dataset
    def tfidf_transform(self):
        idf = self.compute_idf()
        return np.array([
            self.tfidf_vectorize(tokens, idf)
            for tokens in self.df["tokens"]
        ])

    # tạo embedding vector trung bình cho 1 câu
    def embed_sentence(self, tokens):
        vectors = []
        for token in tokens:
            if token in self.word2index:
                idx = self.word2index[token]
                vectors.append(self.embed_matrix[idx])
        if len(vectors) == 0:
            return np.zeros(self.embed_dim)
        return np.mean(vectors, axis=0)

    # embedding cho toàn dataset
    def embedding_transform(self):
        return np.array([
            self.embed_sentence(tokens)
            for tokens in self.df["tokens"]
        ])