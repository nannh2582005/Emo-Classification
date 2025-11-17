class DataLabeler:
    def __init__(self, df, label_column):
        self.df = df
        self.label_column = label_column

        self.label2id = self.create_label_dict()   # tạo dict nhãn → id
        self.id2label = {v: k for k, v in self.label2id.items()}  # dict id → nhãn

        self.df["label_id"] = self.encode_labels()  # thêm cột mã hóa

    # lấy danh sách nhãn và gán id 0,1,2,...
    def create_label_dict(self):
        unique_labels = self.df[self.label_column].unique()
        return {label: idx for idx, label in enumerate(unique_labels)}

    # đổi nhãn text thành số
    def encode_labels(self):
        return self.df[self.label_column].map(self.label2id)

    # xem trước 5 dòng
    def review_labels(self, n=5):
        print(self.df[[self.label_column, "label_id"]].head(n))