class DataLoader:
    # file_path là đường dẫn của dữ liệu
    def __init__(self, file_path):
        # khởi tạo đối tượng
        self.file_path = file_path # đường dẫn đến file dữ liệu (csv, xlsx, json)
        self.df = self.load_data(file_path)
    # hàm nhập dữ liệu từ nhiều file khác nhau 
    def load_data(self, file_path):
        # kiểm tra file có tồn tại hay không
        if not os.path.exists(self.file_path):
            raise FileNotFoundError("File không tồn tại!")
        # lấy phần mở rộng của file 
        ext = os.path.splitext(file_path)[-1].lower() 
        # file csv
        if ext == '.csv':
            self.df = pd.read_csv(self.file_path)
        elif ext in ['.xlsx', '.xls']:
            self.df = pd.read_excel(self.file_path)
        elif ext == '.json':
            self.df = pd.read_json(self.file_path)
        else:
            raise ValueError("Không hỗ trợ định dạng này, vui lòng nhập định dạng [csv, xlsx, json]")
        print(f"Đọc thành công file {file_path}")
        return self.df
    # xem trước 5 dòng dữ liệu 
    def review_data(self, n = 5):
        print(self.df.head(n)) 
def main():
    path = r"data.xlsx"
    df = DataLoader(path)
    df.review_data()

if __name__ == "__main__":
    main()