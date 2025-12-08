# model_layer/logistic.py
import joblib
import os
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

class LogisticRegressionModel:
    def __init__(self, random_state=42):
        """
        tạo mô hình Logistic Regression.
        random_state: cố định seed để kết quả tái lập được 
        """
        self.random_state = random_state
        self.model = LogisticRegression(
            random_state=self.random_state,
            max_iter=1000,   # Tăng số vòng lặp để mô hình hội tụ tốt hơn với dữ liệu văn bản
            solver='lbfgs',  # Solver mặc định tốt cho multiclass
            
        )
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None

    def split_data(self, X, y, test_size=0.2):
        """
        Chia dữ liệu train/test
        Sử dụng stratify=y để đảm bảo tỉ lệ nhãn giữa 2 tập tương đồng
        """
        print(f"chia dữ liệu với tỉ lệ Test: {test_size*100}%...")
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, 
            test_size=test_size, 
            random_state=self.random_state, 
            stratify=y 
        )
        print(f"  Train ({self.X_train.shape[0]}), Test ({self.X_test.shape[0]})")
        return self.X_train, self.X_test, self.y_train, self.y_test

    def train(self):
        """Huấn luyện mô hình"""
        if self.X_train is None:
            raise ValueError(" chạy split_data trước khi train!")
        
        
        self.model.fit(self.X_train, self.y_train)
        print(" huấn luyện mô hình")

    def evaluate(self, target_names=None):
        """Đánh giá mô hình trên tập Test"""
        print("\n đánh giá:")
        y_pred = self.model.predict(self.X_test)
        
        acc = accuracy_score(self.y_test, y_pred)
        print(f"(Accuracy): {acc:.4f}")
        
        print("\nbáo cáo")
        print(classification_report(self.y_test, y_pred, target_names=target_names))
        
        print("\nMa trận nhầm lẫn (Confusion Matrix):")
        print(confusion_matrix(self.y_test, y_pred))

    def save_model(self, filepath="models/logistic_model.pkl"):
        """Lưu mô hình ra file"""
        # Tạo thư mục nếu chưa có
        directory = os.path.dirname(filepath)
        if not os.path.exists(directory):
            os.makedirs(directory)
            
        joblib.dump(self.model, filepath)
        print(f"💾 Đã lưu mô hình tại: {filepath}")

    def load_model(self, filepath="models/logistic_model.pkl"):
        """Nạp mô hình từ file"""
        if os.path.exists(filepath):
            self.model = joblib.load(filepath)
            print(f"nạp mô hình từ: {filepath}")
        else:
            raise FileNotFoundError(f"Không tìm thấy  {filepath}")