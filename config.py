"""Lưu tham số và đường dẫn cho mô hình, bao gồm:
- tham số tiền xử lý (STOPWORDS, EMOJI_FILE, TEENCODE_FILE)
- tham số đặc trưng (TFIDF_MAX_FEATURE)
- tham số mô hình (SVM_PARAMS)
- tham số tối ưu (SVM_PARAM_GRID)"""
import os 

class Config:
    # đường dẫn đến
    if "__file__" in globals():  
        BASE_DIR = os.path.dirname(os.path.abspath(__file__)) # lấy đường dẫn tuyệt đối của project ..:/../Emo-Classification  
    else:
        BASE_DIR = os.getcwd()  # nếu chạy trong Jupyter Notebook
    MODEL_DIR = os.path.join(BASE_DIR, "models_saver") # đường dẫn đến thư mục lưu mô hình huấn luyện
    LOG_DIR = os.path.join(BASE_DIR, "logs") # đường dẫn đến thư mục lưu file log
    DATA_DIR = os.path.join(BASE_DIR, "data")

    # đường dẫn file
    DATA_FILE = os.path.join(DATA_DIR, 'data.xlsx')
    EMOJI_FILE = os.path.join(DATA_DIR, 'emoji_dict.json')
    TEENCODE_FILE = os.path.join(DATA_DIR, 'teencode_dict.json')
    STOPWORDS_FILE = os.path.join(DATA_DIR, 'vietnamese-stopwords.txt')

    # lựa chọn cho tiền xử lý dữ liệu 
    REMOVE_URLS = True
    NORMALIZE_UNICODE = True
    REMOVE_SPECIAL_CHARS = True
    REMOVE_STOPWORDS = True

    # tham số đặc trưng của TFIDF
    TFIDF_MAX_FEATURES = 5000
    TFIDF_NGRAM = (1, 2)

    # tham số cho mô hình 
    SVM_DEFAULT = {"C": 1.0, "kernel": "linear", "probability": True}

    # tham số cho huấn luyện mô hình 
    TEST_SIZE = 0.2
    RANDOM_STATE = 42

    # tham số tối ưu mô hình 
    SVM_GRID = {"C": [0.1, 1.0, 5.0], "kernel": ["linear", "rbf"]}
