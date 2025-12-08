# Emo-Classification
ĐỒ ÁN CUỐI KỲ PYTHON CHO KHOA HỌC DỮ LIỆU

Mục tiêu đồ án: Xây dựng hệ thống phân loại cảm xúc Tiếng Việt\
Phân loại cảm xúc thành 3 nhóm chính:
- Tích cực
- Tiêu cực
- Trung tính

Pipeline: data_loading -> data_preprocessing -> data_feature -> data_modeling -> evaluate
## Cài đặt 
### Cài đặt thư viện
``` bash
pip install -r requirements.txt
```
### Clone repo
```bash
git clone https://github.com/nannh2582005/Emo-Classification.git
cd Emo-Classification
```
## Pipeline xử lý dữ liệu 
### 1. Data Loader:
Đọc dữ liệu: hỗ trợ các file đuôi .csv, .xlsx, .xls, .\
Trả về dữ liệu có dạng pandas.DataFrame
### 2. Data Processor 
Bao gồm các bước:
- chuẩn hóa về unicode
- chuyển về chữ thường 
- loại bỏ URL và các ký hiệu đặc biệt (!\"#$%\'()*+,....)
- loại bỏ khoảng trắng thừa
- chuẩn hóa các ký tự lặp ('huhuuuuuu' -> 'huhu')
- tokenization bằng ViTokenizer
- loại bỏ stopwords
- gộp nhãn dữ liệu (Enjoyment, Surprise -> Tích cực,...)
- chuẩn hóa teencode (dùng teencode_dict.json)
- chuẩn hóa emoji (dùng emoji_dict.json)
- mã hóa nhãn thành số

Tất cả được gọi trong hàm preprocess.preprocess()
### 3. Data Feature
Dùng phương pháp TFIDF để đưa dữ liệu về dạng vector để làm đầu vào cho mô hình học máy, fit + transform danh sách token đã được xử lý
### 4. Data Model
Huấn luyện các mô hình học máy:
- SVM
- LinearRegression
- Naive Bayes
### 5. Visualization 
Trực quan dữ liệu

