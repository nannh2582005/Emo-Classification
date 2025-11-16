import pandas as pd
import numpy as np
import os
import re

from collections import Counter


from DataPreprocessor import data_preprocessing

def main():
    # đường dẫn dữ liệu (có thể thay đổi)
    path = r"D:\HK1 NĂM 3\PYTHON FOR DATA SCIENCE\ĐỒ ÁN\emotion-classification\data.xlsx" 

    data = data_preprocessing(file_path=path)
    data.review_data()

if __name__ == "__main__":
    main()