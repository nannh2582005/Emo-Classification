import pandas as pd
import numpy as np
import os
import re

from collections import Counter

from DataPreprocessor.DataLoader import DataLoader
from DataPreprocessor.DataLabeling import DataLabeling
from DataPreprocessor.DataEncoder import DataEncoder

def main():
    # đường dẫn dữ liệu (có thể thay đổi)
    path = r"data/test_nor_811.xlsx" 

    data = DataLoader(file_path=path)
    data.review_data()

if __name__ == "__main__":
    main()