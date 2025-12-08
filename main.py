import pandas as pd
import numpy as np
import os
import re

from processor_layer.loader import DataLoader
from processor_layer.processor import DataProcessor
from feature_layer.tfidf import TFIDF

def main():
    # đường dẫn dữ liệu (có thể thay đổi)
    path = r"data/data.xlsx" 

    data = DataLoader(path)
    df = data.load_data()
    print(df.head(5))

if __name__ == "__main__":
    main()
