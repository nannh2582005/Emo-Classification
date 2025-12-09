import os
import logging
# lấy đường dẫn LOG_DIR từ config
from config import Config

def setup_logger(name: str = "pipeline") -> logging.Logger:
    """
    Tạo logger dùng chung cho toàn bộ dự án.
    
    - Ghi log ra màn hình (console)
    - Ghi log vào file (pipeline.log)
    - Không bị ghi trùng khi gọi nhiều lần
    """

    # Tạo thư mục log nếu chưa có
    os.makedirs(Config.LOG_DIR, exist_ok=True)

    # Lấy logger theo tên
    logger = logging.getLogger(name)

    # Nếu logger đã có handler trước đó -> trả về luôn để tránh ghi log trùng
    if logger.handlers:
        return logger

    # Mức log muốn ghi
    logger.setLevel(logging.INFO)

    # Định dạng log hiển thị
    format_str = "[%(asctime)s] [%(levelname)s] %(message)s"
    date_format = "%Y-%m-%d %H:%M:%S"
    formatter = logging.Formatter(format_str, date_format)

    # Ghi log ra màn hình
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # Ghi log vào file 
    log_file = os.path.join(Config.LOG_DIR, name+'.log')
    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger
