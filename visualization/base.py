# visualization/base.py

from abc import ABC, abstractmethod
import os

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix


class VisualizationBase(ABC):
    def __init__(self, save_dir: str = "images"):
        """
        save_dir: thư mục dùng để lưu hình vẽ.
        """
        self._save_dir = save_dir
        os.makedirs(self._save_dir, exist_ok=True)

    def plot_confusion_matrix(self, y_true, y_pred, labels, title: str):
        """Vẽ ma trận nhầm lẫn và trả về figure."""
        cm = confusion_matrix(y_true, y_pred)

        fig = plt.figure(figsize=(6, 5))
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=labels,
            yticklabels=labels,
        )
        plt.xlabel("Dự đoán")
        plt.ylabel("Thực tế")
        plt.title(title)
        plt.tight_layout()
        return fig

    def save_figure(self, fig, filename: str):
        """Lưu figure vào thư mục đã cấu hình."""
        path = os.path.join(self._save_dir, filename)
        fig.savefig(path)
        print(f"Đã lưu hình tại: {path}")

    @abstractmethod
    def visualize(self, model, target_names=None):
        """Trực quan hoá kết quả của mô hình (class con triển khai)."""
        pass
