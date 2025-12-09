from .base import VisualizationBase
import matplotlib.pyplot as plt

class NaiveVisualization(VisualizationBase):
    def __init__(self):
        super().__init__("images")

    def visualize(self, model, target_names=None):
        y_true = model.y_test
        y_pred = model.model.predict(model.X_test)

        fig = self.plot_confusion_matrix(
            y_true, y_pred, labels=target_names,
            title="Confusion Matrix - Naive Bayes"
        )

        fig.canvas.manager.set_window_title("Naive Bayes Visualization")

        self.save_figure(fig, "naive_confusion_matrix.png")
        plt.show()
        plt.close(fig)
