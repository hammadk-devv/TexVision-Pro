import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, jaccard_score

class MetricCalculator:
    def __init__(self, mode='classification'):
        self.mode = mode

    def compute(self, y_true, y_pred, average='weighted'):
        """
        Compute standard classification metrics.
        y_true: Ground truth labels (numpy array)
        y_pred: Predicted labels (numpy array)
        """
        metrics = {}
        metrics['accuracy'] = accuracy_score(y_true, y_pred)
        
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_true, y_pred, average=average, zero_division=0
        )
        metrics['precision'] = precision
        metrics['recall'] = recall
        metrics['f1'] = f1
        
        return metrics

    def compute_iou(self, y_true, y_pred):
        """
        Compute IoU for segmentation (placeholder for future extension).
        """
        return jaccard_score(y_true, y_pred, average='macro', zero_division=0)
