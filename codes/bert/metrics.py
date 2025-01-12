import os
import pandas as pd
from sklearn.metrics import accuracy_score, roc_auc_score, matthews_corrcoef, confusion_matrix
from sklearn.metrics import precision_score, recall_score


def calculate_metrics(predictions, targets):
    """
    Calculate various classification metrics.

    Args:
        predictions (torch.Tensor): Model predictions.
        targets (torch.Tensor): Ground truth labels.

    Returns:
        dict: Dictionary containing accuracy, ROC AUC, MCC, sensitivity,
              specificity, precision, and recall.
    """
    binary_predictions = (predictions >= 0.5).int().cpu().numpy()
    targets = targets.cpu().numpy()

    accuracy = accuracy_score(targets, binary_predictions)
    roc_auc = roc_auc_score(targets, binary_predictions)
    mcc = matthews_corrcoef(targets, binary_predictions)

    tn, fp, fn, tp = confusion_matrix(targets, binary_predictions).ravel()
    sensitivity = tp / (tp + fn) if (tp + fn) != 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) != 0 else 0.0

    precision = precision_score(targets, binary_predictions)
    recall = recall_score(targets, binary_predictions)

    metrics_dict = {
        'Accuracy': accuracy,
        'ROC_AUC': roc_auc,
        'MCC': mcc,
        'Sensitivity': sensitivity,
        'Specificity': specificity,
        'Precision': precision,
        'Recall': recall
    }

    return metrics_dict


def log_metrics(metrics_dict, log_file='metrics_log.csv'):
    """
    Log metrics to a CSV file.

    Args:
        metrics_dict (dict): Dictionary containing metrics.
        log_file (str): Path to the log file.
    """
    if not os.path.exists(log_file):
        df = pd.DataFrame(columns=['Epoch'] + list(metrics_dict.keys()))
        df.to_csv(log_file, index=False)

    df = pd.read_csv(log_file)
    new_df = pd.DataFrame([metrics_dict])
    updated_df = pd.concat([df, new_df], ignore_index=True)
    updated_df.to_csv(log_file, index=False)

    print(f"Metrics logged to {log_file}")

