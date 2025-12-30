import math

def build_confusion_matrix(y_true, y_pred):
    """
    Costruisce la matrice di confusione per classificazione binaria.
    """
    all_unique_classes = sorted(list(set(y_true) | set(y_pred)))
    if len(all_unique_classes) < 2:
        return 0, 0, 0, 0

    positive_class = max(all_unique_classes)
    negative_class = min(all_unique_classes)
    tp = tn = fp = fn = 0

    for true, predicted in zip(y_true, y_pred):
        if true == positive_class and predicted == positive_class:
            tp += 1
        elif true == negative_class and predicted == negative_class:
            tn += 1
        elif predicted == positive_class and true == negative_class:
            fp += 1
        elif predicted == negative_class and true == positive_class:
            fn += 1
    return tp, tn, fp, fn

def calculate_accuracy_rate(y_true, y_pred):
    """Calcola l'Accuracy Rate."""
    correct = sum(1 for true, predicted in zip(y_true, y_pred) if true == predicted)
    total = len(y_true)
    return correct / total if total > 0 else 0

def calculate_error_rate(y_true, y_pred):
    """Calcola l'Error Rate."""
    return 1 - calculate_accuracy_rate(y_true, y_pred)

def calculate_sensitivity(y_true, y_pred):
    """Calcola la Sensitivity (True Positive Rate)."""
    tp, _, _, fn = build_confusion_matrix(y_true, y_pred)
    denominator = tp + fn
    return tp / denominator if denominator > 0 else 0

def calculate_specificity(y_true, y_pred):
    """Calcola la Specificity (True Negative Rate)."""
    _, tn, fp, _ = build_confusion_matrix(y_true, y_pred)
    denominator = tn + fp
    return tn / denominator if denominator > 0 else 0

def calculate_geometric_mean(y_true, y_pred):
    """Calcola la Geometric Mean."""
    sensitivity = calculate_sensitivity(y_true, y_pred)
    specificity = calculate_specificity(y_true, y_pred)
    return math.sqrt(sensitivity * specificity) if sensitivity > 0 and specificity > 0 else 0

def calculate_roc_curve(y_true, y_pred_proba):
    """Calcola i punti (FPR, TPR) per la curva ROC."""
    true_unique_classes = sorted(set(y_true))
    if len(true_unique_classes) < 2:
        return None, None

    positive_class = max(true_unique_classes)
    y_true_binary = [1 if y == positive_class else 0 for y in y_true]

    sorted_indices = sorted(range(len(y_pred_proba)), key=lambda i: y_pred_proba[i], reverse=True)
    sorted_y_true = [y_true_binary[i] for i in sorted_indices]

    n_pos = sum(y_true_binary)
    n_neg = len(y_true_binary) - n_pos

    tpr_list = [0]
    fpr_list = [0]
    tp = 0
    fp = 0

    for i, actual in enumerate(sorted_y_true):
        if actual == 1:
            tp += 1
        else:
            fp += 1
        tpr = tp / n_pos if n_pos > 0 else 0
        fpr = fp / n_neg if n_neg > 0 else 0
        tpr_list.append(tpr)
        fpr_list.append(fpr)

    return fpr_list, tpr_list

def calculate_auc(fpr, tpr):
    """Calcola l'Area Under the Curve (AUC) usando il metodo dei trapezi."""
    if fpr is None or tpr is None:
        return None
    area = 0
    for i in range(len(fpr) - 1):
        width = fpr[i + 1] - fpr[i]
        height_avg = (tpr[i] + tpr[i + 1]) / 2
        area += width * height_avg
    return area

def calculate_metrics(y_true, y_pred, y_pred_proba=None):
    """Calcola tutte le metriche di valutazione."""
    metrics = {
        'accuracy': calculate_accuracy_rate(y_true, y_pred),
        'error_rate': calculate_error_rate(y_true, y_pred),
        'sensitivity': calculate_sensitivity(y_true, y_pred),
        'specificity': calculate_specificity(y_true, y_pred),
        'gmean': calculate_geometric_mean(y_true, y_pred),
        'auc': None
    }
    if y_pred_proba is not None:
        fpr, tpr = calculate_roc_curve(y_true, y_pred_proba)
        if fpr is not None and tpr is not None:
            metrics['auc'] = calculate_auc(fpr, tpr)
    return metrics
