import math

def build_confusion_matrix(y_true, y_pred):
    """Costruisce manualmente la matrice di confusione per classificazione binaria"""
    tp = tn = fp = fn = 0

    for true, predicted in zip(y_true, y_pred):
        """
        zip combina due liste elemento per elemento, creando coppie di valori corrispondenti. 
        Qui zip(y_true, y_pred) prende il primo elemento da y_true e il primo elemento da y_pred, 
        poi il secondo da entrambi, e così via.
        In ogni iterazione, true contiene un valore reale e predicted contiene la corrispondente predizione."""
        if true == predicted == 1:
            tp += 1
        elif true == predicted == 0:
            tn += 1
        elif predicted == 1 and true == 0:
            fp += 1
        elif predicted == 0 and true == 1:
            fn += 1

    return tp, tn, fp, fn


def calculate_accuracy_rate(y_true, y_pred):
    """
    Calcola Accuracy Rate manualmente
    Accuracy = (TP + TN) / (TP + TN + FP + FN)
    """
    correct = sum(1 for true, predicted in zip(y_true, y_pred) if true == predicted)
    total = len(y_true)
    return correct / total if total > 0 else 0


def calculate_error_rate(y_true, y_pred):
    """
    Error Rate = 1 - Accuracy = (FP + FN) / (TP + TN + FP + FN)
    """
    return 1 - calculate_accuracy_rate(y_true, y_pred)


def calculate_sensitivity(y_true, y_pred):
    """
    Sensitivity True Positive Rate
    Sensitivity = TP / (TP + FN)
    Capacità del modello di rilevare i casi positivi
    """
    tp, tn, fp, fn = build_confusion_matrix(y_true, y_pred)
    denominator = tp + fn
    return tp / denominator if denominator > 0 else 0


def calculate_specificity(y_true, y_pred):
    """
    Specificity (True Negative Rate)
    Specificity = TN / (TN + FP)
    Capacità del modello di rilevare i casi negativi
    """
    tp, tn, fp, fn = build_confusion_matrix(y_true, y_pred)
    denominator = tn + fp
    return tn / denominator if denominator > 0 else 0


def calculate_geometric_mean(y_true, y_pred):
    """
    Geometric Mean
    G-Mean = sqrt(Sensitivity × Specificity)
    Media geometrica di sensibilità e specificità
    """
    sensitivity = calculate_sensitivity(y_true, y_pred)
    specificity = calculate_specificity(y_true, y_pred)
    return math.sqrt(sensitivity * specificity)

