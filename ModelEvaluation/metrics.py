import math

def build_confusion_matrix(y_true, y_pred):

    # Se nei dati reali e predetti compare una sola classe in totale, non è
    # possibile costruire una matrice di confusione completa (es. TP, FN).
    # Restituire zero per tutto è l'opzione più sicura per evitare crash
    # e far sì che le metriche derivate (sens, spec) risultino 0.
    all_unique_classes = sorted(list(set(y_true) | set(y_pred)))
    if len(all_unique_classes) < 2:
        return 0, 0, 0, 0

    positive_class = max(all_unique_classes)
    negative_class = min(all_unique_classes)
    tp = tn = fp = fn = 0
    """
    zip combina due liste elemento per elemento, creando coppie di valori corrispondenti. 
    Qui zip(y_true, y_pred) prende il primo elemento da y_true e il primo elemento da y_pred, 
    poi il secondo da entrambi, e così via.
    In ogni iterazione, true contiene un valore reale e predicted contiene la corrispondente predizione.
    """
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
    correct = sum(1 for true, predicted in zip(y_true, y_pred) if true == predicted)
    total = len(y_true)
    return correct / total if total > 0 else 0

def calculate_error_rate(y_true, y_pred):
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
    # Se non ci sono almeno due classi (es. solo positivi o solo negativi),
    # non è possibile calcolare una curva ROC significativa.
    if len(true_unique_classes) < 2:
        return None, None

    # Identifica la classe positiva come il valore massimo (convenzione comune).
    positive_class = max(true_unique_classes)
    # Binarizza le etichette reali: 1 per la classe positiva, 0 per le altre.
    y_true_binary = [1 if y == positive_class else 0 for y in y_true]

    # Ordina gli indici dei dati in base alla probabilità predetta, dal più alto al più basso (reverse=True).
    # Le predizioni con maggiore confidenza (più vicine a 1.0) verranno valutate per prime.
    sorted_indices = sorted(range(len(y_pred_proba)), key=lambda i: y_pred_proba[i], reverse=True)

    # Riordina le etichette reali (Ground Truth) seguendo l'ordine degli indici calcolato sopra.
    # Questo permette di scorrere i dati simulando l'abbassamento progressivo della soglia di decisione
    # e aggiornare TP/FP incrementalmente, invece di ricalcolare tutto per ogni soglia.
    sorted_y_true = [y_true_binary[i] for i in sorted_indices]

    # Calcola il numero totale di campioni positivi e negativi nel dataset.
    n_pos = sum(y_true_binary)
    n_neg = len(y_true_binary) - n_pos

    # Inizializza le liste FPR e TPR con il punto (0, 0).
    tpr_list = [0]
    fpr_list = [0]
    tp = 0
    fp = 0

    # Scorre i campioni ordinati per probabilità.
    # Ogni passo rappresenta l'inclusione di un nuovo campione come "predetto positivo"
    # abbassando la soglia di decisione.
    for i, actual in enumerate(sorted_y_true):
        if actual == 1:
            tp += 1 # Se è realmente positivo, aumenta i True Positives
        else:
            fp += 1 # Se è realmente negativo, aumenta i False Positives
        tpr = tp / n_pos if n_pos > 0 else 0
        fpr = fp / n_neg if n_neg > 0 else 0
        tpr_list.append(tpr)
        fpr_list.append(fpr)

    return fpr_list, tpr_list

def calculate_auc(fpr, tpr):
    """
    Calcola l'Area Under the Curve (AUC) usando il metodo dei trapezi.
    """
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
