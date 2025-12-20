import math

def build_confusion_matrix(y_true, y_pred):
    """Costruisce la matrice di confusione per classificazione binaria"""
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
    Calcola Accuracy Rate
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


def calculate_roc_curve(y_true, y_pred_proba):
    """
    curva ROC
    Ritorna: - False Positive Rate e True Positive Rate
    y_pred_proba: probabilità predette (valori tra 0 e 1)
    """
    # Ordina per probabilità decrescente
    sorted_indices = sorted(range(len(y_pred_proba)), key=lambda i: y_pred_proba[i], reverse=True)
    sorted_y_true = [y_true[i] for i in sorted_indices]
    sorted_y_proba = [y_pred_proba[i] for i in sorted_indices]

    # Calcola TP, FP, TN, FN totali
    n_pos = sum(y_true)  # Numero di positivi
    n_neg = len(y_true) - n_pos  # Numero di negativi

    tpr_list = [0]  # Inizia da (0, 0)
    fpr_list = [0]

    tp = 0
    fp = 0

    # Itera attraverso i threshold
    for i, (actual, proba) in enumerate(zip(sorted_y_true, sorted_y_proba)):
        if actual == 1:
            tp += 1
        else:
            fp += 1

        # Calcola TPR e FPR
        # - False Positive Rate e True Positive Rate
        tpr = tp / n_pos if n_pos > 0 else 0
        fpr = fp / n_neg if n_neg > 0 else 0

        tpr_list.append(tpr)
        fpr_list.append(fpr)

    return fpr_list, tpr_list


def calculate_auc(fpr, tpr):
    """
    Calcola l'Area Under the Curve (AUC) usando il metodo dei trapezi
    Integra la curva ROC
    """
    return trapezoidal_integration(fpr, tpr)


def trapezoidal_integration(x, y):
    """
    Metodo dei trapezi per calcolare l'integrale numericamente
    Utile per calcolare l'area sotto la curva (es. ROC-AUC)
    """
    area = 0
    for i in range(len(x) - 1):
        # Area del trapezio = (base) * (altezza_media)
        width = x[i + 1] - x[i]
        height_avg = (y[i] + y[i + 1]) / 2
        area += width * height_avg
    return area


def calculate_metrics(y_true, y_pred, y_pred_proba=None):
    """
    Calcola tutte le metriche di valutazione

    y_true: valori reali
    y_pred: predizioni (binarie 0/1)
    y_pred_proba: probabilità predette  ROC/AUC
    """

    metrics = {
        'accuracy': calculate_accuracy_rate(y_true, y_pred),
        'error_rate': calculate_error_rate(y_true, y_pred),
        'sensitivity': calculate_sensitivity(y_true, y_pred),
        'specificity': calculate_specificity(y_true, y_pred),
        'gmean': calculate_geometric_mean(y_true, y_pred),
        'roc': None,
        'auc': None
    }

    # # Se sono disponibili le probabilità, calcola ROC e AUC
    # if y_pred_proba is not None:
    #     fpr, tpr = calculate_roc_curve(y_true, y_pred_proba)
    #     auc = calculate_auc(fpr, tpr)
    #
    #     metrics['roc'] = {'fpr': fpr, 'tpr': tpr}
    #     metrics['auc'] = auc
    #
    # return metrics


def display_metrics(metrics, selected_metrics):
    """Visualizza le metriche selezionate con descrizione"""
    metrics_labels = {
        '1': ('Accuracy Rate', 'accuracy', 'Percentuale di previsioni corrette'),
        '2': ('Error Rate', 'error_rate', 'Percentuale di previsioni errate'),
        '3': ('Sensitivity (Recall)', 'sensitivity', 'Capacità di rilevare i positivi'),
        '4': ('Specificity', 'specificity', 'Capacità di rilevare i negativi'),
        '5': ('Geometric Mean', 'gmean', 'Media geometrica di sensibilità e specificità')
    }

    print("\n" + "=" * 60)
    print("METRICHE DI VALUTAZIONE DEL MODELLO")
    print("=" * 60)

    for choice in selected_metrics:
        if choice in metrics_labels:
            label, key, description = metrics_labels[choice]
            value = metrics[key]
            print(f"\n{label}:")
            print(f"  Descrizione: {description}")
            print(f"  Valore: {value:.4f}")

    # Mostra AUC se disponibile
    if metrics.get('auc') is not None:
        print(f"\nArea Under the Curve (AUC):")
        print(f"  Descrizione: Area sotto la curva ROC (0-1, più alto è meglio)")
        print(f"  Valore: {metrics['auc']:.4f}")

    print("\n" + "=" * 60 + "\n")


def select_metrics():
    """Menu interattivo per selezionare le metriche"""
    print("\n" + "=" * 60)
    print("SELEZIONE METRICHE DI VALUTAZIONE")
    print("=" * 60)
    print("1. Accuracy Rate - Percentuale di previsioni corrette")
    print("2. Error Rate - Percentuale di previsioni errate")
    print("3. Sensitivity (Recall) - Capacità di rilevare i positivi")
    print("4. Specificity - Capacità di rilevare i negativi")
    print("5. Geometric Mean - Media geometrica di sensibilità e specificità")
    print("0. Seleziona tutte le metriche")
    print("=" * 60)

    choice = input("\nInserisci i numeri delle metriche (separati da virgola, es: 1,2,3): ").strip()

    if choice == "0":
        return ['1', '2', '3', '4', '5']

    selected = [c.strip() for c in choice.split(',') if c.strip() in ['1', '2', '3', '4', '5']]
    return selected if selected else ['0']
