
from Preprocessing.feature_target_variables import X, Y
from ModelDevelopment.knn_scratch import KNN
from ModelEvaluation.metrics import calculate_metrics, display_metrics

# Configurazione
k = 3

print("Dataset Info:")
print(f"Numero di righe: {len(X)}")
print(f"Numero di feature: {len(X.iloc[0])}")
print(f"Valori unici in Y: {Y.unique()}")
print(f"Distribuzione Y: {Y.value_counts().to_dict()}")

# Addestra il modello KNN
knn_model = KNN(X.values.tolist(), Y.values.tolist(), k)

# Testa il modello
y_true = Y.values.tolist()
# Faccio delle predizioni sui dati di training stessi -> (non c'è differenza tra train e test in questo debug)
y_pred = knn_model.test(X.values.tolist())

y_pred_proba = knn_model.test_proba(X.values.tolist())

print(f"\nPrimis 10 y_true: {y_true[:10]}")
print(f"Primis 10 y_pred: {y_pred[:10]}")
print(f"Primis 10 y_pred_proba: {y_pred_proba[:10]}")

# Calcola metriche
metrics = calculate_metrics(y_true, y_pred, y_pred_proba)

print("\nMetriche:")
print(f"Accuracy: {metrics['accuracy']:.4f}")
print(f"Error Rate: {metrics['error_rate']:.4f}")
print(f"Sensitivity: {metrics['sensitivity']:.4f}")
print(f"Specificity: {metrics['specificity']:.4f}")
print(f"G-Mean: {metrics['gmean']:.4f}")
if metrics['auc'] is not None:
    print(f"AUC: {metrics['auc']:.4f}")
else:
    print("AUC: None")

