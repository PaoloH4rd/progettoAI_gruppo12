import pandas as pd
import random

from Preprocessing.data_cleaner import df_clean
from Preprocessing.feature_target_variables import X, Y
from ModelDevelopment.knn_scratch import KNN
from ModelEvaluation.metrics import calculate_metrics, display_metrics, select_metrics


if __name__ == "__main__":
    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)

    print("Dataset pulito:")
    print(df_clean.head().to_markdown(index=False, numalign="left", stralign="left"))

    print("\nFeature (X):")
    print(X.head().to_markdown(index=False, numalign="left", stralign="left"))

    print("\nVariabile target (Y):")
    print(Y.head().to_markdown(index=False, numalign="left", stralign="left"))

    # Esempio di utilizzo del modello KNN
    k = int(input("\nInserisci un numero di vicini da considerare (k): "))  # Numero di vicini

    # Seleziona un punto test randomico dal dataset
    random_index = random.randint(0, len(X) - 1)
    test_point = X.iloc[random_index].values.tolist()
    actual_group = Y.iloc[random_index]

    print(f"\n--- Punto test selezionato casualmente (indice {random_index}) ---")
    print(f"Valori delle feature: {test_point}")
    print(f"Gruppo reale: {actual_group}")

    # Addestra il modello KNN (usando tutti i dati come training)
    knn_model = KNN(X.values.tolist(), Y.values.tolist(), k)

    # Testa il modello sul punto casuale
    prediction = knn_model.test([test_point])

    print(f"\n=== PREDIZIONE KNN ===")
    print(f"Gruppo predetto: {prediction[0]}")
    print(f"======================")

    # Seleziona e valuta le metriche
    selected_metrics = select_metrics()
    y_pred = knn_model.test(X.values.tolist())
    metrics = calculate_metrics(Y.values.tolist(), y_pred)
    y_true = Y.values.tolist()
    display_metrics(metrics, selected_metrics)






