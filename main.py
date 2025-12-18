
import pandas as pd
import random

from Preprocessing.data_cleaner import df_clean
from Preprocessing.feature_target_variables import X, Y
from ModelDevelopment.knn_scratch import KNN
from ModelEvaluation.Holdout import Holdout


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

# Esempio di utilizzo della tecnica di Holdout
print(f"\n--- Esempio di Holdout ---")
holdout = Holdout(0.2, 42)
X_train, X_test, Y_train, Y_test = holdout.split_train_test(X.values.tolist(), Y.values.tolist())

print(f"Dimensione training set: {len(X_train)}")
print(f"Dimensione test set: {len(X_test)}")
print(f"Primi 5 elementi del training set X: {X_train[:5]}")
print(f"Primi 5 elementi del training set Y: {Y_train[:5]}")
print(f"Primi 5 elementi del test set X: {X_test[:5]}")
print(f"Primi 5 elementi del test set Y: {Y_test[:5]}")




