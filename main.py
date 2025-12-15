import pandas as pd

from Preprocessing.data_cleaner import df_clean
from Preprocessing.feature_target_variables import X, Y
from ModelDevelopment.knn_scratch import KNN


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
    k = int(input("inserisci un numero di vicini da considerare: "))  # Numero di vicini

    knn_model = KNN(X.values.tolist(), Y.values.tolist(), k)
    predictions = knn_model.test(X.values.tolist())
    print("\nPredizioni del modello KNN:")
    print(predictions)





