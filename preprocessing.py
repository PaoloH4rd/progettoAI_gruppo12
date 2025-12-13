import pandas as pd

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)

# Leggi il file CSV in un DataFrame
df = pd.read_csv('version_1.csv')

# Mostra le prime 5 righe
print(df.head().to_markdown(index=False, numalign="left", stralign="left"))

# Stampa i nomi delle colonne e i loro tipi di dati
print(df.info())


# 1. Uniformazione dei dati: sostituzione delle virgole con punti e conversione in numerico
cols_to_fix = ['Single Epithelial Cell Size', 'Bland Chromatin']

for col in cols_to_fix:
    # Assicurati che la colonna sia di tipo stringa prima della sostituzione, per sicurezza
    df[col] = df[col].astype(str).str.replace(',', '.')
    # Converti in numerico, forzando gli errori a NaN (anche se ci aspettiamo che siano correggibili)
    df[col] = pd.to_numeric(df[col], errors='coerce')