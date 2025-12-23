import pandas as pd

# Legge il file CSV in un DataFrame
df = pd.read_csv('version_1.csv')

# Stampa le informazioni iniziali sul DataFrame
print(df.info())

# 1. Uniformare i dati: Sostituire le virgole con i punti e convertire in numerico
cols_to_fix = ['Single Epithelial Cell Size', 'Bland Chromatin']

for col in cols_to_fix:
    try:
        # Assicurarsi che la colonna sia di tipo stringa prima della sostituzione
        df[col] = df[col].astype(str).str.replace(',', '.')
        # Convertire in numerico, forzando gli errori a NaN (anche se ci aspettiamo che siano correggibili)
        df[col] = pd.to_numeric(df[col], errors='coerce')
    except Exception as e:
        print(f"Errore durante l'elaborazione della colonna '{col}': {e}")

# 2. Analisi dei valori mancanti e sostituzione dei valori non validi
# Verificare il conteggio dei valori nulli in ogni colonna

null_counts = df.isnull().sum()
print("Conteggio valori nulli per colonna:\n", null_counts)

# Controllare le righe dove la variabile target 'classtype_v1' è mancante
missing_target_df = df[df['classtype_v1'].isnull()]
print("\nPrime 5 righe con 'classtype_v1' mancante:")
print(missing_target_df.head().to_markdown(index=False, numalign="left", stralign="left"))

# Strategia: Rimuovere le righe con valori target mancanti perché non possono essere utilizzate per l'apprendimento supervisionato
df_clean = df.dropna(subset=['classtype_v1'])

# Controllare nuovamente i valori nulli dopo aver rimosso le righe con target mancante
print("\nConteggio valori nulli dopo aver eliminato i target mancanti:\n", df_clean.isnull().sum())

# Rimuovere le righe duplicate
df_clean = df_clean.drop_duplicates()
print(f"Rimosse {len(df) - len(df_clean)} righe duplicate")

# Rimuovere le colonne che non sono utili per l'analisi
columns_to_drop = ['Sample code number', 'BareNucleix_wrong', 'Blood Pressure', 'Heart Rate']
df_clean = df_clean.drop(columns=columns_to_drop, errors='ignore')

# Salvare il dataset pulito
df_clean.to_csv('version_1_cleaned.csv', index=False)

print("\nDimensioni dopo la pulizia:", df_clean.shape)

df_clean= pd.read_csv('version_1_cleaned.csv')