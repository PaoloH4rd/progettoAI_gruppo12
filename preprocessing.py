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

# 2. Indagine sui valori mancanti
# Verifica se i valori mancanti in `classtype_v1` si sovrappongono con altri
null_counts = df.isnull().sum()
print("Conteggio valori nulli per colonna:\n", null_counts)

# Controlla le righe in cui il target `classtype_v1` è nullo
missing_target_df = df[df['classtype_v1'].isnull()]
print("\nPrime 5 righe con 'classtype_v1' mancante:")
print(missing_target_df.head().to_markdown(index=False, numalign="left", stralign="left"))

# Strategia: elimina le righe in cui la variabile target (classtype) è mancante,
# poiché non possiamo usarle facilmente per l'apprendimento supervisionato.
df_clean = df.dropna(subset=['classtype_v1'])

# Verifica se ci sono ancora valori mancanti nelle feature dopo aver eliminato i target mancanti
print("\nConteggio valori nulli dopo aver eliminato i target mancanti:\n", df_clean.isnull().sum())

print("\nDimensioni dopo la pulizia:", df_clean.shape)

# Salva il dataset pulito
df_clean.to_csv('version_1_cleaned.csv', index=False)

# Ora definiamo le feature e il target per l'apprendimento automatico
target_col = 'classtype_v1'

# escludo il target e 'Sample code number' che è solo un identificativo
feature_cols = [col for col in df.columns if col not in [target_col, 'Sample code number']]

X = df[feature_cols]
y = df[target_col]