# AI per la classificazione dei tumori - Progetto Fondamenti di Intelligenza Artificiale

Questo progetto implementa un sistema di intelligenza artificiale per la classificazione dei tumori (benigni o maligni) utilizzando un modello **K-Nearest Neighbors (K-NN)** sviluppato interamente da zero. L'obiettivo principale è fornire uno strumento di analisi robusto, supportato da diverse tecniche di validazione e metriche di valutazione avanzate.

## 🎯 Obiettivo del Progetto

Sviluppare un modello di apprendimento automatico capace di classificare i tumori in base alle caratteristiche fornite nel dataset, verificandone le prestazioni attraverso metodologie rigorose di validazione.

## 🚀 Caratteristiche Principali

- **KNN da zero**: Implementazione manuale della logica del classificatore e del calcolo delle distanze (Euclidea).
- **Preprocessing**: Moduli dedicati alla pulizia e alla preparazione dei dati (`data_cleaner.py`).
- **Tecniche di Validazione**:
  - **Holdout**: Divide il dataset in training set (es. 70%) e test set (es. 30%). È un metodo rapido, sebbene sensibile alla specifica divisione dei dati.
  - **K-Fold Cross Validation**: Divide il dataset in $K$ parti uguali. Esegue $K$ esperimenti ruotando il fold di test, fornendo una stima della performance più stabile e robusta.
  - **Stratified Shuffle Split**: Esegue più divisioni casuali mantenendo la proporzione originale delle classi (stratificazione). Ideale per garantire che ogni split sia rappresentativo del dataset originale.
- **Metriche di Valutazione**: Calcolo manuale di Accuracy, Error Rate, Sensitivity, Specificity, Geometric Mean e AUC (Area Under the Curve).
- **Visualizzazione**: Generazione di matrici di confusione, curve ROC e grafici delle performance.
- **Containerizzazione**: Supporto Docker per facilitare la distribuzione e l'esecuzione.

## 📁 Struttura del Progetto

```text
progettofia/
├── main.py                     # Entry point dell'applicazione (CLI)
├── model.py                    # Wrapper del modello
├── ModelDevelopment/
│   └── knn_scratch.py          # Logica del KNN (distanze e predizioni)
├── ModelEvaluation/
│   ├── metrics.py              # Calcolo manuale di tutte le metriche
│   ├── holdout_validation.py   # Implementazione Holdout
│   ├── cross_validation.py     # Implementazione K-Fold
│   └── ...                     # Altri metodi di validazione
├── Preprocessing/
│   ├── data_cleaner.py         # Script per la pulizia del dataset
│   └── feature_target_variables.py # Gestione feature e target
├── contenitore csv/            # Directory per i dataset
├── output/                     # Grafici e report generati
├── Dockerfile                  # Configurazione Docker
└── requirements.txt            # Dipendenze generate
```

## 🛠️ Installazione e Setup

### Gestione Pacchetti
Per la gestione delle dipendenze è stato utilizzato `pip-tools`.
- I pacchetti principali sono definiti in `requirements.in`.
- Per rigenerare il file delle dipendenze:
  ```bash
  pip-compile requirements.in
  ```

### Inizializzazione Virtual Environment
1. Crea il venv: `python -m venv .venv`
2. Attiva il venv: 
   - Linux/macOS: `source .venv/bin/activate`
   - Windows: `.venv\Scripts\activate`
3. Installa e sincronizza:
   ```bash
   pip install -r requirements.txt
   pip-sync
   ```

### Esecuzione con Docker
```bash
docker-compose up --build
```

## 💻 Utilizzo

Esegui il file principale per avviare l'interfaccia interattiva:

```bash
python main.py
```

Il programma chiederà interattivamente:
- Il file da analizzare (e procederà alla pulizia se necessario).
- Il valore di **k** (numero di vicini); il sistema suggerirà un valore ottimale basato sull'Error Rate.
- Il metodo di validazione desiderato (**Holdout**, **K-Fold**, **Stratified Shuffle Split**).
- Le metriche verranno calcolate automaticamente e i risultati (inclusi i grafici) saranno salvati nella cartella `output/`.

---
*Progetto realizzato per il corso di Fondamenti di Intelligenza Artificiale.*
