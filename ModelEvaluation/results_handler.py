import os
import time
import math
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from abc import ABC, abstractmethod
from .metrics import build_confusion_matrix, calculate_roc_curve


class BaseResultsHandler(ABC):
    """
    Classe base astratta per la gestione dei risultati.
    Definisce l'interfaccia comune e implementa la logica di plotting.
    """
    def __init__(self, y_true, y_pred, y_pred_proba, filename_prefix, output_dir='output'):
        self.y_true = y_true
        self.y_pred = y_pred
        self.y_pred_proba = y_pred_proba
        self.output_dir = output_dir
        self.auc_score = 0.0  # Verrà impostato dalle sottoclassi
        self.filename_prefix = filename_prefix

    def _create_output_dir(self):
        """Crea la directory di output se non esiste."""
        try:
            if not os.path.exists(self.output_dir):
                os.makedirs(self.output_dir)
                print(f"  - Creata directory '{self.output_dir}' per i risultati.")
        except OSError as e:
            print(f"  - ERRORE nella creazione della directory '{self.output_dir}': {e}")
            return False
        return True

    @abstractmethod
    def plot_confusion_matrix(self):
        """Metodo astratto per generare la matrice di confusione."""
        pass

    @abstractmethod
    def plot_roc_curve(self):
        """Metodo astratto per generare la curva ROC."""
        pass

    @abstractmethod
    def save_results(self):
        """Metodo astratto che le sottoclassi devono implementare per salvare i risultati."""
        pass


class HoldoutResultsHandler(BaseResultsHandler):
    """Handler specifico per i risultati di una validazione Holdout."""
    def __init__(self, metrics, y_true, y_pred, y_pred_proba, filename_prefix, output_dir='output'):
        super().__init__(y_true, y_pred, y_pred_proba, filename_prefix, output_dir)
        self.metrics = metrics
        self.auc_score = metrics.get('auc') if metrics.get('auc') is not None else 0.0

    def plot_confusion_matrix(self):
        """Genera una singola matrice di confusione."""
        try:
            tp, tn, fp, fn = build_confusion_matrix(self.y_true, self.y_pred)
            cm = [[tn, fp], [fn, tp]]

            plt.figure(figsize=(8, 6))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                        xticklabels=['Benigno', 'Maligno'], yticklabels=['Benigno', 'Maligno'])
            plt.title('Matrice di Confusione (Holdout)')
            plt.xlabel('Classe Predetta')
            plt.ylabel('Classe Reale')
            filepath = os.path.join(self.output_dir, f'{self.filename_prefix}_confusion_matrix.png')
            plt.savefig(filepath)
            plt.close()
            print(f"  - Grafico '{filepath}' salvato correttamente.")
        except Exception as e:
            print(f"  - ERRORE nella generazione della matrice di confusione: {e}")

    def plot_roc_curve(self):
        """Genera una singola curva ROC."""
        try:
            fpr, tpr = calculate_roc_curve(self.y_true, self.y_pred_proba)
            if fpr is not None and tpr is not None:
                plt.figure(figsize=(8, 6))
                plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'Curva ROC (AUC = {self.auc_score:.4f})')
                plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
                plt.xlabel('False Positive Rate')
                plt.ylabel('True Positive Rate')
                plt.title('Curva ROC (Holdout)')
                plt.legend(loc='lower right')
                plt.grid(True)
                filepath = os.path.join(self.output_dir, f'{self.filename_prefix}_roc_curve.png')
                plt.savefig(filepath)
                plt.close()
                print(f"  - Grafico '{filepath}' salvato correttamente.")
        except Exception as e:
            print(f"  - ERRORE nella generazione della curva ROC: {e}")

    def save_results(self):
        """Salva il CSV e i grafici per la validazione Holdout."""
        print("\n--- Salvataggio risultati (Holdout) in corso... ---")
        if not self._create_output_dir():
            return

        try:
            metrics_record = self.metrics.copy()
            metrics_record['Validation_Type'] = 'Holdout_Test'
            df_results = pd.DataFrame([metrics_record]).set_index('Validation_Type')
            filepath = os.path.join(self.output_dir, f'{self.filename_prefix}_results.csv')
            df_results.to_csv(filepath, float_format='%.4f')
            print(f"  - Risultati salvati correttamente in '{filepath}'")
        except Exception as e:
            print(f"  - ERRORE nel salvataggio del file CSV: {e}")

        #chiama la funzione base per plottare la matrice di confusione e la curva ROC
        self.plot_confusion_matrix()
        self.plot_roc_curve()
        print("--- Operazioni completate. ---")
        time.sleep(2)
        print("\n" + "=" * 60)
        print("AVVISO: I risultati dettagliati e i grafici sono stati salvati.")
        print("Controlla la cartella 'output' nella directory del progetto.")
        print("=" * 60)


class KFoldResultsHandler(BaseResultsHandler):
    """
    Salva le metriche di ogni fold, le statistiche aggregate (media, std dev)
    e genera un grafico sulla distribuzione delle performance.
    Genera inoltre la Matrice di Confusione e la Curva ROC aggregate (se i dati sono forniti).
    """
    def __init__(self, all_fold_metrics, filename_prefix, output_dir='output',
                 y_true_all=None, y_pred_all=None, y_pred_proba_all=None, all_fold_raw_data=None):
        super().__init__(y_true_all, y_pred_all, y_pred_proba_all, filename_prefix, output_dir)
        self.all_fold_metrics = all_fold_metrics
        self.all_fold_raw_data = all_fold_raw_data if all_fold_raw_data is not None else []
        # Calcola AUC medio per visualizzazione nel grafico ROC (se disponibile)
        aucs = [m.get('auc') for m in all_fold_metrics if m.get('auc') is not None]
        if aucs:
            self.auc_score = sum(aucs) / len(aucs)

    def _plot_performance_distribution(self):
        """Genera e salva un box plot della distribuzione delle metriche sulle fold."""
        try:
            if not self.all_fold_metrics:
                return
            df_folds = pd.DataFrame(self.all_fold_metrics)
            plt.figure(figsize=(12, 7))
            sns.boxplot(data=df_folds[['accuracy', 'sensitivity', 'specificity', 'gmean', 'auc']])
            plt.title('Distribuzione delle Performance sulle k-Fold')
            plt.ylabel('Punteggio')
            plt.xticks(rotation=10)
            plt.grid(True)
            plt.tight_layout()
            filepath = os.path.join(self.output_dir, f'{self.filename_prefix}_performance_distribution.png')
            plt.savefig(filepath)
            plt.close()
            print(f"  - Grafico '{filepath}' salvato correttamente.")
        except Exception as e:
            print(f"  - ERRORE nella generazione del box plot delle performance: {e}")

    def plot_confusion_matrix(self):
        """
        Genera una griglia di matrici di confusione, una per ogni fold.
        Questo fornisce una visione compatta delle performance su ogni split.
        """
        if not self.all_fold_raw_data:
            return

        try:
            num_folds = len(self.all_fold_raw_data)
            # Calcola righe e colonne per la griglia
            cols = 3 if num_folds > 4 else 2
            rows = math.ceil(num_folds / cols)

            fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows))
            axes = axes.flatten() # Appiattisce l'array di assi per iterarci facilmente

            for i, fold_data in enumerate(self.all_fold_raw_data):
                tp, tn, fp, fn = build_confusion_matrix(fold_data['y_true'], fold_data['y_pred'])
                cm = [[tn, fp], [fn, tp]]

                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[i], cbar=False,
                            xticklabels=['B', 'M'], yticklabels=['B', 'M'])
                axes[i].set_title(f'Fold {i + 1}')
                axes[i].set_xlabel('Pred')
                axes[i].set_ylabel('Real')

            # Nasconde gli assi vuoti se num_folds non riempie la griglia
            for j in range(num_folds, len(axes)):
                axes[j].axis('off')

            plt.suptitle(f'Matrici di Confusione per {num_folds}-Fold CV', fontsize=16)
            plt.tight_layout()
            filepath = os.path.join(self.output_dir, f'{self.filename_prefix}_folds_confusion_matrix.png')
            plt.savefig(filepath)
            plt.close()
            print(f"  - Grafico '{filepath}' salvato correttamente.")
        except Exception as e:
            print(f"  - ERRORE nella generazione delle matrici di confusione K-Fold: {e}")

    def plot_roc_curve(self):
        """
        Genera un grafico con tutte le curve ROC dei singoli fold sovrapposte (Spaghetti Plot).
        Mostra la variabilità del modello tra i diversi fold.
        """
        if not self.all_fold_raw_data:
            return

        try:
            plt.figure(figsize=(10, 8))

            # Plot delle curve per ogni singolo fold
            for i, fold_data in enumerate(self.all_fold_raw_data):
                fpr, tpr = calculate_roc_curve(fold_data['y_true'], fold_data['y_pred_proba'])
                if fpr is not None and tpr is not None:
                    plt.plot(fpr, tpr, lw=2, alpha=0.8, label=f'Fold {i+1}')

            # Plot della linea casuale
            plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r', label='Random', alpha=.8)

            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title(f'Curve ROC per {len(self.all_fold_raw_data)}-Fold CV (AUC Medio = {self.auc_score:.4f})')
            plt.legend(loc="lower right")
            filepath = os.path.join(self.output_dir, f'{self.filename_prefix}_folds_roc_curve.png')
            plt.savefig(filepath)
            plt.close()
            print(f"  - Grafico '{filepath}' salvato correttamente.")
        except Exception as e:
            print(f"  - ERRORE nella generazione delle curve ROC K-Fold: {e}")

    def save_results(self):
        """Salva il CSV e i grafici per la validazione K-Fold."""
        print("\n--- Salvataggio risultati (K-Fold Pura) in corso... ---")
        if not self._create_output_dir():
            return

        try:
            records = []
            for i, fold_metrics in enumerate(self.all_fold_metrics):
                record = fold_metrics.copy()
                record['Fold'] = f'Fold {i+1}'
                records.append(record)
            df_folds = pd.DataFrame(records).set_index('Fold')

            numeric_df = df_folds.select_dtypes(include='number')
            avg_metrics = numeric_df.mean().to_dict()
            std_metrics = numeric_df.std().to_dict()
            avg_metrics['Fold'] = 'Average'
            std_metrics['Fold'] = 'Std_Dev'
            df_summary = pd.DataFrame([avg_metrics, std_metrics]).set_index('Fold')

            df_results = pd.concat([df_folds, df_summary])
            filepath = os.path.join(self.output_dir, f'{self.filename_prefix}_results.csv')
            df_results.to_csv(filepath, float_format='%.4f')
            print(f"  - Risultati salvati correttamente in '{filepath}'")
        except Exception as e:
            print(f"  - ERRORE nel salvataggio del file CSV: {e}")

        self._plot_performance_distribution()

        # Genera i grafici aggregati solo se i dati concatenati sono stati passati
        self.plot_confusion_matrix()
        self.plot_roc_curve()

        print("--- Operazioni completate. ---")
        time.sleep(2)
        print("\n" + "=" * 60)
        print("AVVISO: I risultati dettagliati e i grafici sono stati salvati.")
        print("Controlla la cartella 'output' nella directory del progetto.")
        print("=" * 60)
