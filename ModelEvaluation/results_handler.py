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


class MultiRunResultsHandler(BaseResultsHandler):
    """
    Classe base per handler che gestiscono risultati multipli (es. K-Fold, Shuffle Split).
    Centralizza la logica di plotting e salvataggio per validazioni iterative multiple.
    Evito  la duplicazione del codice tra K-Fold e Stratified Shuffle Split.
    Cambio solo i titoli e le etichette nei metodi specifici (plot specifici).
    """
    def __init__(self, metrics_list, raw_data_list, filename_prefix, output_dir='output', 
                 run_label='Run', y_true_all=None, y_pred_all=None, y_pred_proba_all=None):
        super().__init__(y_true_all, y_pred_all, y_pred_proba_all, filename_prefix, output_dir)
        self.metrics_list = metrics_list
        self.raw_data_list = raw_data_list if raw_data_list is not None else []
        self.run_label = run_label
        
        # Calcola AUC medio
        aucs = [m.get('auc') for m in metrics_list if m.get('auc') is not None]
        if aucs:
            self.auc_score = sum(aucs) / len(aucs)

    def _plot_performance_distribution(self, title):
        """Genera e salva un box plot della distribuzione delle metriche."""
        try:
            if not self.metrics_list:
                return
            df_runs = pd.DataFrame(self.metrics_list)
            plt.figure(figsize=(12, 7))
            sns.boxplot(data=df_runs[['accuracy', 'sensitivity', 'specificity', 'gmean', 'auc']])
            plt.title(title)
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

    def plot_confusion_matrix(self, title_template="Matrici di Confusione", subplot_label_template="Run"):
        """
        Genera una griglia di matrici di confusione.
        """
        if not self.raw_data_list:
            return

        try:
            num_runs = len(self.raw_data_list)
            cols = 3 if num_runs > 4 else 2
            rows = math.ceil(num_runs / cols)

            fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows))
            axes = axes.flatten()

            for i, run_data in enumerate(self.raw_data_list):
                tp, tn, fp, fn = build_confusion_matrix(run_data['y_true'], run_data['y_pred'])
                cm = [[tn, fp], [fn, tp]]

                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[i], cbar=False,
                            xticklabels=['B', 'M'], yticklabels=['B', 'M'])
                axes[i].set_title(f'{subplot_label_template} {i + 1}')
                axes[i].set_xlabel('Pred')
                axes[i].set_ylabel('Real')

            for j in range(num_runs, len(axes)):
                axes[j].axis('off')

            plt.suptitle(title_template.format(n=num_runs), fontsize=16)
            plt.tight_layout()
            filepath = os.path.join(self.output_dir, f'{self.filename_prefix}_runs_confusion_matrix.png')
            plt.savefig(filepath)
            plt.close()
            print(f"  - Grafico '{filepath}' salvato correttamente.")
        except Exception as e:
            print(f"  - ERRORE nella generazione delle matrici di confusione multiple: {e}")

    def plot_roc_curve(self, title_template="Curve ROC", label_template="Run"):
        """
        Genera un grafico con tutte le curve ROC sovrapposte.
        """
        if not self.raw_data_list:
            return

        try:
            plt.figure(figsize=(10, 8))

            for i, run_data in enumerate(self.raw_data_list):
                fpr, tpr = calculate_roc_curve(run_data['y_true'], run_data['y_pred_proba'])
                if fpr is not None and tpr is not None:
                    plt.plot(fpr, tpr, lw=2, alpha=0.8, label=f'{label_template} {i+1}')

            plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r', label='Random', alpha=.8)

            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title(title_template.format(n=len(self.raw_data_list), auc=self.auc_score))
            plt.legend(loc="lower right")
            filepath = os.path.join(self.output_dir, f'{self.filename_prefix}_runs_roc_curve.png')
            plt.savefig(filepath)
            plt.close()
            print(f"  - Grafico '{filepath}' salvato correttamente.")
        except Exception as e:
            print(f"  - ERRORE nella generazione delle curve ROC multiple: {e}")

    def save_results(self):
        """Salva il CSV e i grafici per la validazione multipla."""
        print(f"\n--- Salvataggio risultati ({self.run_label}s) ---")
        if not self._create_output_dir():
            return

        try:
            records = []
            for i, metrics in enumerate(self.metrics_list):
                record = metrics.copy()
                record[self.run_label] = f'{self.run_label} {i+1}'
                records.append(record)
            df_runs = pd.DataFrame(records).set_index(self.run_label)
            numeric_df = df_runs.select_dtypes(include='number')
            avg_metrics = numeric_df.mean().to_dict()
            std_metrics = numeric_df.std().to_dict()
            avg_metrics[self.run_label] = 'Average'
            std_metrics[self.run_label] = 'Std_Dev'
            df_summary = pd.DataFrame([avg_metrics, std_metrics]).set_index(self.run_label)
            df_results = pd.concat([df_runs, df_summary])
            filepath = os.path.join(self.output_dir, f'{self.filename_prefix}_results.csv')
            df_results.to_csv(filepath, float_format='%.4f')
            print(f"  - Risultati salvati correttamente in '{filepath}'")

        except Exception as e:

            print(f"  - ERRORE nel salvataggio del file CSV: {e}")

        self._plot_specific_graphs()
        print("--- Operazioni completate. ---")
        time.sleep(2)
        print("\n" + "=" * 60)
        print("AVVISO: I risultati dettagliati e i grafici sono stati salvati.")
        print("Controlla la cartella 'output' nella directory del progetto.")
        print("=" * 60)
    
    @abstractmethod
    def _plot_specific_graphs(self):
        """Metodo astratto per chiamare i plot specifici con i titoli corretti."""
        pass


class KFoldResultsHandler(MultiRunResultsHandler):
    """
    Handler specifico per K-Fold Cross Validation.
    """
    def __init__(self, all_fold_metrics, filename_prefix, output_dir='output',
                 y_true_all=None, y_pred_all=None, y_pred_proba_all=None, all_fold_raw_data=None):
        super().__init__(all_fold_metrics, all_fold_raw_data, filename_prefix, output_dir, 
                         run_label='Fold', y_true_all=y_true_all, y_pred_all=y_pred_all, y_pred_proba_all=y_pred_proba_all)

    def _plot_specific_graphs(self):
        self._plot_performance_distribution('Distribuzione delle Performance sulle k-Fold')
        self.plot_confusion_matrix(title_template='Matrici di Confusione per {n}-Fold CV', subplot_label_template='Fold')
        self.plot_roc_curve(title_template='Curve ROC per {n}-Fold CV (AUC Medio = {auc:.4f})', label_template='Fold')


class StratifiedShuffleSplitResultsHandler(MultiRunResultsHandler):
    """
    Handler specifico per Stratified Shuffle Split.
    """
    def __init__(self, all_experiment_metrics, filename_prefix, output_dir='output',
                 y_true_all=None, y_pred_all=None, y_pred_proba_all=None, all_experiment_raw_data=None):
        super().__init__(all_experiment_metrics, all_experiment_raw_data, filename_prefix, output_dir, 
                         run_label='Experiment', y_true_all=y_true_all, y_pred_all=y_pred_all, y_pred_proba_all=y_pred_proba_all)

    def _plot_specific_graphs(self):
        self._plot_performance_distribution('Distribuzione delle Performance su Stratified Shuffle Split')
        self.plot_confusion_matrix(title_template='Matrici di Confusione per {n} Esperimenti', subplot_label_template='Exp')
        self.plot_roc_curve(title_template='Curve ROC per {n} Esperimenti (AUC Medio = {auc:.4f})', label_template='Exp')
