# AI per la classificazione dei tumori

  # Descrizione del progetto

    L'obiettivo è sviluppare un modello di apprendimento automatico e verificarne le prestazioni per classificare i tumori in base alle caratteristiche fornite.

    Il sistema include diverse tecniche di validazione e metriche di valutazione per analizzare le prestazioni del modello. In particolare implementa un classificatore
    K-Nearest Neighbors (K-NN), per classificare tumori come benigni o maligni, e tre approcci per la validazione, l'Holdout, il K-fold Cross Validation e lo Stratified
    Shuffle Split.

    # Holdout
    
    Divide il dataset una sola volta in due parti fisse: training set (es. 70%) e test set (es. 30%). Il modello viene addestrato sul training set e valutato
    sul test set. Semplice ma può dare risultati instabili se la divisione è sfortunata.

    # K-Fold Cross Validation
    
    Divide il dataset in K parti (fold) di dimensione uguale. Esegue K esperimenti: in ogni iterazione, usa K-1 fold per il training e 1 fold per il test,
    ruotando quale fold viene usato per il test. Le performance finali sono la media dei K esperimenti. Fornisce una stima più robusta delle performance
    rispetto a Holdout.
    
    # Stratified Shuffle Split
    
    Simile a Holdout ripetuto più volte: ad ogni iterazione, mescola casualmente i dati e li divide in training/test mantenendo le stesse proporzioni 
    delle classi (stratificazione). Esegue K divisioni random indipendenti. La stratificazione garantisce che ogni split abbia la stessa distribuzione 
    di classi del dataset originale, utile per dataset sbilanciati.

  # Come Eseguire il Codice
    > python main.py

  Il programma chiederà interattivamente:

    - Il valore di k (numero di vicini)
    - Il metodo di validazione (Holdout, K-fold Cross Validation, Stratified Shuffle Split)
    - Le metriche da calcolare
 # Per la gestione dei pacchetti pip del venv è stato utilizzato pip-tools
   - i pacchetti principali sono nel file requirements.in
   - per generare il file requirements.txt :
   > pip-compile requirements.in    
 # Per inizializzare il venv
   > pip install -r requirements.txt
   > pip-sync
   
