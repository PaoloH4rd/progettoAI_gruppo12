"""
Test per la funzione binary_stratified_shuffle_split

Questi test verificano la corretta implementazione della funzione binary_stratified_shuffle_split:
- Generazione del numero corretto di split
- Nessuna sovrapposizione tra train e test set
- Tutti i campioni sono utilizzati (train OR test)
- Mantenimento della proporzione delle classi (stratificazione) per dataset bilanciati e sbilanciati
- Dimensione corretta del test set (20%)
- Split diversi per ogni esperimento
- Riproducibilità con lo stesso seed
- Diversità dei risultati con seed diversi

"""


import unittest
import numpy as np
from ModelEvaluation.stratified_shuffle_split_validation import binary_stratified_shuffle_split

class TestBinaryStratifiedShuffleSplit(unittest.TestCase):
    """Test per la funzione binary_stratified_shuffle_split."""

    def setUp(self):
        """Prepara i dati di test comuni a tutti i test."""
        # Dataset bilanciato: 50 classe 0, 50 classe 1
        self.Y_balanced = [0] * 50 + [1] * 50
        # Dataset sbilanciato: 80 classe 0, 20 classe 1
        self.Y_unbalanced = [0] * 80 + [1] * 20

    def test_correct_number_of_experiments(self):
        """Verifica che venga generato il numero corretto di split."""
        # Arrange
        n_experiments = 5

        # Act
        splits = list(binary_stratified_shuffle_split(
            self.Y_balanced, n_experiments=n_experiments, test_size=0.2
        ))

        # Assert
        self.assertEqual(len(splits), n_experiments)

    def test_train_test_no_overlap(self):
        """Verifica che train e test non abbiano elementi in comune."""
        # Arrange & Act
        for train_idx, test_idx in binary_stratified_shuffle_split(
            self.Y_balanced, n_experiments=3, test_size=0.2
        ):
            # Assert
            train_set = set(train_idx)
            test_set = set(test_idx)
            intersection = train_set & test_set
            self.assertEqual(len(intersection), 0,
                           "Train e test non devono avere elementi in comune")

    def test_all_samples_used(self):
        """Verifica che tutti i campioni siano utilizzati (train OR test)."""
        # Arrange
        expected_indices = set(range(len(self.Y_balanced)))

        # Act & Assert
        for train_idx, test_idx in binary_stratified_shuffle_split(
            self.Y_balanced, n_experiments=3, test_size=0.2
        ):
            all_indices = set(train_idx) | set(test_idx)
            self.assertEqual(all_indices, expected_indices,
                           "Tutti i campioni devono essere usati")

    def test_stratification_preserved_balanced_dataset(self):
        """Verifica che la proporzione delle classi sia mantenuta (dataset bilanciato)."""
        # Arrange
        Y = np.array(self.Y_balanced)
        expected_class_0_in_test = 10  # 50 * 0.2
        expected_class_1_in_test = 10  # 50 * 0.2

        # Act
        for train_idx, test_idx in binary_stratified_shuffle_split(
            Y, n_experiments=1, test_size=0.2
        ):
            y_test = Y[test_idx]
            class_0_count = np.sum(y_test == 0)
            class_1_count = np.sum(y_test == 1)

            # Assert
            self.assertEqual(class_0_count, expected_class_0_in_test)
            self.assertEqual(class_1_count, expected_class_1_in_test)

    def test_stratification_preserved_unbalanced_dataset(self):
        """Verifica che la proporzione delle classi sia mantenuta (dataset sbilanciato)."""
        # Arrange
        Y = np.array(self.Y_unbalanced)
        expected_class_0_in_test = 16  # int(80 * 0.2)
        expected_class_1_in_test = 4   # int(20 * 0.2)

        # Act
        for train_idx, test_idx in binary_stratified_shuffle_split(
            Y, n_experiments=1, test_size=0.2
        ):
            y_test = Y[test_idx]
            class_0_count = np.sum(y_test == 0)
            class_1_count = np.sum(y_test == 1)

            # Assert
            self.assertEqual(class_0_count, expected_class_0_in_test)
            self.assertEqual(class_1_count, expected_class_1_in_test)

    def test_correct_test_size(self):
        """Verifica che la dimensione del test set sia il 20%."""
        # Arrange
        total_samples = len(self.Y_balanced)
        expected_test_size = int(total_samples * 0.2)
        expected_train_size = total_samples - expected_test_size

        # Act & Assert
        for train_idx, test_idx in binary_stratified_shuffle_split(
            self.Y_balanced, n_experiments=1, test_size=0.2
        ):
            self.assertEqual(len(test_idx), expected_test_size)
            self.assertEqual(len(train_idx), expected_train_size)

    def test_different_splits_each_experiment(self):
        """Verifica che ogni esperimento produca split diversi."""
        # Arrange & Act
        splits = list(binary_stratified_shuffle_split(
            self.Y_balanced, n_experiments=3, test_size=0.2
        ))
        test_sets = [set(split[1]) for split in splits]

        # Assert
        all_same = all(t == test_sets[0] for t in test_sets)
        self.assertFalse(all_same, "Gli esperimenti devono avere split diversi")

    def test_reproducibility_with_same_seed(self):
        """Verifica che lo stesso seed produca gli stessi risultati."""
        # Arrange & Act
        splits_1 = list(binary_stratified_shuffle_split(
            self.Y_balanced, n_experiments=2, test_size=0.2, random_seed=42
        ))
        splits_2 = list(binary_stratified_shuffle_split(
            self.Y_balanced, n_experiments=2, test_size=0.2, random_seed=42
        ))

        # Assert
        for i in range(len(splits_1)):
            np.testing.assert_array_equal(splits_1[i][0], splits_2[i][0])
            np.testing.assert_array_equal(splits_1[i][1], splits_2[i][1])

    def test_different_seeds_produce_different_results(self):
        """Verifica che seed diversi producano risultati diversi."""
        # Arrange & Act
        splits_1 = list(binary_stratified_shuffle_split(
            self.Y_balanced, n_experiments=1, test_size=0.2, random_seed=42
        ))
        splits_2 = list(binary_stratified_shuffle_split(
            self.Y_balanced, n_experiments=1, test_size=0.2, random_seed=123
        ))

        # Assert
        self.assertFalse(
            np.array_equal(splits_1[0][1], splits_2[0][1]),
            "Seed diversi devono produrre risultati diversi"
        )



if __name__ == '__main__':
    unittest.main()
