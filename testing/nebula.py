import unittest
import torch
from nebula.nebula import  MSELoss, CrossEntropyLoss, MultiLabelSoftMarginLoss,  Nebula

class TestNebula(unittest.TestCase):

    def setUp(self):
        self.nebula = Nebula()

        self.tolerance = 1e-5
        self.y_true_regression = torch.tensor([1.1, 2.2, 3.3, 4.4, 5.5], dtype=torch.float)
        self.y_pred_regression = torch.tensor([1.0, 3.0, 2.0, 5.0, 4.0], dtype=torch.float)

        # Assuming 3 classes
        self.y_true_classification = torch.tensor([0, 2, 1, 0, 2], dtype=torch.long)
        self.y_pred_classification = torch.rand((5, 3), dtype=torch.float)  # Random probabilities for classes
    
    def test_same_shape(self):
        y_true = torch.rand((10, 10))
        y_pred = torch.rand((10, 10))
        self.nebula.compute_loss(y_pred, y_true)

    def test_different_shape(self):
        y_true = torch.rand((10, 10))
        y_pred = torch.rand((10, 11))
        with self.assertRaises(ValueError):
            self.nebula.compute_loss(y_pred, y_true)

    def test_empty_tensors(self):
        y_true = torch.tensor([])
        y_pred = torch.tensor([])
        with self.assertRaises(ValueError):
            self.nebula.compute_loss(y_pred, y_true)

    def test_multidimensional_tensors(self):
        y_true = torch.rand((10, 10, 10))
        y_pred = torch.rand((10, 10, 10))
        self.nebula.compute_loss(y_pred, y_true)

    def test_y_true_unique_values_less_than_10(self):
        y_true = torch.tensor([1, 1, 2, 2, 3, 3, 4, 4, 5, 5])
        y_pred = torch.rand((10,))
        self.nebula.compute_loss(y_pred, y_true)

    def test_y_true_unique_values_greater_than_10(self):
        y_true = torch.arange(1, 11)
        y_pred = torch.rand((10,))
        self.nebula.compute_loss(y_pred, y_true)

    def test_negative_integers_in_y_true(self):
        y_true = torch.tensor([-1, -2, -3, -4, -5, 1, 2, 3, 4, 5])
        y_pred = torch.rand((10,))
        self.nebula.compute_loss(y_pred, y_true)

    def test_non_negative_integers_in_y_true(self):
        y_true = torch.arange(10)
        y_pred = torch.rand((10,))
        self.nebula.compute_loss(y_pred, y_true)

    def test_sparse_tensor(self):
        y_true = torch.zeros((10, 10))
        y_pred = torch.rand((10, 10))
        self.nebula.compute_loss(y_pred, y_true)

    def test_dense_tensor(self):
        y_true = torch.ones((10, 10))
        y_pred = torch.rand((10, 10))
        self.nebula.compute_loss(y_pred, y_true)

    def test_probability_distributions(self):
        y_true = torch.FloatTensor([0.1, 0.2, 0.3, 0.4])
        y_pred = torch.FloatTensor([0.1, 0.2, 0.3, 0.4])
        self.nebula.compute_loss(y_pred, y_true)

    def test_log_probabilities(self):
        y_true = torch.randn((10, 10))
        y_pred = torch.log_softmax(torch.randn((10, 10)), dim=1)
        self.nebula.compute_loss(y_pred, y_true)

    def test_domain_knowledge_classification(self):
        y_true = torch.randint(0, 2, (10,))
        y_pred = torch.rand((10,))
        self.nebula.domain_knowledge = "classification"
        self.nebula.compute_loss(y_pred, y_true)

    def test_domain_knowledge_regression(self):
        y_true = torch.randn((10,))
        y_pred = torch.rand((10,))
        self.nebula.domain_knowledge = "regression"
        self.nebula.compute_loss(y_pred, y_true)

    def test_user_input_classification(self):
        y_true = torch.randint(0, 2, (10,))
        y_pred = torch.rand((10,))
        self.nebula.user_input = "classification"
        self.nebula.compute_loss(y_pred, y_true)

    def test_user_input_regression(self):
        y_true = torch.randn((10,))
        y_pred = torch.rand((10,))
        self.nebula.user_input = "regression"
        self.nebula.compute_loss(y_pred, y_true)

    def test_y_true_values_in_range_0_1(self):
        y_true = torch.rand((10, 10))
        y_pred = torch.rand((10, 10))
        self.nebula.compute_loss(y_pred, y_true)

    def test_unbalanced_classes_in_y_true(self):
        y_true = torch.tensor([0, 0, 0, 0, 0, 0, 0, 0, 0, 1])
        y_pred = torch.rand((10,))
        self.nebula.compute_loss(y_pred, y_true)

    def test_balanced_classes_in_y_true(self):
        y_true = torch.tensor([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
        y_pred = torch.rand((10,))
        self.nebula.compute_loss(y_pred, y_true)

    def test_large_tensors(self):
        y_true = torch.rand((10000, 10000))
        y_pred = torch.rand((10000, 10000))
        self.nebula.compute_loss(y_pred, y_true)

    def test_multilabel_classification_tensor(self):
        y_true = torch.tensor([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        y_pred = torch.rand((3, 3))
        self.nebula.compute_loss(y_pred, y_true)

    def test_y_pred_max_value_greater_than_0_9(self):
        y_true = torch.rand((10,))
        y_pred = torch.rand((10,)) + 0.9
        self.nebula.compute_loss(y_pred, y_true)

    def test_loss_function_reuse_from_cache(self):
        y_true = torch.rand((10,))
        y_pred = torch.rand((10,))
        self.nebula.compute_loss(y_pred, y_true)
        self.assertEqual(id(self.nebula.loss_function), id(self.nebula.loss_function_cache[id(y_true)]))

    def test_nebula_for_regression(self):
        nebula_loss_function = Nebula(domain_knowledge="regression")
        loss = nebula_loss_function.compute_loss(self.y_pred_regression, self.y_true_regression)
        expected_loss = MSELoss().compute_loss(self.y_pred_regression, self.y_true_regression)
        self.assertTrue(torch.isclose(loss, expected_loss, atol=self.tolerance))

    def test_nebula_for_classification(self):
        nebula_loss_function = Nebula(domain_knowledge="classification")
        loss = nebula_loss_function.compute_loss(self.y_pred_classification, self.y_true_classification)
        expected_loss = CrossEntropyLoss().compute_loss(self.y_pred_classification, self.y_true_classification)
        self.assertTrue(torch.isclose(loss, expected_loss, atol=self.tolerance))

    def test_nebula_for_multi_label_classification(self):
        # For multi-label classification, let's assume each instance can belong to any of the 3 classes
        y_true_multi_label_classification = torch.tensor([[1, 0, 1], [0, 1, 0], [1, 1, 0], [0, 0, 1], [1, 0, 0]], dtype=torch.float)
        y_pred_multi_label_classification = torch.rand((5, 3), dtype=torch.float)  # Random probabilities for classes

        nebula_loss_function = Nebula()
        loss = nebula_loss_function.compute_loss(y_pred_multi_label_classification, y_true_multi_label_classification)
        expected_loss = MultiLabelSoftMarginLoss().compute_loss(y_pred_multi_label_classification, y_true_multi_label_classification)
        self.assertTrue(torch.isclose(loss, expected_loss, atol=self.tolerance))

    # Add more tests for other scenarios...
if __name__ == "__main__":
    unittest.main()