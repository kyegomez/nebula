import unittest
import torch
from nebula.nebula import  MSELoss, CrossEntropyLoss, MultiLabelSoftMarginLoss,  Nebula

class TestNebula(unittest.TestCase):

    def setUp(self):
        self.tolerance = 1e-5
        self.y_true_regression = torch.tensor([1.1, 2.2, 3.3, 4.4, 5.5], dtype=torch.float)
        self.y_pred_regression = torch.tensor([1.0, 3.0, 2.0, 5.0, 4.0], dtype=torch.float)

        # Assuming 3 classes
        self.y_true_classification = torch.tensor([0, 2, 1, 0, 2], dtype=torch.long)
        self.y_pred_classification = torch.rand((5, 3), dtype=torch.float)  # Random probabilities for classes

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
