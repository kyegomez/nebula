import torch
import torch.nn as nn
import numpy as np

# Continual Learning Mechanism Class
class ContinualLearningMechanism(nn.Module):
    def __init__(self, pretrained_model=None):
        super(ContinualLearningMechanism, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 10),
        )
        if pretrained_model:
            self.model.load_state_dict(pretrained_model.state_dict())

    def forward(self, x):
        return self.model(x)


# Contrastive Learning Component Class
class ContrastiveLearningComponent(nn.Module):
    def __init__(self):
        super(ContrastiveLearningComponent, self).__init__()

    def forward(self, x, x_augmented):
        return torch.norm(x - x_augmented, p=2)


# Meta Learner Class
class MetaLearner(nn.Module):
    def __init__(self):
        super(MetaLearner, self).__init__()
        self.l1_loss = nn.L1Loss()
        self.l2_loss = nn.MSELoss()

    def forward(self, x, y, task):
        if task == 'task1':
            return self.l1_loss(x, y)
        elif task == 'task2':
            return self.l2_loss(x, y)


# Nebula Class
class Nebula(nn.Module):
    def __init__(self):
        super(Nebula, self).__init__()
        self.learning_mechanism = ContinualLearningMechanism()
        self.contrastive_component = ContrastiveLearningComponent()
        self.metalearner = MetaLearner()

    def forward(self, x, y, x_augmented, task):
        output = self.learning_mechanism(x)
        loss_task = self.metalearner(output, y, task)
        loss_contrastive = self.contrastive_component(x, x_augmented)

        # Here is where we combine the losses. The alpha and beta parameters
        # could be additional parameters to the class that could be learned
        alpha = 0.5
        beta = 0.5
        total_loss = alpha * loss_task + beta * loss_contrastive

        return total_loss