# !pip install deap
import random
import torch
import torch.nn as nn
import torch.optim as optim

import numpy as np
from deap import creator, base, tools, algorithms

# Create the model
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(10, 32)
        self.fc2 = nn.Linear(32, 2)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        return torch.sigmoid(self.fc2(x))

# Define a simple neural network for loss function
class LossNet(nn.Module):
    def __init__(self):
        super(LossNet, self).__init__()
        self.fc1 = nn.Linear(2, 10)
        self.fc2 = nn.Linear(10, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        return torch.relu(self.fc2(x))

class Nebula:
    def __init__(self, model, loss_function, toolbox, X, y, population_size=10, num_generations=100):
        self.model = model
        self.loss_function = loss_function
        self.toolbox = toolbox
        self.X = X
        self.y = y
        self.population_size = population_size
        self.num_generations = num_generations
        self.toolbox.register("evaluate", self.evaluate)
        self.hof = tools.HallOfFame(1)
        self.stats = tools.Statistics(lambda ind: ind.fitness.values)
        self.stats.register("avg", np.mean)
        self.stats.register("min", np.min)
        self.stats.register("max", np.max)

    def evaluate(self, individual):
        weights = torch.Tensor(individual).view(self.loss_function.fc1.weight.shape)
        with torch.no_grad():
            self.loss_function.fc1.weight.data = weights
            output = self.model(self.X)
            loss = self.loss_function(output, self.y)
        return loss.item(),

    def train(self):
        pop = self.toolbox.population(n=self.population_size)
        pop, logbook = algorithms.eaSimple(pop, self.toolbox, cxpb=0.5, mutpb=0.2, ngen=self.num_generations, stats=self.stats, halloffame=self.hof, verbose=True)
        print(f'Best loss function: {self.hof[0]}')

# Initialize the model and the loss function
model = Net()
loss_function = LossNet()
toolbox = base.Toolbox()

# Define genetic algorithm related settings
creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Individual", list, fitness=creator.FitnessMin)
toolbox.register("attr_float", random.random)
toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_float, n=100)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("mate", tools.cxTwoPoint)
toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=1, indpb=0.1)
toolbox.register("select", tools.selTournament, tournsize=3)

# Assume we have some data X, y
X = torch.randn(100, 10)
y = torch.randint(0, 2, (100,))

# Create Nebula instance and train
nebula = Nebula(model, loss_function, toolbox, X, y)
nebula.train()
