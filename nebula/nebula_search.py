import torch 
import torch.nn as nn
import numpy as np
from sklearn.model_selection import train_test_split

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(10, 32)
        self.fc2 = nn.Linear(32, 3)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        return torch.sigmoid(self.fc2(x))
    

class NebulaSearch:
    def __init__(self, model, loss_functions, num_iterations, lr):
        self.model = model
        self.loss_functions = loss_functions
        self.num_iterations = num_iterations
        self.lr = lr
        self.prob_dist = np.ones(len(loss_functions)) / len(loss_functions)

    def train(self, X, y):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)


        for i in range(self.num_iterations):
            chosen_loss_func = np.random.choice(self.loss_functions, p=self.prob_dist)

            self.model.zero_grad()

            outputs = self.model(X)
            loss = chosen_loss_func(outputs, y)
            loss.backward()
            optimizer.step()


            performance = self.evaluate(X, y)
            self.prob_dist = self.update_distribution(performance)

        print(f"Best loss function: {self.loss_functions[np.argmax(self.prob_dist)]}")

    def evaluate(self, X, y):
        with torch.no_grad():
            outputs = self.model(X)
            loss = self.loss_functions[np.argmax(self.prob_dist)](outputs, y)
            return -loss.item()
        
    def update_distribution(self, performance):
        self.prob_dist *= np.exp(performance)
        self.prob_dist /= np.sum(self.prob_dist)
        return self.prob_dist
    

#assume we have some data X, Y
X = torch.randn(100, 10)
y = torch.randint(0, 2, (100,))

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

loss_functions = [nn.CrossEntropyLoss(), nn.BCELoss(), nn.MSELoss()]
model = Net()

nebula = NebulaSearch(model, loss_functions, num_iterations=100, lr=0.001)
nebula.train(X_train, y_train)
