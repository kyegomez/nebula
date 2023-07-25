import torch 
import torch.nn as nn
import numpy as np


#define the loss function class
class LossFunction:
    def compute_loss(self, y_pred, y_true):
        raise NotImplemented("compute_loss method must be implemented")
    

#implement specific loss functions that inherit from LossFunction
class L1Loss(LossFunction):
    def __init__(self):
        self.loss_function = nn.L1Loss()

    def compute_loss(self, y_pred, y_true):
        return self.loss_function(y_pred, y_true)
    

class MSELoss(LossFunction):
    def __init__(self):
        self.loss_function = nn.MSELoss()

    def compute_loss(self, y_pred, y_true):
        return self.loss_function(y_pred, y_true)
    

class CrossEntropyLoss(LossFunction):
    def __init__(self):
        self.loss_function = nn.CrossEntropyLoss()

    def compute_loss(self, y_pred, y_true):
        return self.loss_function(y_pred, y_true)
    

"""

all pytorch loss functions

"""

def one_hot_encoding(y_true, num_classes):
    y_true_one_hot = torch.zeros(y_true.size(0), num_classes)
    y_true_one_hot.scatter_(1, y_true.unsqueeze(1), 1)
    return y_true_one_hot

class Nebula(LossFunction):
    def __init__(self, domain_knowledge=None, user_input=None):
        self.loss_function = None
        self.domain_knowledge = domain_knowledge
        self.user_input = user_input
        self.loss_function_cache = {}


    def determine_loss_function(self, y_pred, y_true):
        is_classification = None

        #op 1 check range of values in y_true
        if is_classification is None:
            unique_values = np.unique(y_true)
            if len(unique_values) <= 10 and np.all(np.equal(np.mod(unique_values, 1), 0)): 
                is_classification = True

        #==============================================>

        

        if is_classification is None:
            value_counts = np.bincount(y_true.flatten().to(dtype=torch.int32).numpy())
            if np.all(value_counts > 0):
                is_classification = True
        

        #==============================================>


        #op 3 analyze the dimension of y_pred
        if y_pred.ndim > 2:
             #handle segmentation problem
          pass

        #==============================================>

        if is_classification is None:
            sparsity = np.count_nonzero(y_true) / y_true.numel()
            if sparsity < 0.5:
                self.loss_functions = torch.nn.BCEWithLogitsLoss()
                self.compute_Loss = self.loss_function
                return

        #==============================================>

        #v4
        y_pred_flat = y_pred.flatten().numpy()
        y_true_flat = y_true.flatten().numpy()
        if y_pred_flat.shape != y_true_flat.shape:
            y_pred_flat = y_pred_flat[:y_true_flat.size]
        correlation = np.corrcoef(y_pred_flat, y_true_flat)[0, 1]





        #==============================================>


        if is_classification is None:
            if self.domain_knowledge == "classification":
                is_classification = True
            elif self.domain_knowledge == "regression":
                is_classification = False

            
        #==============================================>



        #v3
        if is_classification is None:
            if torch.max(y_pred) > 0.9:
                is_classification = True

        #==============================================>

        


        #v2
        if is_classification is None:
            class_balance = value_counts / np.sum(value_counts)
            if np.any(class_balance < 0.1):
                is_classification = True

        #==============================================>


        #==============================================>



        #v2
        if is_classification is None:
            if self.user_input == "classification":
                is_classification = True
            elif self.user_input == "regression":
                is_classification = False

        #set the loss function based on the determined problem type
        if is_classification:
            self.loss_function = CrossEntropyLoss()
        else:
            self.loss_function = MSELoss()

    def compute_loss(self, y_pred, y_true):
        dataset_id = id(y_true)
        if dataset_id not in self.loss_function_cache:  # Fix the attribute name here
            self.determine_loss_function(y_pred, y_true)
            self.loss_function_cache[dataset_id] = self.loss_function

        cached_loss_function = self.loss_function_cache[dataset_id]
        return cached_loss_function.compute_loss(y_pred, y_true)

    


    
        