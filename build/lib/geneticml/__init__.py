import numpy as np
import pandas as pd
from sklearn import model_selection
from sklearn.metrics import accuracy_score, roc_auc_score, log_loss, mean_squared_error
from geneticml.genetic_algorithms import algorithms
from geneticml.params import params 

def Main():
    param1=params()
    algorithm1 = algorithms()
    param1.hello()
    algorithm1.hello()

if __name__=="__main__":
    Main()