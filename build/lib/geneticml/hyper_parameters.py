import numpy as np
import random
import warnings
from tqdm import tqdm
import pickle
warnings.simplefilter("ignore")
import sys

#classifiers
from sklearn.ensemble import RandomForestClassifier,GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC

#regressors
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor


class Estimator:
    """This class defines the model with random or passed hyper-parameters"""

    def __init__(self, algorithm, randomize = True, params = {}):
        self.algo_name = algorithm().__class__.__name__
        self.algorithm = algorithm
        if randomize==True and len(params)==0:
            try:
                self.params = self.lookup(self.algo_name)
            except Exception as e:
                print(e)
                print("The estimator %s is currently not supported."%self.algo_name)
                print("If you would really like to use this specific estimator, send us a mail as we will add it.")
                sys.exit("Exit Code: 1")
        else:
            self.params = params
        self.model = self.algorithm(**self.params)
        
    """------------------House keeping------------------------"""
    def estimator_name(self):
        return self.algorithm.__class__.__name__
    
    def rand_or_get(self,switch,param):
        if param is not None:
            return switch[param]
        else:
            return switch[np.random.randint(len(switch))]
        
    """-------------very common------------"""
    def learning_rate_init(self,lri=None):
        if lri is not None:
            return lri
        else:
            return np.random.random()/200

    ################################################################
    ################################################################
    """ --------Multi layer perceptron-------"""
    def hidden_layer_sizes(self,layers=None):
        if layers is not None:
            return layers
        else:
            return (np.random.randint(50,150),np.random.randint(5,50))
    
    def activation(self,activation = None):
        switch = {0:"identity",1:"logistic",2:"tanh",3:"relu"}
        return self.rand_or_get(switch,activation)
    
    def solver(self,solver_num = None):
        switch = {0:"sgd",1:"adam"}
        return self.rand_or_get(switch,solver_num)

    def alpha(self,alpha_num=None): #penalty term
        if alpha_num is not None:
            return alpha_num
        else:
            return np.random.random()/1000
        
    def learning_rate(self, lrstr = None):
        switch = {0:"constant",1:"invscaling",2:"adaptive"}
        return self.rand_or_get(switch,lrstr)

    def max_iter(self,iter = None):
        if iter is not None:
            return iter
        else:
            return np.random.randint(50,300)
    
    ################################################################
    ################################################################
    """ ---------Ensemble(ada, forrest, gradient, decisionTree )---------"""
    def n_estimators(self,n = None):
        if n is not None:
            return n
        else:
            return np.random.randint(20,300)

    def max_depth(self,depth = None):
        if depth is not None:
            return depth
        else:
            return np.random.randint(3,10)
    
    def max_features(self,feat = None):
        switch = {0:None,1:"auto",2:"log2",3:"sqrt"}
        return self.rand_or_get(switch, feat)

    def splitter(self, split = None):
        switch = {0:'best',1:'random'}
        return self.rand_or_get(switch, split)

    def criterion(self,crit = None):
        switch = {0:'mse',1:'friedman_mse', 2:'mae'}
        return self.rand_or_get(switch, crit)
    ######################################################################
    ######################################################################
    """-------------------Linear(Logistic,svm)------------------------"""
    def penalty(self,pen = None):
        switch = {0:"l1",1:"l2"}
        return self.rand_or_get(switch, pen)

    def C(self, c_value = None):
        if c_value is not None:
            return c_value
        else:
            return np.random.uniform(0.5,2)

    def dual(self,d_value = None):
        switch = {0:True,1:False}
        return self.rand_or_get(switch, d_value)

    ####################################################################
    ####################################################################
    """------------------LOOKUP TABLE--------------------"""
    def lookup(self, algo):
        lookuptable = {
            "RandomForestClassifier":{
                "n_estimators":self.n_estimators(),
                "max_depth":self.max_depth(),
                "max_features":self.max_features()
                },
            "GradientBoostingClassifier":{
                "n_estimators": self.n_estimators(),
                "max_depth":self.max_depth(),
                "max_features":self.max_features()
            },
            "MLPClassifier":{
                "hidden_layer_sizes": self.hidden_layer_sizes(),
                "learning_rate_init": self.learning_rate_init(),
                "learning_rate": self.learning_rate(),
                "max_iter": self.max_iter(),
                "solver": self.solver(),
                "activation": self.activation(),
                "alpha": self.alpha()
            },
            "LogisticRegression":{
                "C":self.C(),
                "penalty":self.penalty(),
                "max_iter":self.max_iter()
            },
            "LinearSVC":{
                "C": self.C(),
                "penalty":self.penalty(),
                "max_iter":self.max_iter(),
                "dual":self.dual()
            },
            "DecisionTreeRegressor":{
                "criterion":self.criterion(),
                "splitter": self.splitter(),
                "max_depth":self.max_depth(),
                "max_features":self.max_features()
            },
            "RandomForestRegressor":{
                "n_estimators":self.n_estimators(),
                "max_depth":self.max_depth(),
                "max_features":self.max_features()
            },
            "GradientBoostingRegressor":{
                "n_estimators": self.n_estimators(),
                "max_depth":self.max_depth(),
                "max_features":self.max_features()
            }
        }
        return lookuptable[algo]

#if __name__ == "__main__":
#run()
