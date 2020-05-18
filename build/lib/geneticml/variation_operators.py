from sklearn.metrics import accuracy_score,roc_auc_score
from sklearn.model_selection import train_test_split
from geneticml.hyper_parameters import Estimator
from geneticml.email_train import Email_pypy

from tqdm import tqdm
import numpy as np
import pandas as pd
import pickle
import random
import warnings
warnings.simplefilter("ignore")
import sys


"""DE: """

class differential_evolution:
    def __init__(self, X, Y, algorithm, improvement, population_size, mutation_prob, elitism, crossover_prob, email = False):
        self.algorithm = algorithm
        self.algo_name = algorithm().__class__.__name__
        self.improvement=improvement
        self.population_size = population_size
        self.mutation_prob = mutation_prob
        self.elitism = elitism
        self.crossover_prob = crossover_prob
        self.population = []
        self.generation_params=[]
        self.new_population = []
        self.fitness = {}
        self.best = {"params":{},"score":0,'sklearn_score':0,'best_fitted_model':None}
        self.email = email

        self.target = 0.5 #initial target AUC equivalent to a random model
        self.max_gen = 50 #Max number of generations
        self.increase_thrashold = 0.2 #highest allowed improvement percentage
        self.decay_generations = 4 #change mutation probabilities after x generations
        self.nochange = 5 #stop process if there is no improvement in x generations
        self.X=X
        self.Y=Y
        self.x_train,self.x_test,self.y_train,self.y_test = train_test_split(self.X,self.Y,
                                                 test_size=0.2,random_state=42)
        self.find_target() #calculates taget AUC as percentage of default out of the box sklearn model

    def find_target(self):
        default = self.algorithm()
        ###test classification###
        target_count_binary = len(set(self.y_train))==len(set(self.y_test))==2
        if target_count_binary==True and self.improvement<=self.increase_thrashold:
            default.fit(self.x_train,self.y_train)
            pred = default.predict_proba(self.x_test)[:,1]
            self.best['sklearn_score'] = roc_auc_score(self.y_test,pred)
            self.target = min(1,self.best['sklearn_score']*(1+self.improvement))
            print("EA optomization target set to AUC score of %s"%self.target)
            print("----------------------------------------\n")
        else:
            sys.exit("Error: Only binary classification suported for less than %s improvement over base sklearn model."%self.thrashold)

        
    def random_genome(self):
        Model_instance = Estimator(self.algorithm, randomize=True)
        return Model_instance.params
    
    def mating(self,parent1,parent2):
        param_len = len(parent1)
        assert param_len == len(parent2),"Both parents should have the same gene count"
        offspring = {}
        prob = random.random()
        for k in parent1:
            if prob<=(1-self.mutation_prob)/2:
                offspring[k] = parent1[k]
            elif prob<=(1-self.mutation_prob):
                offspring[k] = parent2[k]
            else:
                offspring[k] = self.random_genome()[k]
        return offspring
    
    def calc_fitness(self,individual):
        try:
            model_eval = Estimator(self.algorithm, params=individual)
            model_eval.model.fit(self.x_train,self.y_train)
            pred = model_eval.model.predict_proba(self.x_test)[:,1]
            score = roc_auc_score(self.y_test,pred)
            return score
        except Exception as e:
            print(e)
            model_eval = Estimator(self.algorithm)
            model_eval.model.fit(self.x_train,self.y_train)
            pred = model_eval.model.predict_proba(self.x_test)[:,1]
            score = roc_auc_score(self.y_test,pred)
            return score
    
    def adaptive_probs(self):
        self.mutation_prob -= 0.001
        self.crossover_prob -= 0.01 
        self.elitism += 0.01
    
    def Main(self):
        
        #population init
        Converged = False
        Generation = 1;
        no_change_gens = 0
        for k in range(self.population_size):
            self.population.append(self.random_genome())
        self.generation_params.append(self.population)
        #----------------------------selection----------------------------
        pbar = tqdm(range(self.max_gen))
        
        while(Converged==False):
            print("Gen (%s)"%Generation)
            """calc fitness and do selection"""
            self.fitness["Generation %s"%Generation]= list(map(self.calc_fitness,self.population))
            
            sortedindexes =  list(np.flip(np.argsort(self.fitness["Generation %s"%Generation])))
            self.best['score'] = max(self.best['score'],pd.Series(self.fitness["Generation %s"%Generation])[sortedindexes[0]])
            self.best['params'] = pd.Series(self.population)[sortedindexes[0]]
            print("(best: %s, Avg: %s): "%(self.best['score'],np.mean(self.fitness["Generation %s"%Generation])))
            
            if Generation%4==0:
                message = """Gen: %s 
                SKlearn: %s 
                Best: %s"""%(Generation,self.best["sklearn_score"],self.best["score"])
                if self.email:
                	mail=Email_pypy(message)
            print("\n")
            
            """-elite top 10% straight to new generation
                -cross over for other 85%, only within top 50%
                -new entrants for last 5%
            """
            self.new_population.extend(list(pd.Series(self.population)[sortedindexes[:int(self.elitism*self.population_size)]]))
            self.new_population.extend([self.random_genome() for _ in range(int((1-self.crossover_prob-self.elitism)*self.population_size))])
            
            while(len(self.new_population)!=len(self.population)):
                """ Crossover from only the top 50% from previous population """
                p1=random.choice(list(pd.Series(self.population)[sortedindexes[:int(0.5*self.population_size)]]))
                p2=random.choice(list(pd.Series(self.population)[sortedindexes[:int(0.5*self.population_size)]]))
                self.new_population.append(self.mating(p1,p2))
            
            self.population=self.new_population
            self.new_population=[]
            self.generation_params.append(self.population)
            pbar.update(1)
            
            #Test stoping criteria, convergence, max-generations or no change, send mail for stop.
            if self.best['score']>=self.target or Generation>=self.max_gen or no_change_gens>self.nochange:
                Converged=True
                message = """Gen: %s 
                SKlearn: %s 
                Best: %s"""%("Converged",self.best["sklearn_score"],self.best["score"])
                if self.email:
                    mail=Email_pypy(message)
            else:
                if Generation>2 and abs(self.best["score"]-self.fitness["Generation %s"%Generation][-2])<0.001:
                    no_change_gens+=1
                
                Generation+=1
                
                if Generation%self.decay_generations==0:
                    self.adaptive_probs()
        pbar.close()
        print("Best model was fitted and returned.")
        self.best['best_fitted_model'] = self.algorithm(**self.best['params'])
        self.best['best_fitted_model'].fit(self.X,self.Y)
        del self.x_train,self.x_test,self.y_train,self.y_test, self.X, self.Y



##############################################################################




