# geneticml

## About
 This library uses a number of genetic algorithms to help you with aoutomated hyper-parameter optimization for most algorithms in sklear and some famous lone-stading ones such as XGBoost, Catboost, LightGMB.

## Supported Models 
Eventually all models with the __.fit__, __.predict__, __.predict_proba__ methods will be suported. This does not include keras, pytorch or tensorflow 2.0 models.


## Usage 

```python
pip install geneticml
```

in your python script or ipynb

```python
from  geneticml.variation_operators import differential_evolution 
from sklearn.ensemble import RandomForestClassifier

#sample data
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
```

load your dataset in the X, y form. you can split your train and test data, but the code will create a validation set to test the evolution candidate's fitness.

```python
data =  load_breast_cancer()
X = data.data
y = data.target

x_train,x_test,y_train,y_test = train_test_split(X,y,test_size=0.25,random_state=45)
```

create the EA object

```python
test = differential_evolution(x_train,
								y_train, 
								RandomForestClassifier, 
								improvement = 0.1, 
								population_size=10,
								mutation_prob=0.13,
								elitism=0.15,
								crossover_prob=0.70)

```

run EA search (might take time depending on dataset size)

```python
test.Main()

#best model
test.best
```

To use the best fitted model on your test set and measure AUC score

```python
test_pred = test.best['best_fitted_model'].predict_proba(x_test)[:,1]
roc_auc_score(y_test,test_pred)
```
-----


