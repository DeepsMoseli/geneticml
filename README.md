# geneticml

## About
 This library uses genetic algorithms to for automated hyper-parameter optimization in Machine learning algorithms

## Supported Models 
Currently supports RandomForestClassifier, GradientBoostingClassifier, LogisticRegression, MLPClassifier.<br>
__coming soon:__ xgboost, lightboost, catboost, regression models and metrics.

## Usage 

```python
pip install -i https://test.pypi.org/simple/ geneticml
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

### One plus one EA with Gradient boosting

```python

from  geneticml.variation_operators import one_plus_one
from sklearn.ensemble import GradientBoostingClassifier

test2 = one_plus_one(x_train,
	y_train, 
	GradientBoostingClassifier, 
	improvement = 0.1, 
	mutation_prob = 0.9, 
	email=False)

test2.Main()

#best model
test2.best
```


