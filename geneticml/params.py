# Sklearn parameters to tune per model

class params:
    def __init__(self):
        self.param_list = ["RandomForest",
                            "DecisionTree",
                            "LogisticRegression",
                            "Adaboost",
                            "CatBoost",
                            "XGBoost",
                            "LightGMB"]




    def hello(self):
        print("Hello from params")


    