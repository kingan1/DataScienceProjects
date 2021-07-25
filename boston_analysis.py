import pandas as pd
from sklearn.datasets import load_boston
from sklearn.linear_model import LinearRegression, RidgeCV, SGDRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
import random
import numpy as np
import matplotlib.pyplot as plt

random.seed(1)
np.random.seed(1)

dataset_boston = load_boston()

# Make the data a pandas dataframe
X = pd.DataFrame(dataset_boston.data)
X.columns = dataset_boston.feature_names

# Splitting dataframe to X and y
y = dataset_boston.target

# Split into training and testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)
# Will store the results for all models
scores = {}

"""
    Tests the given model. Firs the model with training data,
    predicts on the X test data. Prints out a summary of the 
    accuracy of the model and then adds to global dictionary

"""


def test_model(model, model_name):
    model.fit(X_train, y_train)
    pred_y = model.predict(X_test)

    MSE = mean_squared_error(y_test, pred_y)
    MAE = mean_absolute_error(y_test, pred_y)
    r2 = r2_score(y_test, pred_y)

    print(f"\n\t{model_name}\n")
    print(f'MSE for {model_name} on test: {MSE}')
    print(f'MAE for {model_name} on test: {MAE}')
    print(f'R2 for {model_name} on test: {r2}')

    model_result = {'MSE': MSE, 'MAE': MAE, 'R2': r2}

    scores[model_name] = model_result
    return pred_y


# Exploration
print(f'Number of instances: {len(X.index)}')
print(f'Columns in dataframe: {X.columns}')

print(f'Target variable is Median value.')
print(f'Avg: {sum(y) / len(y)}, Min: {min(y)}, Max: {max(y)}')

# Exploring the X variables
print(f'Types of explanatory variables: f{X.dtypes}')

##################################
#       Building our model       #
##################################

# Linear Regression
lr = LinearRegression()
pred_y_lr = test_model(lr, "Linear Regression")

# Ridge Regression
ridge_model = RidgeCV()
pred_y_ridge = test_model(ridge_model, "Ridge Regression")


sgd_model = make_pipeline(StandardScaler(),
                          SGDRegressor())
pred_y_sgd = test_model(sgd_model, "SGD Regressor")
# More advanced regression techniques

print(pd.DataFrame(scores).T)

# take the best model and see what went wrong

def graph_expected_actual(y_pred, model_name):
    plt.figure()
    plt.scatter(y_test, y_pred)
    true_min, true_max = min(min(y_pred), min(y_test)), max(max(y_pred), max(y_test))
    plt.xlim(true_min-10, true_max+10)
    plt.ylim(true_min-10, true_max+10)
    plt.plot(true_min, true_max, 'k-', color = 'r')
    # plt.ylim(y_lim)
    # plt.xlim(x_lim)
    plt.show()
    plt.title(f"Expected vs actual for {model_name}")
    plt.show()

graph_expected_actual(pred_y_ridge, "Ridge Regression")