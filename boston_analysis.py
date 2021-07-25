import pandas as pd
from sklearn.datasets import load_boston
from sklearn.linear_model import LinearRegression, RidgeCV, SGDRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

dataset_boston = load_boston()

# Make the data a pandas dataframe
df = pd.DataFrame(dataset_boston.data)
df.columns = dataset_boston.feature_names

# Exploration
print(f'Number of instances: {len(df.index)}')
print(f'Columns in dataframe: {df.columns}')

# Splitting dataframe to X and y
y = dataset_boston.target

print(f'Target variable is Median value.')
print(f'Avg: {sum(y)/len(y)}, Min: {min(y)}, Max: {max(y)}')

# Exploring the X variables
print(f'Types of explanatory variables: f{df.dtypes}')

##################################
#       Building our model       #
##################################

# variables to keep track of scores over all models
scores = {}

lr = LinearRegression()
lr.fit(df, y)
pred = lr.predict(df)

print("\n\tLinear Regression\n")
print(f'MSE for linear model on test: {mean_squared_error(y, pred)}')
print(f'MAE for linear model on test: {mean_absolute_error(y, pred)}')
print(f'R2 for linear model on test: {r2_score(y, pred)}')

lr_result = {'MSE': mean_squared_error(y, pred), 'MAE': mean_absolute_error(y, pred), 'R2': r2_score(y, pred)}

scores['lr'] = lr_result

ridge_model = RidgeCV()
ridge_model.fit(df, y)
pred = ridge_model.predict(df)

print("\n\tRidge Regression\n")
print(f'MSE for ridge model on test: {mean_squared_error(y, pred)}')
print(f'MAE for ridge model on test: {mean_absolute_error(y, pred)}')
print(f'R2 for ridge model on test: {r2_score(y, pred)}')

ridge_result = {'MSE': mean_squared_error(y, pred), 'MAE': mean_absolute_error(y, pred), 'R2': r2_score(y, pred)}

scores['ridge'] = ridge_result


from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
sgd_model = make_pipeline(StandardScaler(),
                    SGDRegressor())
sgd_model.fit(df, y)
pred = sgd_model.predict(df)

print("\n\tSGD Regression\n")
print(f'MSE for sgd model on test: {mean_squared_error(y, pred)}')
print(f'MAE for sgd model on test: {mean_absolute_error(y, pred)}')
print(f'R2 for sgd model on test: {r2_score(y, pred)}')

sgd_result = {'MSE': mean_squared_error(y, pred), 'MAE': mean_absolute_error(y, pred), 'R2': r2_score(y, pred)}

scores['sgd'] = sgd_result
# More advanced regression techniques