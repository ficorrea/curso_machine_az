# Regression Template

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv('02 - Regressão/Position_Salaries.csv')
X = dataset.iloc[:, 1:2].values
y = dataset.iloc[:, 2].values

# Fitting 
# Criar o regressor

# Predicting with Polynomial
y_pred = regressor.predict(6.5)

# Visualising Polynomial Regression
plt.scatter(X, y, color='red')
plt.plot(X, regressor.predict(X), color='blue')
plt.title('')
plt.xlabel('Position')
plt.ylabel('Salary')
plt.show()

# Para melhor resolução
X_grid = np.arange(min(X), max(X), 0.1)   
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(X, y, color='red')
plt.plot(X_grid, regressor.predict(X_grid), color='blue')
plt.title('')
plt.xlabel('Position')
plt.ylabel('Salary')
plt.show()