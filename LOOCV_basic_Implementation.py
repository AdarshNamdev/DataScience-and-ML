# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error as MSE
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import seaborn as sb

np.random.seed(1)
from sklearn.metrics import mean_squared_error as MSE

def LOOCV(exog, endog, model):           # "exog" is called 'exogeneous' and is the dataset wih only Independent Variable;
                                         # "endog" is called endogeneous and represent the Class/Target variable
                                         # "model" here represent the estimator used to fit training dataset.
    all_indices = list(exog.index)
    test_err = []
    for indx in all_indices:
        # Obtaining the test data
        x_test = exog.iloc[[indx]]
        y_test = endog.iloc[[indx]]
        
        # obtaining the train data
        train_indices = all_indices.copy()
        train_indices.remove(indx)
        x_train = exog.iloc[train_indices]
        y_train = endog.iloc[train_indices]
        
        # fittin model and predicting on test data
        model.fit(x_train, y_train)
        y_pred = model.predict(x_test)
        
        # computing the estimated test mse (Mean Squared Error)
        model_test_error = MSE(y_true = y_test, y_pred = y_pred)
        test_err.append(model_test_error)
        
        
    return np.mean(test_err)
    


Estimated_test_error_per_model = []
X = np.random.normal(loc = 0, scale = 1.0, size = 100)
Y = np.random.normal(loc = 0, scale = 1.0, size = 100)
df1 = pd.DataFrame({'X': X, 'Y': Y})
x = df1[['X']]
y = df1['Y']

# Model #1:
LinReg = LinearRegression()
mse = LOOCV(exog = x, endog = y, model = LinReg)
print("MSE for model 1: ", mse)
Estimated_test_error_per_model.append(mse)


df2 = pd.DataFrame({'X':X, 'X2': X**2, 'Y': Y})
x = df2[['X', 'X2']]
y = df2['Y']

# Model #2:
LinReg = LinearRegression()
mse = LOOCV(exog = x, endog = y, model = LinReg)
print("MSE for model 2: ", mse)
Estimated_test_error_per_model.append(mse)

df3 = pd.DataFrame({'X':X, 'X2': X**2, 'X3': X**3, 'Y': Y})
x = df3[['X', 'X2', 'X3']]
y = df3['Y']

# Model #3:
LinReg = LinearRegression()
mse = LOOCV(exog = x, endog = y, model = LinReg)
print("MSE for model 3: ", mse)
Estimated_test_error_per_model.append(mse)

df4 = pd.DataFrame({'X':X, 'X2': X**2, 'X3': X**3, 'X4': X**4, 'Y': Y})
x = df4[['X', 'X2', 'X3', 'X4']]
y = df4['Y']

# Model #4:
LinReg = LinearRegression()
mse = LOOCV(exog = x, endog = y, model = LinReg)
print("MSE for model 4: ", mse)
Estimated_test_error_per_model.append(mse)

df5 = pd.DataFrame({'X':X, 'X2': X**2, 'X3': X**3, 'X4': X**4, 'X5': X**5, 'Y': Y})
x = df5[['X', 'X2', 'X3', 'X4', 'X5']]
y = df5['Y']

# Model #5:
LinReg = LinearRegression()
mse = LOOCV(exog = x, endog = y, model = LinReg)
print("MSE for model 5: ", mse)
Estimated_test_error_per_model.append(mse)

df6 = pd.DataFrame({'X':X, 'X2': X**2, 'X3': X**3, 'X4': X**4, 'X5': X**5, 'X6': X**6, 'Y': Y})
x = df6[['X', 'X2', 'X3', 'X4', 'X5', 'X6']]
y = df6['Y']

# Model #6:
LinReg = LinearRegression()
mse = LOOCV(exog = x, endog = y, model = LinReg)
print("MSE for model 6: ", mse)
Estimated_test_error_per_model.append(mse)

df7 = pd.DataFrame({'X':X, 'X2': X**2, 'X3': X**3, 'X4': X**4, 'X5': X**5, 'X6': X**6, 'X7': X**7, 'Y': Y})
x = df7[['X', 'X2', 'X3', 'X4', 'X5', 'X6', 'X7']]
y = df7['Y']

# Model #7:
LinReg = LinearRegression()
mse = LOOCV(exog = x, endog = y, model = LinReg)
print("MSE for model 7: ", mse)
Estimated_test_error_per_model.append(mse)

plt.figure(figsize= (8, 6))
sb.set_style("darkgrid")
plt.xlabel("Degree Of Polynomial")
plt.ylabel("Mean Squared Error")
plt.scatter(x = np.arange(0, 7), y = Estimated_test_error_per_model, marker = "o")
plt.plot(Estimated_test_error_per_model, 'r-')
plt.xticks()
plt.show()