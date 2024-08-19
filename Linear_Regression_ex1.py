import matplotlib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
data=pd.read_csv('Experience-Salary.csv')
print(data.isnull().sum())
print(data.describe())
plt.figure(1)
plt.subplot(2,1,1)
plt.boxplot(data['exp(in months)'].tolist())
plt.subplot(2,1,2)
plt.boxplot(data['salary(in thousands)'].tolist())
plt.show()
Q1_1=data.describe()['exp(in months)']['25%']
Q3_1=data.describe()['exp(in months)']['75%']
Q1_2=data.describe()['salary(in thousands)']['25%']
Q3_2=data.describe()['salary(in thousands)']['75%']
IQR_1=data.describe()['exp(in months)']['75%']-data.describe()['exp(in months)']['25%']
IQR_2=data.describe()['salary(in thousands)']['75%']-data.describe()['exp(in months)']['25%']
data=data[(data['salary(in thousands)']<=Q3_2+1.5*IQR_2) & (data['salary(in thousands)']>=Q1_2-1.5*IQR_2)]
data=data[(data['exp(in months)']<=Q3_1+1.5*IQR_1) & (data['exp(in months)']>=Q1_1-1.5*IQR_1)]
print(data.count())
plt.figure(2)
plt.subplot(2,1,1)
plt.boxplot(data['exp(in months)'].tolist())
plt.subplot(2,1,2)
plt.boxplot(data['salary(in thousands)'].tolist())
plt.show()
x=data['exp(in months)'].tolist()
y=data['salary(in thousands)'].tolist()
x=np.array(x).reshape(-1,1)
y=np.array(y)
Model=LinearRegression()
Model.fit(x,y)
y_predict=Model.predict(x)
cost_function=mean_squared_error(y,y_predict)
r2 = r2_score(y, y_predict)
# Print model parameters and metrics
print("Coefficients:", Model.coef_)
print("Intercept:", Model.intercept_)
print("Mean Squared Error:", cost_function)
print("R^2 Score:", r2)
# Plotting the results
plt.scatter(x, y, color='blue', label='Data points')
plt.plot(x, y_predict, color='red', linewidth=2, label='Regression line')
plt.xlabel('Experience ( months )')
plt.ylabel('Salary ( thousands )')
plt.title('Linear Regression')
plt.legend()
plt.show()


