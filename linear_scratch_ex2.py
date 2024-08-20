###################### Start ##################################
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
data=pd.read_csv('Experience-Salary.csv')
print(data.head())

######################### Functions ###################################
def remove_outlier(data,column):
    Q1=data[column].quantile(0.25)
    Q3=data[column].quantile(0.75)
    IQR=Q3-Q1
    upper_limit=Q3+1.5*IQR
    lower_limit=Q1-1.5*IQR
    data_filtered=data[(data[column]<=upper_limit) & (data[column] >= lower_limit)]
    return data_filtered
def gradient_descent(m_now,b_now,L,x,y):
    n=len(x)
    m_gradient,b_gradient=0,0
    for i in range(n):
        m_gradient+=-1/n*x[i]*(y[i]-(m_now*x[i]+b_now))
        b_gradient+= -1/n * (y[i] - (m_now * x[i] + b_now))
    m=m_now-L*m_gradient
    b=b_now-L*b_gradient
    return m[0],b[0]

def cost_function(m,b,x,y):
    n=len(x)
    mean_error=0
    for i in range(n):
        mean_error+=(y[i]-(m*x[i]+b))**2
    mean_error/=float(2*n)
    return mean_error[0]

#################### Plotting Outliers ############################

plt.subplot(2,2,1)
plt.boxplot(data['exp(in months)'])
plt.title('experience before cleaning')
plt.subplot(2,2,3)
plt.boxplot(data['salary(in thousands)'])
plt.title('Salary before cleaning')
plt.subplots_adjust(hspace=0.4)
data_cleaned=remove_outlier(data,'exp(in months)')
data_cleaned=remove_outlier(data_cleaned,'salary(in thousands)')
plt.subplot(2,2,2)
plt.boxplot(data_cleaned['exp(in months)'])
plt.title('experience after cleaning')
plt.subplot(2,2,4)
plt.boxplot(data_cleaned['salary(in thousands)'])
plt.title('Salary after cleaning')
plt.subplots_adjust(wspace=0.35)
plt.show()
print(data_cleaned.describe())
print(data_cleaned.isnull().sum())

######################### Data Splitting and Gradient Descent ###############################

x=data_cleaned['exp(in months)'].values.reshape(-1,1)
y=data_cleaned['salary(in thousands)'].values
X_Train,X_Test,Y_Train,Y_Test=train_test_split(x,y,test_size=0.2,random_state=1)
m,b,L,repeat=0.5,3,0.001,1001
error=cost_function(m, b, X_Train,Y_Train)
print(f"Cost Function Error= {error}")
losses=[]
for i in range(repeat):
    m,b=gradient_descent(m,b,L,X_Train,Y_Train)
    if (i%1==0):
        error=cost_function(m, b, X_Train,Y_Train)
        losses.append(error)
        #print(f"Cost Function Error at iteration {i}= {error:.3f}")
print(f'm= {m} and b= {b}')
y_predict=[m*x+b for x in X_Train]

################### Plotting Regression Line and Test points #########################

r2 = r2_score(Y_Train, y_predict)
print(f"The Cost Function Error Of Test Values= {cost_function(m, b, X_Test,Y_Test):.3f}") # Accuracy
print(f"R^2 Score: {r2:.3f}")
plt.scatter(X_Train,Y_Train,color='blue',label='Train Data')
plt.scatter(X_Test,Y_Test,color='green',label='Test Data')
plt.plot(X_Train,y_predict,color='red')
plt.legend()
plt.xlabel('experience (months)')
plt.ylabel('salary (thousands)')
plt.title('Experience Against Salary')
plt.show()

################################# Plotting Lossy Function ########################################
"""plt.plot(losses)
plt.xlabel('Iteration')
plt.ylabel('Loss')
plt.title('Loss Function of Linear Regression During Gradient Descent')
plt.show()"""

fig, (ax1, ax2) = plt.subplots(1, 2, constrained_layout=True, figsize=(12,4))
ax1.plot(losses[:100])
ax2.plot(100 + np.arange(len(losses[100:])), losses[100:])
ax1.set_title("Cost vs. iteration(start)");  ax2.set_title("Cost vs. iteration (end)")
ax1.set_ylabel('Cost')            ;  ax2.set_ylabel('Cost')
ax1.set_xlabel('iteration step')  ;  ax2.set_xlabel('iteration step')
plt.show()

########################################## End ##############################################
