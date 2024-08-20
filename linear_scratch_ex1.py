import pandas as pd
import matplotlib.pyplot as plt
data=pd.read_csv('Experience-Salary.csv')
print(data.head())
def remove_outlier(data,column):
    Q1=data[column].quantile(0.25)
    Q3=data[column].quantile(0.75)
    IQR=Q3-Q1
    upper_limit=Q3+1.5*IQR
    lower_limit=Q1-1.5*IQR
    data_filtered=data[(data[column]<=upper_limit) & (data[column] >= lower_limit)]
    return data_filtered
def gradient_descent(m_now,b_now,L,points):
    n=len(points)
    m_gradient,b_gradient=0,0
    for i in range(n):
        x = points['exp(in months)'].iloc[i]
        y = points['salary(in thousands)'].iloc[i]
        m_gradient+=-1/n*x*(y-(m_now*x+b_now))
        b_gradient+= -1/n * (y - (m_now * x + b_now))
    m=m_now-L*m_gradient
    b=b_now-L*b_gradient
    return m,b

def cost_function(m,b,data):
    n=len(data)
    mean_error=0
    for i in range(n):
        x=data['salary(in thousands)'].iloc[i]
        y=data['exp(in months)'].iloc[i]
        mean_error+=(y-(m*x+b))**2
    mean_error/=float(2*n)
    return mean_error

data_cleaned=remove_outlier(data,'exp(in months)')
data_cleaned=remove_outlier(data_cleaned,'salary(in thousands)')
print(data_cleaned.describe())
print(data_cleaned.isnull().sum())
m,b,L,repeat=0.5,3,0.001,301
error=cost_function(m,b,data_cleaned)
print(f"Cost Function Error= {error}")
for i in range(repeat):
    m,b=gradient_descent(m,b,L,data_cleaned)
    if (i%50==0):
        error = cost_function(m, b, data_cleaned)
        print(f"Cost Function Error at iteration {i}= {error:.3f}")
print(f'm= {m} and b= {b}')
x=data_cleaned['exp(in months)'].tolist()
y=data_cleaned['salary(in thousands)'].tolist()
y_predict=[m*x+b for x in x]
plt.scatter(x,y,color='black')
plt.plot(x,y_predict,color='red')
plt.xlabel('experience (months)')
plt.ylabel('salary (thousands)')
plt.title('Experience Against Salary')
plt.show()


