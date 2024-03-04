# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the standard Libraries
2. Set variables for assigning dataset values.
3. .Import linear regression from sklearn.
4. .Assign the points for representing in the graph.

## Program:
```
/*
Program to implement the simple linear regression model for predicting the marks scored.
Developed by: DHARSAN KUMAR R
RegisterNumber:212223240028
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error,mean_squared_error
df=pd.read_csv('student_scores.csv')
df.head()
df.tail()
x = df.iloc[:,:-1].values
x
y = df.iloc[:,1].values
y
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=1/3,random_state=0)
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train,y_train)
y_pred = regressor.predict(x_test)
y_pred
y_test
#Graph plot for training data
plt.scatter(x_train,y_train,color='black')
plt.plot(x_train,regressor.predict(x_train),color='purple')
plt.title("Hours vs Scores(Training set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()
#Graph plot for test data
plt.scatter(x_test,y_test,color='red')
plt.plot(x_train,regressor.predict(x_train),color='blue')
plt.title("Hours vs Scores(Testing set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()
mse=mean_absolute_error(y_test,y_pred)
print('MSE = ',mse)
mae=mean_absolute_error(y_test,y_pred)
print('MAE = ',mae)
rmse=np.sqrt(mse)
print("RMSE= ",rmse) 
*/
```

## Output:
df.head()
![image](https://github.com/DHARSAN23014208/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/149365413/55818a2e-0e31-4029-bbb2-47a43d536809)
df.tail()
![image](https://github.com/DHARSAN23014208/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/149365413/4e4ef5ab-86c2-4639-8ca1-37ce036c47c8)
Array value of X
![image](https://github.com/DHARSAN23014208/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/149365413/5388da20-0c01-418b-9a9c-e544a4a1f25c)
Array value of Y
![image](https://github.com/DHARSAN23014208/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/149365413/75698fe1-334b-42c2-9575-a0c71475009f)
Values of Y prediction
![image](https://github.com/DHARSAN23014208/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/149365413/b86310c9-ed2c-47d7-a633-e86d9df3a602)
Array values of Y test
![image](https://github.com/DHARSAN23014208/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/149365413/2a2e3830-03f6-4f55-9468-057914ef4bbe)
Training Set Graph
![image](https://github.com/DHARSAN23014208/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/149365413/ce4c4a8b-07b7-4885-b684-1a7ef1f21e7e)
Test Set Graph
![image](https://github.com/DHARSAN23014208/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/149365413/c92ea562-783d-4efc-aeeb-3910069b058e)
Values of MSE, MAE and RMSE
![image](https://github.com/DHARSAN23014208/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/149365413/517293a1-eac6-46a8-aea5-16965e85a80a)

## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
