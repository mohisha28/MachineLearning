#importing libraries
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import pylab as pl

#loading dataset & displaying
df=pd.read_csv('student.csv')
print(student.head(5))

#select features that we want to use for regression
cdf=df[['classes','quizes','marks']]
cdf.head()

plt.scatter(cdf.classes, cdf.marks, color='blue')
plt.xlabel('classes')
plt.ylabel('marks')
plt.show()

plt.scatter(cdf.quizes, cdf.marks, color='red')
plt.xlabel('quizes')
plt.ylabel('marks')
plt.show()

from sklearn import linear_model
regr = linear_model.LinearRegression()
x = df[['classes','quizes']]
y = df[['marks']]
regr.fit(x,y)

print('Coefficients: ', regr.coef_)

#predict marks when he attends 2 classes & 2 quizzes
predictedmarks = regr.predict([[2,2]])
print(predictedmarks)
