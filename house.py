# Importing libraraies
import numpy as np 
import matplotlib.pyplot as plt
import pandas as pd

# Importing dataset
dataset=pd.read_csv(r'C:\Users\velug\OneDrive\Desktop\Data Science\SEP\sep 20th- slr\20th- slr\SLR - House price prediction\House_data.csv')
space = dataset['sqft_living']
price = dataset['price']

x = np.array(space).reshape(-1, 1)
y = np.array(price)

#Splitting the data into train And test
from sklearn.model_selection import train_test_split
xtrain,xtest,ytrain,ytest=train_test_split(x,y,test_size=1/3,random_state=0)

#Fitting simple linear regression to the Training Set
from sklearn.linear_model import LinearRegression 
regressor = LinearRegression()
regressor.fit(xtrain, ytrain)

#Predicting the prices
pred=regressor.predict(xtest)


#Visulizing the training test results
plt.scatter(xtrain,ytrain,color='green')
plt.plot(xtrain,regressor.predict(xtrain),color='black')
plt.title("Visuals for Training dataset")
plt.xlabel("Space")
plt.ylabel("Price")
plt.show()


#Visualizing the Test Results 
plt.scatter(xtest, ytest, color= 'red')
plt.plot(xtrain, regressor.predict(xtrain), color = 'blue')
plt.title("Visuals for Test DataSet")
plt.xlabel("Space")
plt.ylabel("Price")
plt.show()