import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

#pd.read_excel for excel docs
dataset = pd.read_csv('Salary_Data.csv')
x = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

#print(x)
#print(y)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 0)

#print(x_test)

#simple line regressor 
regressor = LinearRegression()
regressor.fit(x_train, y_train)

y_pred = regressor.predict(x_test)

#plot training data. Result should be a nice line in which we can predict an expected result
plt.scatter(x_train, y_train, color = 'red')
plt.plot(x_train, regressor.predict(x_train), color = 'blue')
plt.title("Salary vs Experience (Trg Set)")
plt.xlabel("Years of Exp")
plt.ylabel("Salary")
plt.show()

#plot test data
plt.scatter(x_test, y_test, color = 'red')
#can plot the training or test data, they should be exactly the same
plt.plot(x_train, regressor.predict(x_train), color = 'blue')
plt.title("Salary vs Experience (Test Set)")
plt.xlabel("Years of Exp")
plt.ylabel("Salary")
plt.show()
