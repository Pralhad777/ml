'''linear model:- 
'''


#importing all the modules that needed
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn

'''to save best model'''
import pickle 

from sklearn import linear_model
from sklearn.utils import shuffle
from matplotlib import style

#loading the data
data = pd.read_csv("student-mat.csv", sep=";")
print(data.head())

#using only the require feature
data = data[["G1", "G2", "G3", "studytime", "failures", "absences"]]
print(data.head())
print(data.info())

#separating the target and predictors value
predict = "G3"
x = np.array(data[["G1", "G2", "studytime", "failures", "absences"]])
y = np.array(data[[predict]])

#spliting the data randomly in train and test
x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size=0.1)

best = 0.0
#fitting the data
linear = linear_model.LinearRegression()
linear.fit(x_train, y_train)
accuracy = linear.score(x_test, y_test)


#checking the model
print(accuracy)
print('Coefficients:\t', linear.coef_)
print('intercept:\t',linear.intercept_)

predictions = linear.predict(x_test)

for x in range(len(predictions)):
    print(predictions[x],x_test[x],y_test[x] )
print(best)

style.use("ggplot")
plt.scatter(data["G2"], data[predict])
plt.show()
