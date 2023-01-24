import pandas as pd 
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split

# read the data
data = pd.read_csv("book1.csv")

# select the relevant columns
data = data[["average", "absent", "studyhr", "advice"]]

# separate the predictors and the target variable
x = data[["average", "absent", "studyhr"]]
y = data["advice"]

# split the data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1)

# initialize the decision tree classifier
knn = KNeighborsClassifier()

# fit the model on the training data
knn.fit(x_train, y_train)

# make predictions on the test data
y_pred = knn.predict(x_test)

# evaluate the model performance
print("Accuracy: ", accuracy_score(y_test, y_pred))
#print("Confusion Matrix: \n", confusion_matrix(y_test, y_pred))

'''# create new data for predictions
new_data = [[8,10,6]]

# make predictions on new data
predictions = knn.predict(new_data)

#print the predictions
print(predictions)'''