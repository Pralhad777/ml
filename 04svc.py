#importing required model
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt

from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split


# read the data
data = pd.read_csv("book1.csv")

# select the relevant columns
data = data[["average", "absent", "studyhr", "advice"]]
print(data.info())

# separate the predictors and the target variable
x = data[["average", "absent", "studyhr"]]
y = data["advice"]

# split the data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(x, y)

# initialize the decision tree classifier
svc = SVC()

# fit the model on the training data
svc.fit(x_train, y_train)

# make predictions on the test data
y_pred = svc.predict(x_test)


# create new data for predictions
new_data = [[70, 5, 8], [55, 18, 5], [39, 4, 6]]

# make predictions on new data
for matrics in new_data: 
    matrics = np.array(matrics)
    matrics = matrics.reshape(-1,3)
    predictions = svc.predict(matrics)

    #print the predictions
    print(matrics)
    print("Average Score: ", matrics[0][0])
    print("Absence: ", matrics[0][1])
    print("Study Hour: ", matrics[0][2])

    predictions = str(predictions)
    length = len(predictions)
    predictions1 = predictions[2:(length-2)]
    predictions1 = '\t=> ' + predictions1
    print("Follow the following advises")

    predictions1 = predictions1.replace(".", ".\n\t=>")
    predictions1 = ",\n\t=>".join(predictions1.split(","))

    length = len(predictions1)
    if predictions1[length-1] == '>':
        predictions1 = predictions1[:length-2]
        
    print(predictions1)

# evaluate the model performance
confusion_matrix = confusion_matrix(y_test, y_pred)
print("Accuracy: ", accuracy_score(y_test, y_pred))
print("Confusion Matrix: \n", confusion_matrix)

# Displying using



