import pandas as pd
import sklearn

from sklearn.linear_model import LogisticRegression

data = pd.read_csv('book1.csv')
data = data[["average", "absent", "studyhr","category", "advice"]]

x = data[["average", "absent", "studyhr"]]
y = data["category"]

x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size=0.1)

log = LogisticRegression()
log.fit(x_train, y_train)

accuracy = log.score(x_test, y_test)
print(accuracy)

new_data = [[70, 5, 8], [75, 8, 5], [80, 12, 3]]
new_predictions = log.predict(new_data)
print(new_predictions)