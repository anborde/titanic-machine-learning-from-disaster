import pandas as pd
import numpy as np
import csv
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

# Reading data from csv file
data_df = pd.read_csv('train.csv', usecols=['PassengerId', 'Pclass','Sex','Survived'], index_col='PassengerId', na_values=['NaN'])

# Removing null data
data_df = data_df.dropna()

# Converting Sex column to integer where Male - 0, Female - 1
data_df['Sex'] = data_df['Sex'].replace('male', 0)
data_df['Sex'] = data_df['Sex'].replace('female', 1)


# Printing sample data
print(data_df.tail())

# Dividing Data Into Features & Labels
features = data_df.loc[:, 'Pclass':'Sex'].values
labels = data_df.loc[:, 'Survived'].values


# Splitting data into training and testing data
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.10, shuffle=True)

# Implementing KNN
model =SVC(kernel='rbf', C=1000)

model.fit(X_train, y_train)

# Printing Accuracy
print('Accuracy:', model.score(X_test, y_test))


test_df = pd.read_csv('test.csv', usecols=['PassengerId', 'Pclass', 'Sex', 'Age'], index_col='PassengerId', na_values=['NaN'])

# Converting Sex column to integer where Male - 0, Female - 1
test_df['Sex'] = test_df['Sex'].replace('male', 0)
test_df['Sex'] = test_df['Sex'].replace('female', 1)

test_features = test_df.loc[:, 'Pclass':'Sex'].values
test_pid = test_df.index.values

result = model.predict(test_features)

data = np.transpose(np.vstack((test_pid, result)))

# Writing Results to csv file
with open('result.csv', 'w', newline='') as f:
    writer = csv.writer(f)

    writer.writerows([['PassengerId', 'Survived']])
    writer.writerows(data)

