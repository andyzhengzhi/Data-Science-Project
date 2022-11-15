# This is a sample Python script.

# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.


def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press ⌘F8 to toggle the breakpoint.


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print_hi('PyCharm')

# See PyCharm help at https://www.jetbrains.com/help/pycharm/

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

train = pd.read_csv("/Users/andyliang/Desktop/ U of T/MMF 1922 Data Science/train.csv")
test = pd.read_csv("/Users/andyliang/Desktop/ U of T/MMF 1922 Data Science/test.csv")

# Create arrary of categorial variables to be encoded
categorical_cols = ['cut', 'color', 'clarity']
le = LabelEncoder()
# apply label encoder on categorical feature columns
train[categorical_cols] = train[categorical_cols].apply(lambda col: le.fit_transform(col))

train_X = train.drop(['price'],axis=1)
train_Y = train['price']


# Check if there's missing data
#print(train.isnull().sum())


rdf = RandomForestRegressor(n_estimators=50,oob_score=True)
rdf.fit(train_X, train_Y)

predicted_Y = rdf.predict(train_X)
test[categorical_cols] = test[categorical_cols].apply(lambda col: le.fit_transform(col))
test_predicted_Y = rdf.predict(test)
print(mean_squared_error(train_Y, predicted_Y, squared = False))
test['price'] = test_predicted_Y
df = test[["id","price"]].set_index("id")
df.to_csv("submission.csv")

