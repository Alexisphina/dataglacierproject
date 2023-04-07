import pickle

import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import pickle

#import csv data

df = pd.read_csv("iris_flower.csv")

#view data

df.head()

#select the independent and dependent variables
X = df[["Sepal_Length", "Sepal_Width", "Petal_Length", "Petal_Width"]]
y = df["Class"]

#Split data into train and test

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=50)

#Feature scaling
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

#Instantiate model
classifier = RandomForestClassifier()

#fit model
classifier.fit(X_train, y_train)

#create pickle file of model
pickle.dump(classifier, open("model.pkl", "wb"))