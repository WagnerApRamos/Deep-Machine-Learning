# Artificial Neural Network

# Importing the libraries
# import numpy as np
# import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
# Importing Keras libraries and packages
# import keras
from keras.models import Sequential
from keras.layers import Dense

# Part 1 - Data Preprocessing

# Importing the dataset
dataset = pd.read_csv('Churn_Modelling.csv')
X = dataset.iloc[:, 3:13].values
y = dataset.iloc[:, 13].values

# Encoding categorical data
labelencoder_X_1 = LabelEncoder()
X[:, 1] = labelencoder_X_1.fit_transform(X[:, 1])
labelencoder_X_2 = LabelEncoder()
X[:, 2] = labelencoder_X_2.fit_transform(X[:, 2])
onehotencoder = OneHotEncoder(categorical_features=[1])
X = onehotencoder.fit_transform(X).toarray()
X = X[:, 1:]

# Splitting the dataset into the Training set and Test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,
                                                    random_state=0)

# Feature Scaling
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


# Part 2 Now Let's make the ANN

# Initialising the ANN
classifier = Sequential()

# Adding the input layer and the fisrt hidden layer
classifier.add(Dense(units=6,
                     kernel_initializer='uniform',
                     activation='relu', input_dim=11))

# Adding the second hidden layer
classifier.add(Dense(units=6,
                     kernel_initializer='uniform',
                     activation='relu'))

# Adding the output layer
classifier.add(Dense(units=1,
                     kernel_initializer='uniform',
                     activation='sigmoid'))

# Compiling the ANN
classifier.compile(optimizer='adam',
                   loss='binary_crossentropy',
                   metrics=['accuracy'])

# Fitting the ANN to the Training set
classifier.fit(X_train, y_train, batch_size=10, epochs=100)

# Part 3 Making the predictions and evaluating the model

# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)