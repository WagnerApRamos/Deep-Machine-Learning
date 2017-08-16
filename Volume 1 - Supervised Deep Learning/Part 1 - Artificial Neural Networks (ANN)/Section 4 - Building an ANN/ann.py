# Artificial Neural Network

# Importing the libraries
# import numpy as np
# import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
# Importing Keras libraries and packages
# import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout

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

# Adding the input layer and the fisrt hidden layer with Dropout
classifier.add(Dense(units=6,
                     kernel_initializer='uniform',
                     activation='relu', input_dim=11))
classifier.add(Dropout(rate=0.1))

# Adding the second hidden layer
classifier.add(Dense(units=6,
                     kernel_initializer='uniform',
                     activation='relu'))
classifier.add(Dropout(rate=0.1))

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
y_pred = y_pred > 0.5


# Predicting a single observation

"""predicting if the customer with the following informations will leave the bank:
Geography: France
Credit Scor: 600
Gender: Male
Age: 40
Tenure: 3
Balance: 60000
Number of products: 2
Has Credit Card: Yes
Is active member: Yes
Estimated Salary: 50000
"""

new_prediction = classifier.predict(sc.transform(np.array([[0, 0, 600, 1, 40, 3, 60000, 2, 1, 1, 50000]])))
new_prediction = (new_prediction > 0.5)


# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

# Part 4 - Evaluating, Improving and Tunning the ANN

# Evaluating the ANN
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score
from keras.models import Sequential
from keras.layers import Dense
def build_classifier():
    classifier = Sequential()
    classifier.add(Dense(units=6,
                         kernel_initializer='uniform',
                         activation='relu', input_dim=11))
    classifier.add(Dense(units=6,
                         kernel_initializer='uniform',
                         activation='relu'))
    classifier.add(Dense(units=1,
                         kernel_initializer='uniform',
                         activation='sigmoid'))
    classifier.compile(optimizer='adam',
                       loss='binary_crossentropy',
                       metrics=['accuracy'])
    
    return classifier
    
classifier = KerasClassifier(build_fn=build_classifier,
                             batch_size=10,
                             nb_epoch=100)
accuracies = cross_val_score(estimator=classifier,
                             X=X_train,
                             y=y_train,cv=10,
                             n_jobs=-1)
mean = accuracies.mean()
variance = accuracies.std()

# Improving the ANN


# Tunning the ANN
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV
from keras.models import Sequential
from keras.layers import Dense
def build_classifier(optimizer):
    classifier = Sequential()
    classifier.add(Dense(units=6,
                         kernel_initializer='uniform',
                         activation='relu', input_dim=11))
    classifier.add(Dense(units=6,
                         kernel_initializer='uniform',
                         activation='relu'))
    classifier.add(Dense(units=1,
                         kernel_initializer='uniform',
                         activation='sigmoid'))
    classifier.compile(optimizer=optimizer,
                       loss='binary_crossentropy',
                       metrics=['accuracy'])
    
    return classifier
    
classifier = KerasClassifier(build_fn=build_classifier)
parameters = {'batch_size': [25, 32], 
              'nb_epoch': [100, 500],
              'optimizer': ['adam', 'rmsprop', '']}
grid_search = GridSearchCV(estimator=classifier,
                           param_grid=parameters,
                           scoring='accuracy',
                           cv=10)
grid_search.fit(X=X_train, y=y_train)
best_parameters = grid_search.best_params_
besr_accuracy = grid_search.best_score_
