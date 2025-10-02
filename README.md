import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

#loading the csv data to a pandas data frame
heart_data=pd.read_csv('/content/heart_disease_data.csv')

#print first 5 rows of the dataset
heart_data.head()

#print last 5 rows of the dataset
heart_data.tail()

from matplotlib import pyplot as plt
heart_data['age'].plot(kind='hist', bins=20, title='age')
plt.gca().spines[['top', 'right',]].set_visible(False)

#no of rows and columns in the data set
heart_data.shape

#getting some info about the data
heart_data.info()

#checking for missing values
heart_data.isnull().sum()


#statistical measures about the data
heart_data.describe()

#checking the distribution of target variable
heart_data['target'].value_counts()

X=heart_data.drop(columns='target',axis=1)
Y=heart_data['target']

x_train,x_test,y_train,y_test=train_test_split(X,Y,test_size=0.2,stratify=Y,random_state=2)

print(X.shape,x_train.shape,x_test.shape)

#training the Logistic regression model with Training data
model.fit(x_train,y_train)

#accuracy on training data
x_train_prediction=model.predict(x_train)
training_data_accuracy=accuracy_score(x_train_prediction,y_train)

print('Accuarcy on training data:',training_data_accuracy)

#accuracy on test data
x_test_prediction=model.predict(x_test)
test_data_accuracy=accuracy_score(x_test_prediction,y_test)

print('Accuarcy on Test data:',test_data_accuracy)

input_data=(62,1,0,120,267,0,1,99,1,1.8,1,2,3)
#change the input data to a numpy array
input_data_as_numpy_array=np.asarray(input_data)

#reshape the numpy array as we are predicting for only on instance
input_data_reshaped=input_data_as_numpy_array.reshape(1,-1)

prediction=model.predict(input_data_reshaped)
print(prediction)

if(prediction[0]==0):
  print('The person does not have a heart disease')
else:
  print('The person has heart disease')

  
