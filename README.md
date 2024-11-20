# Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn

## AIM:
To write a program to implement the Decision Tree Classifier Model for Predicting Employee Churn.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the standard libraries.
2. Upload the dataset and check for any null values using .isnull() function.
3. Import LabelEncoder and encode the dataset.
4. Import DecisionTreeClassifier from sklearn and apply the model on the dataset.
5. Predict the values of array.
6. Import metrics from sklearn and calculate the accuracy of    the model on the dataset. 
7. Predict the values of array. 8.Apply to new unknown values.

## Program:
```python
/*
Program to implement the Decision Tree Classifier Model for Predicting Employee Churn.
Developed by: Sriram K
RegisterNumber: 212222080052
*/

import pandas as pd
data=pd.read_csv("Employee.csv")

data.head()

data.info()

data.isnull().sum()

data["left"].value_counts()

from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()

data["salary"]=le.fit_transform(data["salary"])
data.head()

x=data[["satisfaction_level","last_evaluation","number_project","average_montly_hours","time_spend_company","Work_accident","promotion_last_5years","salary"]]
x.head()

y=data["left"]

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=100)

from sklearn.tree import DecisionTreeClassifier
dt=DecisionTreeClassifier(criterion="entropy")
dt.fit(x_train,y_train)
y_pred=dt.predict(x_test)

from sklearn import metrics   
accuracy=metrics.accuracy_score(y_test,y_pred)
accuracy

dt.predict([[0.5,0.8,9,260,6,0,1,2]])
```

## Output:
### DATA HEAD
![DATA HEAD](https://github.com/user-attachments/assets/66512439-b986-4a1d-b2b1-713b3bc3e283)

### DATA INFO
![DATA INFO](https://github.com/user-attachments/assets/5bdc85e9-b497-4fb9-bc6a-cf40cdbad7cb)

### DATA ISNULL
![DATA ISNULL](https://github.com/user-attachments/assets/c98d994b-aea4-441d-8802-fc9ca68a22cc)

### DATA LEFT
![DATA LEFT](https://github.com/user-attachments/assets/f0483f12-b595-47f8-9ee0-caed39c4ddb1)

### X HEAD
![X HEAD](https://github.com/user-attachments/assets/5bcacf1a-b230-4540-ac9e-7f1f1decc665)

### DATA FIT
![DATA FIT](https://github.com/user-attachments/assets/a11c9ea3-b6c2-4958-a156-2b01a464f126)

### ACCURACY
![accuracy](https://github.com/user-attachments/assets/b1a6c8e9-77fb-46c6-b960-b8b5c2e09474)

### PREDICTED VALUES
![predicted](https://github.com/user-attachments/assets/932d85cb-ce90-44b4-b0e1-2b00d82e90b8)

## Result:
Thus the program to implement the  Decision Tree Classifier Model for Predicting Employee Churn is written and verified using python programming.
