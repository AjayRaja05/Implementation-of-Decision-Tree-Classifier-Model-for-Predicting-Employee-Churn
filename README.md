# Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn

## AIM:
To write a program to implement the Decision Tree Classifier Model for Predicting Employee Churn.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm

1.Import pandas

2.Import Decision tree classifier

3.Fit the data in the model

4.Find the accuracy score


## Program:
```
Program to implement the Decision Tree Classifier Model for Predicting Employee Churn.
Developed by: 
RegisterNumber:  
```
```
import pandas as pd
data=pd.read_csv("Employee.csv")
print("data.head():")
data.head()
print("data.info():")
data.info()
print("isnull() and sum():")
data.isnull().sum()
print("data value counts():")
data["left"].value_counts()
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
print("data.head() for Salary:")
data["salary"]=le.fit_transform(data["salary"])
data.head()
print("data.head() for Salary:")
data["salary"]=le.fit_transform(data["salary"])
data.head()
print("x.head():")
x=data[["satisfaction_level","last_evaluation","number_project","average_montly_hours","time_spend_company","Work_accident","promotion_last_5years","salary"]]
x.head()
y=data["left"]
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=100)
from sklearn.tree import DecisionTreeClassifier
dt=DecisionTreeClassifier(criterion="entropy")
dt.fit(x_train,y_train)
y_pred=dt.predict(x_test)
print("Accuracy value:")
from sklearn import metrics
accuracy=metrics.accuracy_score(y_test,y_pred)
accuracy
print("Data Prediction:")
dt.predict([[0.5,0.8,9,260,6,0,1,2]])
from sklearn.tree import plot_tree
import matplotlib.pyplot as plt
plt.figure(figsize=(8,6))
plot_tree(dt, feature_names=x.columns, class_names=['salary', 'left'], filled=True)
plt.show()
print("data.head() for Salary:")
data["salary"]=le.fit_transform(data["salary"])
data.head()
```

## Output:
<img width="1919" height="257" alt="image" src="https://github.com/user-attachments/assets/c6b32ee9-ad29-4982-af66-459132c494f3" />

<img width="1919" height="372" alt="image" src="https://github.com/user-attachments/assets/14c79fbb-1e02-4e1a-af7b-00a87d433972" />

<img width="1919" height="472" alt="image" src="https://github.com/user-attachments/assets/ba171c21-e532-4705-b42f-b9a028eb58fc" />

<img width="1918" height="218" alt="image" src="https://github.com/user-attachments/assets/490f586f-7f55-4744-b1e2-c1047d53dfdc" />

<img width="1918" height="258" alt="image" src="https://github.com/user-attachments/assets/db71c63e-9966-4fb9-91ca-db11fb67d964" />

<img width="1919" height="255" alt="image" src="https://github.com/user-attachments/assets/e91ff5f0-0bf8-4075-9681-9919b2d741a8" />

<img width="1917" height="263" alt="image" src="https://github.com/user-attachments/assets/490862ac-8987-4562-b989-704f546dd4d3" />

<img width="1919" height="58" alt="image" src="https://github.com/user-attachments/assets/dbe6c486-9026-425b-ba20-901cb4f861e3" />

<img width="1919" height="102" alt="image" src="https://github.com/user-attachments/assets/7cf432b5-ea5c-4ad6-86c1-107c845a3e95" />

<img width="730" height="532" alt="image" src="https://github.com/user-attachments/assets/0b0e8a86-7a26-4d63-bfde-012956b1e7d3" />

## Result:
Thus the program to implement the  Decision Tree Classifier Model for Predicting Employee Churn is written and verified using python programming.
