# Implementation-of-SVM-For-Spam-Mail-Detection

## AIM:
To write a program to implement the SVM For Spam Mail Detection.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. 
2. 
3. 
4. 

## Program:
```
/*
Program to implement the SVM For Spam Mail Detection..
Developed by: Lakshmi Priya .V
RegisterNumber:  212223220049
*/
```
```
import pandas as pd
data= pd.read_csv("C:/Users/admin/Desktop/INTR MACH/spam.csv", encoding= 'Windows-1252')
data.head()
data.info()
data.isnull().sum()
x=data["v1"].values
y=data["v2"].values

from sklearn.model_selection import train_test_split
x_train,x_test , y_train, y_test = train_test_split(x,y, test_size=0.2, random_state=0)

from sklearn.feature_extraction.text import CountVectorizer
cv=CountVectorizer()

x_train = cv.fit_transform(x_train)
x_test= cv.transform(x_test)

from sklearn.svm import SVC
svc=SVC()
svc.fit(x_train , y_train)
y_pred=svc.predict(x_test)
y_pred

from sklearn import metrics
accuracy= metrics.accuracy_score(y_test, y_pred)
accuracy

from sklearn.metrics import confusion_matrix, classification_report
con= confusion_matrix(y_test, y_pred)
print(con)cl=classification_report(y_test,y_pred)
print(cl)
```

## Output:
![image](https://github.com/user-attachments/assets/cc3aa22a-c6bc-4d74-bfde-817102ff59d3)
![image](https://github.com/user-attachments/assets/9d65f33f-21b0-4468-9390-305cea71d5eb)
![image](https://github.com/user-attachments/assets/31ee6ee5-364b-4d0b-8dad-2f02a81284db)
![image](https://github.com/user-attachments/assets/7dddc9af-65e4-42e1-a569-a1fa5374fcf9)


## Result:
Thus the program to implement the SVM For Spam Mail Detection is written and verified using python programming.
