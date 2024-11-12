# Implementation-of-SVM-For-Spam-Mail-Detection

## AIM:
To write a program to implement the SVM For Spam Mail Detection.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Import the necessary python packages using import statements.

2.Read the given csv file using read_csv() method and print the number of contents to be displayed using df.head().

3.Split the dataset using train_test_split.

4.Calculate Y_Pred and accuracy.

5.Print all the outputs.

6.End the Program.

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
![image](https://github.com/user-attachments/assets/c307789f-fe6e-4b24-a4df-bf12fff4027b)
![image](https://github.com/user-attachments/assets/2cc376c7-b016-44c7-98db-cebe81f48fcd)
![image](https://github.com/user-attachments/assets/71446883-2de5-4131-b987-0ee967b8df2e)
![image](https://github.com/user-attachments/assets/1d2c715f-28c9-472e-a1ee-b09c1bbe965c)
![image](https://github.com/user-attachments/assets/ef830362-e92c-488b-8153-eb39c9921be3)



## Result:
Thus the program to implement the SVM For Spam Mail Detection is written and verified using python programming.
