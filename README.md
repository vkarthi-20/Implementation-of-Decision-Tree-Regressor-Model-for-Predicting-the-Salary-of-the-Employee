# Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee

## AIM:
To write a program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the standard libraries.
2. Upload the dataset and check for any null values using .isnull() function.
3. Import LabelEncoder and encode the dataset.
4. Import DecisionTreeRegressor from sklearn and apply the model on the dataset.
5. Predict the values of arrays.
6. Import metrics from sklearn and calculate the MSE and R2 of the model on the dataset.
7. Predict the values of array.
8. Apply to new unknown values.

## Program:
```
/*
Program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.
Developed by: Karthi V
RegisterNumber:  212225230130
*/
import pandas as pd

data = pd.read_csv("Salary.csv")

print(data.head())

print(data.info())

print(data.isnull().sum())

from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
data["Position"] = le.fit_transform(data["Position"])

print(data.head())

x = data[["Position", "Level"]]
y = data["Salary"]

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=2
)

from sklearn.tree import DecisionTreeRegressor

dt = DecisionTreeRegressor(random_state=2)

dt.fit(x_train, y_train)

y_pred = dt.predict(x_test)

from sklearn import metrics

mse = metrics.mean_squared_error(y_test, y_pred)
print("Mean Squared Error:", mse)

r2 = metrics.r2_score(y_test, y_pred)
print("R2 Score:", r2)

prediction = dt.predict([[5, 6]])

print("Predicted Salary:", prediction[0])
```

## Output:



<img width="377" height="135" alt="Screenshot 2026-05-17 174819" src="https://github.com/user-attachments/assets/ed79b0c9-8cf5-4886-a4a5-8f4b8c1bbe5c" />




<img width="363" height="212" alt="Screenshot 2026-05-17 174856" src="https://github.com/user-attachments/assets/6745f5cf-7544-4e1f-81c8-d1c2e4dd2b8d" />




<img width="173" height="106" alt="Screenshot 2026-05-17 174906" src="https://github.com/user-attachments/assets/1690de58-c540-43d1-a2f5-58b36d9626c3" />




<img width="391" height="166" alt="Screenshot 2026-05-17 174946" src="https://github.com/user-attachments/assets/56f7afb0-1b4e-4e71-8647-921328581794" />




<img width="342" height="62" alt="Screenshot 2026-05-17 175016" src="https://github.com/user-attachments/assets/238eb1cd-c863-45ac-b237-636b2a6418e7" />


## Result:
Thus the program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee is written and verified using python programming.
