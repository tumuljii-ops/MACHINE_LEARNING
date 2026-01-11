import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import datasets
import matplotlib.pyplot as plt
from LinearRegression import LinearRegression

#-------Load dataset--------
x,y =datasets.make_regression(
    n_samples=100,
    n_features=1,
    noise =15,
    random_state=42
)

x=x.reshape(-1,1)
#-----------data set before training
plt.scatter(x, y)
plt.xlabel("X")
plt.ylabel("y")
plt.title("Dataset before training")
plt.show()

#---------split of datasets----------

x_train,x_temp,y_train,y_temp=train_test_split(x,y,test_size=0.3,random_state=42)

x_val,x_test,y_val,y_test=train_test_split(x_temp,y_temp,test_size=0.5,random_state=42)





model = LinearRegression(lr=0.01, n_iter=1000)
model.fit(x_train, y_train, x_val, y_val)


y_test_pred = model.prediction(x_test)
test_mse = np.mean((y_test - y_test_pred) ** 2)

print("Test MSE:", test_mse)

#------------r2 score--------------
def r2_score(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return 1 - (ss_res / ss_tot)


r2 = r2_score(y_test, y_test_pred)
print("R2 Score:", r2)


#----------regression line after training--------------
y_line = model.prediction(x)

plt.scatter(x, y, label="Actual Data")
plt.plot(x, y_line, color="red", label="Regression Line")
plt.xlabel("X")
plt.ylabel("y")
plt.title("Linear Regression Result")
plt.legend()
plt.show()

#-------------training vs validation error-----------
plt.plot(model.train_loss, label="Training Loss")
plt.plot(model.val_loss, label="Validation Loss")
plt.xlabel("Epochs")
plt.ylabel("MSE Loss")
plt.title("Training vs Validation Loss")
plt.legend()
plt.show()








