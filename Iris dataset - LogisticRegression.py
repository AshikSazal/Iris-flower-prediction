import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

dataset=load_iris()

print(dataset.DESCR)


x=dataset.data
y=dataset.target

plt.plot(x[:,0][y==0],x[:,1][y==0],'r.',label='Setosa')
plt.plot(x[:,0][y==1],x[:,1][y==1],'g.',label='Versicolour')
plt.plot(x[:,0][y==2],x[:,1][y==2],'b.',label='Virginica')
plt.legend()
plt.show()

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=.3)

reg=LogisticRegression()
reg.fit(x_train,y_train)

y_pred=reg.predict(x_test)
print(y_pred)

from sklearn.metrics import mean_squared_error
error=mean_squared_error(y_pred,y_test)
acc=reg.score(x_test,y_test)

print("Coefficient : ",reg.coef_)
print("Intercept : ",reg.intercept_)
print("Mean squared error : ",error)
print("Accuracy : ",acc)


from sklearn.metrics import accuracy_score
print(accuracy_score(y_test,y_pred))

