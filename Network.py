from NetworkCore import Linear,Sequential
from helper import lin,MSE,RMSE,MAE
import numpy as np
import pandas as pd

df = pd.read_csv("Student_Performance.csv")

df["Extracurricular Activities"]=(df["Extracurricular Activities"]=="Yes").astype(int)

Train = df[:7000]
Test = df[7000:]

x_train = np.array(Train.drop(["Performance Index"],axis=1))
y_train = np.array(Train["Performance Index"]).reshape(-1,1)

x_test = np.array(Test.drop(["Performance Index"],axis=1))
y_test = np.array(Test["Performance Index"]).reshape(-1,1)

model = Sequential(
    Linear(5,3),
    Linear(3,1,lin),
    loss=MAE
)
model.summary()

model.fit(x_train,y_train,200,100)

predictions = model.predict(x_test)

print("Test Loss-> MSE",MSE(y_test,predictions),"RMSE: ",RMSE(y_test,predictions), "MAE: ",MAE(y_test,predictions))