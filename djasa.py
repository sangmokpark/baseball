import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

df = pd.read_csv("한국에너지공단_자동차 연비표시제도_12_31_2020.CSV", encoding='CP949')
df.head()

x = df[['CO2배출량', '배기량']]
y = df[['연비']]

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, test_size=0.2)

mlr = LinearRegression()
mlr.fit(x_train, y_train)

my_car = [[150, 2200]]
my_predict = mlr.predict(my_car)
print(my_predict)

y_predict = mlr.predict(x_test)

plt.scatter(y_test, y_predict, alpha=0.4)
plt.scatter(x,y)
plt.xlabel("actual fuel efficiency")
plt.ylabel("predicted fuel efficiency")
plt.title("Fuel efficiency")


plt.show()

car = int(input().split())
car_predict = mlr.predict(car)
print(car)

print(mlr.score(x_train, y_train))