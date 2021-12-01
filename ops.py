from sklearn.linear_model import LinearRegression
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge

df = pd.read_csv('als.CSV', index_col=0, encoding='CP949')
df.head()

X = df['ops']
y = df['타점']
plt.scatter(X,y)
plt.title('야구')
plt.xlabel('ops')
plt.ylabel('타점')

X_train, X_test, y_train, y_test = train_test_split(X.values.reshape(-1,1), y, train_size=0.8, test_size=0.2)

line_fitter = LinearRegression().fit(X_train, y_train)

plt.plot(X, y, 'o')
plt.plot(X,line_fitter.predict(X.values.reshape(-1,1)))
b = input('원하는 ops를 입력하세요')
a = line_fitter.predict([[b]])
a = float(a)
print('입력한 ops에 대한 타점 예측값 :',round(a,3))
k = int(input('시각화?'))
if k == 1:
  plt.show()

print(line_fitter.score(X_train, y_train))