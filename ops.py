from sklearn.linear_model import LinearRegression
import pandas as pd
from matplotlib import pyplot as plt

df = pd.read_csv('als.CSV', index_col=0, encoding='CP949')
df.head()

X = df['ops']
y = df['타점']
plt.scatter(X,y)
plt.title('야구')
plt.xlabel('ops')
plt.ylabel('타점')

line_fitter = LinearRegression()
line_fitter.fit(X.values.reshape(-1,1), y)

plt.plot(X, y, 'o')

b = input('원하는 ops를 입력하세요')
a = line_fitter.predict([[b]])
a = float(a)
print('입력한 ops에 대한 타점 예측값 :',round(a,3))
k = int(input('시각화?'))
if k == 1:
  plt.show()
