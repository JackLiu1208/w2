# w2_homework.py 3.10.11
# linear and non-linear regression homework

# import necessary libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

# load the dataset
data = pd.read_csv(r"C:\Users\Administrator\code\w2\dataset.csv")

# 分割資料
runtime = data[['Runtime']]
fault = data['faults']

# 分割資料集為訓練集和測試集
X_train, X_test, y_train, y_test = train_test_split(runtime, fault, test_size=0.5, random_state=42)

## 線性回歸 ##
# 建立線性回歸模型
model = LinearRegression()

# 擬合模型
model.fit(X_train, y_train)

# 預測測試集
y_pred = model.predict(X_test)

# 計算方均跟誤差與R2
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# 繪製圖輸出
# 繪製觀測值
plt.scatter(X_train, y_train, color='black', label='actual data')

# 繪製預測值
plt.plot(X_test,y_pred, color='blue', linewidth=3, label='prediction faults')

# 標題與標籤
plt.title('linear regression')
plt.xlabel('runtime')
plt.ylabel('fault')

# 顯示圖例
plt.legend()    

# 顯示圖形
plt.show()

## 線性回歸 ##

## 非線性回歸 ##

# 創造多項式特徵生成器
degree = 6
polyreg = make_pipeline(PolynomialFeatures(degree), LinearRegression())

# 擬合模型
polyreg.fit(X_train, y_train)

# 預測測試集
y_pred_poly = polyreg.predict(X_test)

# 繪製圖輸出
# 繪製觀測值
plt.scatter(X_train, y_train, color='black', label='actaul data')

# 繪製預測線
runtime_fit = np.linspace(runtime.min(), runtime.max(), 100)
faults_fit = polyreg.predict(runtime_fit)
plt.plot(runtime_fit, faults_fit, linewidth=3, label='prediction faults')

# 標題與標籤
plt.title('non-linear regression')
plt.xlabel('runtime')
plt.ylabel('fault')
plt.legend()
plt.show()