# 匯入所需的套件
import numpy as np                      # 用於數值與陣列處理
import pandas as pd                    # 用於資料讀取與處理
import matplotlib.pyplot as plt        # 用於資料視覺化
from sklearn.preprocessing import PolynomialFeatures # 建立多項式特徵
from sklearn.linear_model import LinearRegression     # 線性回歸模型
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split  # 資料集分割

# 讀取銷售資料集（CSV 檔）
yearly_sales_data = pd.read_csv(R'D:\USER\Downloads\w2\salesdata.csv')
"""
your code
"""
# 分割資料集為訓練集和測試集
x = yearly_sales_data[['DayOfWeek']]  # 使用星期幾作為特徵
y = yearly_sales_data['Sales']          # 銷售金額作為目標
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
# 創建多項式特徵生成器
degree = 6
model = make_pipeline(PolynomialFeatures(degree),LinearRegression())
poly = make_pipeline(PolynomialFeatures(degree))
# 擬合模型
model.fit(X_train, y_train)
poly.fit(X_train, y_train)

# 預測測試集
y_pred = model.predict(X_test)

# 預測一週（星期一到星期日）的銷售金額
# 計算每星期幾的實際平均銷售金額
# X_week = np.arange(1, 8).reshape(-1, 1)        # 建立 1~7（代表星期一到日）的一維列向量
X_week = pd.DataFrame({'DayOfWeek': np.arange(1, 8)})
X_week_poly = poly.transform(X_week)          # 套用相同的多項式轉換器
y_week_pred = model.predict(X_week)      # 使用訓練好的模型進行預測

# 計算每星期幾的實際平均銷售金額
weekly_sales_avg = yearly_sales_data.groupby('DayOfWeek')['Sales'].mean().reset_index()

# 繪圖：實際平均 vs 預測銷售金額（按星期幾）
plt.figure(figsize=(10, 6))
plt.plot(weekly_sales_avg['DayOfWeek'], weekly_sales_avg['Sales'], 'bo-', label='Average Actual Sales')  # 藍點線：實際平均
plt.plot(X_week, y_week_pred, 'ro-', label='Predicted Sales')# 紅點線：預測結果

# 圖表標題
plt.title('Average Actual vs Predicted Sales by Day of the Week')  

plt.xlabel('Day of the Week')                                       # X 軸：星期幾
plt.xticks(np.arange(1, 8), ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'])  # 對應的星期文字
plt.ylabel('Sales')                                                 # Y 軸：銷售金額

plt.legend()                                                        # 顯示圖例
plt.show()                                                          # 顯示圖形
