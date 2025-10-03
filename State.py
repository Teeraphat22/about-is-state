# นำเข้าไลบรารีที่จำเป็น
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# สร้างข้อมูลตัวอย่าง (X คือ ตัวแปรต้น, y คือ ตัวแปรตาม)
X = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 18]).reshape(-1, 1)
y = np.array([3, 4, 2, 5, 6, 7, 8, 9, 10, 12])

# แบ่งข้อมูลเป็นชุดฝึกและทดสอบ (80% ฝึก, 20% ทดสอบ)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# สร้างโมเดล Linear Regression
model = LinearRegression()

# ฝึกโมเดลด้วยข้อมูลฝึก
model.fit(X_train, y_train)

# ทำนายค่าด้วยข้อมูลทดสอบ
y_pred = model.predict(X_test)

# ประเมินผลโมเดล
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f'Mean Squared Error: {mse:.2f}')
print(f'R^2 Score: {r2:.2f}')

# แสดงกราฟข้อมูลและเส้นถดถอย
plt.scatter(X, y, color='blue', label='Data points')
plt.plot(X, model.predict(X), color='red', label='Regression line')
plt.xlabel('X')
plt.ylabel('y')
plt.legend()
plt.show()
