# 1. 라이브러리 불러오기
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# 2. 예시용 가짜 데이터 생성
# 서로 다른 값이 포함되도록 만듬
data = {
    'nhr': [0.02, 0.03, 0.022, 0.035, 0.021, 0.024, 0.026, 0.025, 0.027, 0.023],
    'rpde': [0.41, 0.39, 0.42, 0.45, 0.44, 0.43, 0.46, 0.40, 0.47, 0.38],
    'parkinson_status': [1, 0, 1, 1, 0, 0, 1, 0, 1, 0]
}
df = pd.DataFrame(data)

# 3. 데이터 시각화
sns.scatterplot(data=df, x='nhr', y='rpde', hue='parkinson_status', palette='coolwarm')
plt.title("nhr vs rpde by Parkinson Status")
plt.show()

# 4. 간단한 모델 학습
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

X = df[['nhr', 'rpde']]
y = df['parkinson_status']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LogisticRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("정확도:", accuracy_score(y_test, y_pred))
