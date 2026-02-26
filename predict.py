import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# 1. 가상 농장 데이터 생성
np.random.seed(42)
n = 1000

# 정상 데이터 (700개)
normal = pd.DataFrame({
    'temperature': np.random.uniform(18, 26, 700),
    'humidity':    np.random.uniform(55, 80, 700),
    'co2':         np.random.uniform(600, 1100, 700),
    'label': 0  # 0 = 정상
})

# 이상 데이터 (300개) — 과습, 고온, CO2 초과
abnormal = pd.DataFrame({
    'temperature': np.random.uniform(28, 38, 300),
    'humidity':    np.random.uniform(83, 98, 300),
    'co2':         np.random.uniform(1150, 1800, 300),
    'label': 1  # 1 = 이상
})

df = pd.concat([normal, abnormal]).sample(frac=1).reset_index(drop=True)

# 2. 학습
X = df[['temperature', 'humidity', 'co2']]
y = df['label']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

print("✅ 모델 학습 완료")
print("\n📊 정확도 리포트")
print(classification_report(y_test, model.predict(X_test),
      target_names=["정상", "이상"]))

# 3. 예측 함수
def check_farm(temperature, humidity, co2):
    input_data = pd.DataFrame([{
        'temperature': temperature,
        'humidity': humidity,
        'co2': co2
    }])
    pred = model.predict(input_data)[0]
    prob = model.predict_proba(input_data)[0][1]

    status = "⚠️  이상 감지" if pred == 1 else "✅ 정상"
    print(f"\n입력값: 온도 {temperature}도 | 습도 {humidity}% | CO2 {co2}ppm")
    print(f"결과: {status}")
    print(f"위험도: {prob*100:.1f}%")

    if pred == 1:
        if humidity > 83:
            print("→ 과습 위험. 즉시 환기 권장")
        if temperature > 28:
            print("→ 고온 감지. 냉방 또는 차광 필요")
        if co2 > 1150:
            print("→ CO2 초과. 환기 가동 필요")

# 4. 테스트
print("\n" + "="*50)
print("테스트 케이스")
print("="*50)
check_farm(20, 65, 900)    # 정상
check_farm(32, 90, 1400)   # 이상
check_farm(24, 85, 1050)   # 경계
