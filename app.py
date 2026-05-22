import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier

# 1. 페이지 설정
st.set_page_config(page_title="설비 고장 예측 AI 서비스", layout="centered")
st.title("🛠️ 설비 실시간 고장 진단 시스템")
st.write("센서 데이터를 바탕으로 설비의 정상/이상 여부를 실시간으로 예측합니다.")

# 2. [가상 데이터 & 데모 학습] 외부 보안 우려를 없애기 위해 앱 구동 시 실시간 학습
@st.cache_resource
def train_demo_model():
    # 정상 데이터 (온도 40~60도, 진동 10~30)
    normal_temp = np.random.uniform(40, 60, 100)
    normal_vib = np.random.uniform(10, 30, 100)
    normal_labels = np.zeros(100)
    
    # 고장 데이터 (온도 70~90도, 진동 40~60)
    fault_temp = np.random.uniform(70, 90, 100)
    fault_vib = np.random.uniform(40, 60, 100)
    fault_labels = np.ones(100)
    
    X = pd.DataFrame({
        'Temperature': np.concatenate([normal_temp, fault_temp]),
        'Vibration': np.concatenate([normal_vib, fault_vib])
    })
    y = np.concatenate([normal_labels, fault_labels])
    
    model = RandomForestClassifier(random_state=42)
    model.fit(X, y)
    return model

model = train_demo_model()

# 3. 사용자 입력 UI (사이드바 또는 메인 화면)
st.subheader("📊 실시간 센서 데이터 입력")
col1, col2 = st.columns(2)

with col1:
    input_temp = st.slider("현재 설비 온도 (°C)", 30.0, 100.0, 50.0, step=0.5)
with col2:
    input_vib = st.slider("현재 설비 진동 수치 (Hz)", 5.0, 70.0, 20.0, step=0.5)

# 4. AI 모델 예측 실행
input_data = pd.DataFrame([[input_temp, input_vib]], columns=['Temperature', 'Vibration'])
prediction = model.predict(input_data)[0]
proba = model.predict_proba(input_data)[0]

# 5. 결과 시각화 및 출력
st.markdown("---")
st.subheader("🔮 AI 진단 결과")

if prediction == 0:
    st.success(f"✅ **정상 가동 중** (정상 확률: {proba[0]*100:.1f}%)")
else:
    st.error(f"🚨 **이상 징후 감지 (고장 위험)** (고장 확률: {proba[1]*100:.1f}%)")
    st.warning("조치 제안: 설비 냉각 시스템 점검 및 베어링 진동 상태 확인이 필요합니다.")
