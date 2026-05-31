import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score

def load_and_preprocess_data(file_path):
    """
    1달치 목업 데이터를 읽어오고 요일/주말 판별 피처 및 
    시계열 지연 피처(yesterday_count)를 생성합니다.
    """
    df = pd.read_csv(file_path)
    
    # captured_at 날짜 타입 파싱
    df['captured_at'] = pd.to_datetime(df['captured_at'])
    
    # 요일 코드 (월=0 ~ 일=6) 및 주말 여부 생성 (토/일 = 1, 평일 = 0)
    df['day_of_week'] = df['captured_at'].dt.dayofweek
    df['is_weekend'] = df['day_of_week'].apply(lambda x: 1 if x >= 5 else 0)
    
    # 시계열 지연 변수: 어제 실제 방문자 수 (Lag Feature)
    df['prev_day_count'] = df['total_count'].shift(1)
    
    # 결측치(첫 날) 제거
    df_clean = df.dropna().reset_index(drop=True)
    return df_clean

def train_ridge_model(df_clean):
    """
    미래 시점(Data Leakage) 피처를 제거하고, 원핫 인코딩 및 스케일링을 거쳐
    안정적인 L2 규제 기반 릿지 회귀 모델을 학습하고 검증합니다.
    """
    # 범주형 변수(weather) 원핫 인코딩
    df_encoded = pd.get_dummies(df_clean, columns=['weather'], drop_first=True)
    
    # 예측 시점(오늘 밤)에 미리 알 수 있는 유효 피처 목록 구성
    features = ['temperature', 'is_weekend', 'prev_day_count']
    features += [col for col in df_encoded.columns if col.startswith('weather_')]
    
    X = df_encoded[features]
    y = df_encoded['total_count']
    
    # Ridge의 L2 가중치 밸런스를 위한 피처 스케일링
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # 데이터셋 분할 (80% 학습, 20% 검증)
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42
    )
    
    # 릿지 모델 선언 및 학습
    model = Ridge(alpha=1.0)
    model.fit(X_train, y_train)
    
    # 검증 평가
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    print("=================== 📊 모델 학습 및 평가 완료 ===================")
    print(f"평균 절대 오차 (MAE): {mae:.2f} 명")
    print(f"결정계수 (R² Score): {r2:.4f}")
    print("=================================================================\n")
    
    print("=================== 🔑 피처별 가중치 (Coefficients) ===================")
    for feat, coef in zip(features, model.coef_):
        print(f" - {feat}: {coef:+.4f}")
    print("=================================================================\n")
    
    return model, scaler, features

def predict_tomorrow(tomorrow_temp, tomorrow_weather, today_count, model, scaler, feature_columns):
    """
    내일 기온 예보, 날씨 예보, 그리고 오늘의 매장 최종 마감 정산 인원수를 통해
    내일의 총 방문객 수를 산출하는 실시간 추론기입니다.
    """
    # 내일 날짜 요일 계산
    tomorrow_date = pd.Timestamp.now() + pd.Timedelta(days=1)
    day_of_week = tomorrow_date.dayofweek
    is_weekend = 1 if day_of_week >= 5 else 0
    
    # 데이터 매핑
    input_data = {
        'temperature': tomorrow_temp,
        'is_weekend': is_weekend,
        'prev_day_count': today_count
    }
    
    # weather 원핫 컬럼값 매칭
    for col in feature_columns:
        if col.startswith('weather_'):
            weather_type = col.replace('weather_', '')
            input_data[col] = 1 if tomorrow_weather.upper() == weather_type.upper() else 0
            
    # 피처 순서 일치 및 스케일링
    input_df = pd.DataFrame([input_data])[feature_columns]
    input_scaled = scaler.transform(input_df)
    
    # 예측 수행 및 자연수 보정
    pred_raw = model.predict(input_scaled)[0]
    final_pred = max(0, int(round(pred_raw)))
    
    kor_days = ["월요일", "화요일", "수요일", "목요일", "금요일", "토요일", "일요일"]
    
    print(f"🔮 [SPOTLINE AI 내일 예측 보고서]")
    print(f"  - 예측 기준 일자: {tomorrow_date.strftime('%Y-%m-%d')} ({kor_days[day_of_week]})")
    print(f"  - 내일 최고 기온: {tomorrow_temp} ℃")
    print(f"  - 내일 기상 상태: {tomorrow_weather}")
    print(f"  - 오늘 최종 방문자수: {today_count} 명")
    print(f"--------------------------------------------------")
    print(f"👉 내일 삼겹살집 예상 방문객 수: 【 {final_pred} 명 】")
    return final_pred

if __name__ == "__main__":
    # 실행 시 모델 훈련 및 가상 시뮬레이션 즉각 가동
    df_clean = load_and_preprocess_data('output.csv')
    model, scaler, features = train_ridge_model(df_clean)
    
    # 비 내리는 평일 시나리오 테스트
    print("[가상 시뮬레이션 실행]")
    predict_tomorrow(
        tomorrow_temp=21.5,
        tomorrow_weather='RAINY',
        today_count=42,
        model=model,
        scaler=scaler,
        feature_columns=features
    )
