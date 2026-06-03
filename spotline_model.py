import pandas as pd
import numpy as np
import joblib
import os
from sklearn.model_selection import train_test_split, GridSearchCV, TimeSeriesSplit
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score

# 모델 파일 저장 경로 설정
SCALER_PATH = 'scaler.pkl'
MODEL_PATH = 'spotline_model.pkl'
FEATURES_PATH = 'spotline_features.pkl'

def load_and_preprocess_data(file_path):
    """
    데이터를 로드하고 시계열 정렬 후, 피처 엔지니어링을 수행합니다.
    """
    df = pd.read_csv(file_path)
    
    # captured_at 날짜 타입 파싱 및 오름차순 정렬 (미래 데이터 누수 및 shift 오류 방지)
    df['captured_at'] = pd.to_datetime(df['captured_at'])
    df = df.sort_values('captured_at').reset_index(drop=True)
    
    # 요일 코드 (월=0 ~ 일=6) 및 주말 여부 생성 (토/일 = 1, 평일 = 0)
    df['day_of_week'] = df['captured_at'].dt.dayofweek
    df['is_weekend'] = df['day_of_week'].apply(lambda x: 1 if x >= 5 else 0)
    
    # 시계열 지연 변수: 어제 실제 방문자 수 (Lag Feature)
    df['prev_day_count'] = df['total_count'].shift(1)
    
    # 결측치(첫 날 등) 제거
    df_clean = df.dropna().reset_index(drop=True)
    return df_clean

def retrain_and_save_model(file_path='output.csv'):
    """
    [배치 파이프라인] 전체 데이터를 로드하여 전처리 및 학습을 진행하고,
    학습된 스케일러, 피처 목록, 모델을 joblib 형태로 직렬화하여 파일로 저장합니다.
    (매일 밤 스케줄러로 실행되는 것을 가정)
    """
    df_clean = load_and_preprocess_data(file_path)
    
    # 범주형 변수(weather) 원핫 인코딩
    df_encoded = pd.get_dummies(df_clean, columns=['weather'], drop_first=True)
    
    # 예측 시점(오늘 밤)에 미리 알 수 있는 유효 피처 목록 구성
    features = ['temperature', 'is_weekend', 'prev_day_count']
    features += [col for col in df_encoded.columns if col.startswith('weather_')]
    
    X = df_encoded[features]
    y = df_encoded['total_count']
    
    # 데이터셋 시계열 분할 (미래 데이터 누수 방지 위해 shuffle=False 적용)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, shuffle=False
    )
    
    # Data Leakage 방지를 위해 Train 데이터로만 스케일러 fit_transform 진행
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    # Test 데이터는 transform만 수행
    X_test_scaled = scaler.transform(X_test)
    
    # 릿지 회귀 그리드서치 하이퍼파라미터 최적화 범위 정의
    param_grid = {
        'alpha': [0.01, 0.05, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0, 100.0]
    }
    
    # 데이터 크기에 따른 동적 TimeSeriesSplit 설정 (데이터가 극도로 적을 경우 처리)
    n_samples = len(X_train)
    n_splits = 5 if n_samples >= 15 else (2 if n_samples >= 5 else 1)
    
    if n_splits > 1:
        tscv = TimeSeriesSplit(n_splits=n_splits)
        cv_method = tscv
    else:
        # 데이터가 너무 적을 경우 교차검증을 생략하고 기본 모델 학습 (Fallback)
        cv_method = None
        
    if cv_method is not None:
        grid_search = GridSearchCV(
            estimator=Ridge(),
            param_grid=param_grid,
            scoring='neg_mean_absolute_error',
            cv=cv_method
        )
        grid_search.fit(X_train_scaled, y_train)
        best_model = grid_search.best_estimator_
        best_alpha = grid_search.best_params_['alpha']
    else:
        best_model = Ridge(alpha=1.0)
        best_model.fit(X_train_scaled, y_train)
        best_alpha = 1.0
        
    # 최적 모델로 Test 세트 예측 및 평가
    y_pred = best_model.predict(X_test_scaled)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    # 학습이 완료된 모델, 스케일러, 피처 구조 파일로 저장
    joblib.dump(scaler, SCALER_PATH)
    joblib.dump(best_model, MODEL_PATH)
    joblib.dump(features, FEATURES_PATH)
    
    print("=================== 📊 배치 모델 재학습 및 저장 완료 ===================")
    print(f"데이터 파일 경로: {file_path}")
    print(f"총 학습 데이터 수: {len(df_clean)}행")
    print(f"최적의 규제 하이퍼파라미터 (Best alpha): {best_alpha}")
    print(f"Test 세트 평가 - 평균 절대 오차 (MAE): {mae:.2f} 명")
    print(f"Test 세트 평가 - 결정계수 (R² Score): {r2:.4f}")
    print("======================================================================\n")
    
    return mae, r2

def predict_tomorrow_live(tomorrow_temp, tomorrow_weather, today_count):
    """
    [실시간 API 파이프라인] 백엔드에서 호출하는 실시간 예측 함수.
    디스크에 저장된 스케일러와 모델 파일을 로드하여 0.01초 내로 예측 수행.
    """
    if not (os.path.exists(SCALER_PATH) and os.path.exists(MODEL_PATH) and os.path.exists(FEATURES_PATH)):
        raise FileNotFoundError("모델 또는 스케일러 파일이 없습니다. retrain_and_save_model()을 먼저 실행해주세요.")
        
    # 저장된 모델, 스케일러, 피처 구조 로드
    scaler = joblib.load(SCALER_PATH)
    model = joblib.load(MODEL_PATH)
    feature_columns = joblib.load(FEATURES_PATH)
    
    # 내일 날짜 요일 계산
    tomorrow_date = pd.Timestamp.now() + pd.Timedelta(days=1)
    day_of_week = tomorrow_date.dayofweek
    is_weekend = 1 if day_of_week >= 5 else 0
    
    # 입력 데이터 매핑
    input_data = {
        'temperature': tomorrow_temp,
        'is_weekend': is_weekend,
        'prev_day_count': today_count
    }
    
    # 학습 시 구성된 weather 원핫 컬럼값 매칭
    for col in feature_columns:
        if col.startswith('weather_'):
            weather_type = col.replace('weather_', '')
            input_data[col] = 1 if tomorrow_weather.upper() == weather_type.upper() else 0
            
    # 피처 순서 일치 및 변환 처리
    input_df = pd.DataFrame([input_data])
    
    # 누락된 컬럼(학습 시에는 있었으나 현재 weather에는 없는 컬럼) 0으로 채움
    for col in feature_columns:
        if col not in input_df.columns:
            input_df[col] = 0
            
    input_df = input_df[feature_columns]
    
    # Data Leakage 없이, 스케일러의 transform()만 적용
    input_scaled = scaler.transform(input_df)
    
    # 예측 수행 및 자연수 보정
    pred_raw = model.predict(input_scaled)[0]
    final_pred = max(0, int(round(pred_raw)))
    
    kor_days = ["월요일", "화요일", "수요일", "목요일", "금요일", "토요일", "일요일"]
    
    print(f"🔮 [SPOTLINE AI 내일 예측 보고서 (실시간 추론 모드)]")
    print(f"  - 예측 기준 일자: {tomorrow_date.strftime('%Y-%m-%d')} ({kor_days[day_of_week]})")
    print(f"  - 내일 최고 기온: {tomorrow_temp} ℃")
    print(f"  - 내일 기상 상태: {tomorrow_weather}")
    print(f"  - 오늘 최종 방문자수: {today_count} 명")
    print(f"--------------------------------------------------")
    print(f"👉 내일 삼겹살집 예상 방문객 수: 【 {final_pred} 명 】")
    return final_pred

if __name__ == "__main__":
    # 1. 스케줄러가 밤마다 실행하는 모델 갱신 (여기서는 2달치 데이터인 studio_results_...csv 활용 예시)
    print("[1. 배치 학습 파이프라인 가동]")
    dataset_path = 'studio_results_20260603_1957.csv' if os.path.exists('studio_results_20260603_1957.csv') else 'output.csv'
    retrain_and_save_model(dataset_path)
    
    # 2. 백엔드 API 서버가 호출을 받았을 때의 시나리오
    print("[2. 백엔드 API 실시간 예측 요청 처리 시나리오]")
    predict_tomorrow_live(
        tomorrow_temp=21.5,
        tomorrow_weather='RAINY',
        today_count=42
    )
