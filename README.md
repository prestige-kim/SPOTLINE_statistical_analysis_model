# 🔮 SPOTLINE: 삼겹살집 내일 방문객 수 예측 통계 모델

본 프로젝트는 비전 AI 데이터 및 1달간의 삼겹살집 방문 이력 목업 데이터(`output.csv`)를 기반으로 **'내일 삼겹살집에 방문할 예상 고객 수'**를 예측하는 통계적 머신러닝 시스템입니다.

> 💡 **주피터 노트북 뷰어 안내**: 만약 GitHub 웹 화면에서 주피터 노트북 파일(`.ipynb`) 렌더링이 지연되거나 소스코드가 제대로 보이지 않는다면, [Jupyter nbviewer 링커](https://nbviewer.org/github/prestige-kim/SPOTLINE_statistical_analysis_model/blob/main/spotline.ipynb)를 통해 쾌적하게 실행 결과 문서를 바로 보실 수 있습니다.
>
> 📁 **순수 파이썬 소스코드 탑재**: 주피터 노트북을 실행하지 않고 코드만 즉각 깃허브에서 검토하시길 원하시는 분들을 위해, 동일한 학습 및 추론 로직을 완벽히 포함한 단일 파이썬 스크립트인 [**`spotline_model.py`**](./spotline_model.py) 파일을 루트 폴더에 추가로 탑재했습니다.

---

## 💡 1. 릿지 회귀(Ridge Regression) 모델 선택 이유

삼겹살집 내일 방문객 수 예측 프로젝트의 머신러닝 알고리즘으로 **릿지 회귀(Ridge Regression)**를 채택한 과학적/비즈니스적 배경은 다음과 같습니다.

### ① 극소형 데이터셋(Small Data)에서의 과적합(Overfitting) 원천 방지
* 현재 수집된 데이터는 **30일치(30개 행)**에 불과합니다.
* 데이터 크기가 이처럼 작을 때 XGBoost, LightGBM, Random Forest 또는 딥러닝(ANN) 같은 복잡한 비선형 모델을 사용하면, 모델이 데이터의 핵심 추세가 아닌 미세한 노이즈와 특이점까지 통째로 암기해 버리는 **과적합(Overfitting)**이 발생하여 실전 예측력이 처참하게 떨어집니다.
* 릿지 회귀는 **L2 가중치 규제(Regularization Penalty)**를 적용하여 모델이 특정 변수에 과도하게 의존하지 않도록 제한하므로, 극소형 데이터 환경에서도 압도적인 일반화 성능을 냅니다.

### ② 최적화 루프 우려가 없는 완벽한 안정성 (Closed-Form Solution)
* 반복적인 경사하강법 학습을 거치는 모델들과 다르게, 릿지 회귀는 수학적으로 단 한 번의 행렬 연산으로 전역 최적값(Global Minimum)을 찾아낼 수 있는 **'닫힌 공식(Closed-form solution)'**이 존재합니다.
* 따라서 최적화 루프에 빠지거나 수렴하지 않는 문제가 원천 차단되며, 언제 실행하더라도 동일하고 안정적인 예측 가중치를 보장합니다.

### ③ 비즈니스 현장에서의 명확한 해석 가능성 (Interpretability)
* 기온, 요일, 날씨 등이 내일 방문 고객 수에 각각 실제 몇 명 정도의 증감 영향을 주는지 **회귀 계수(Coefficients)**를 통해 직관적으로 해석하고 설명할 수 있습니다. 이는 삼겹살집 사장님의 원자재 준비 및 알바생 배치 등 실무 의사결정에 직각적으로 투영됩니다.

---

## 🛠️ 2. 피처 엔지니어링 및 미래 피처 제거 근거 (Data Leakage 방지)

비전 AI API에서 반환하는 JSON 데이터 중 모든 정보를 예측 모델의 입력값으로 쓰면 치명적인 **데이터 누수(Data Leakage)**가 발생합니다.

### 🚫 제거 대상 피처 (미래 수집 정보)
> `avg_dwell_time`, `just_left_count`, `max_empty_table_time`, `max_response_wait_time`, `peak_time`, `people` 정보 등

* **이유**: "내일 방문객 수 예측"은 **오늘 밤 영업이 끝나고 닫는 시점**에 수행합니다.
* 내일 밤이 되어 영업이 다 끝나야만 비전 AI가 최종 집계해서 보내주는 `avg_dwell_time`(체류시간)이나 `just_left_count`(그냥 나간 손님) 같은 피처들은 내일 예측 시점에는 **아직 발생하지 않은 미래 정보**이므로 입력하고 싶어도 물리적으로 존재하지 않습니다.
* 이를 포함시켜 학습용 과거 데이터셋을 돌리면 성능이 100%인 것처럼 보이지만, 실전 배포 시 내일 날짜에 채워줄 데이터가 없어 모델이 즉시 오작동하게 됩니다.

### 🛠️ 가공 및 선택 피처 (예측 시점 가용 정보)
* **`is_weekend` (신규 가공)**: 날짜 정보(`captured_at`)를 기반으로 금/토/일(또는 토/일) 주말 여부를 판별하여 `1`(주말) 또는 `0`(평일) 변환. (평일 ~40명, 주말 ~65명의 삼겹살집 매출 분포 핵심 로직 반영)
* **`prev_day_count` (신규 가공)**: **오늘(내일 예측 시점 기준 어제)** 매장의 최종 정산 방문자 수. 직전 날의 방문 트렌드를 반영하는 핵심 시계열 지연 피처(Lag Feature).
* **`temperature` & `weather` (기존 활용)**: 내일 날짜의 기상 예보를 연동하여 기온과 날씨 정보를 입력.

---

## 📈 3. 주피터 노트북 실행 및 단계별 결과 화면

프로젝트 폴더 내 `spotline.ipynb` 파일을 실행하여 얻은 실제 검증 결과와 출력값입니다.

### 🟢 [Cell 1 ~ 2] 데이터 로드 및 초기 형태
```python
# 1달치 목업 데이터 삽입 스크립트로 생성된 데이터 로드
df = pd.read_csv('output.csv')
print(f"데이터 크기: {df.shape[0]}행 x {df.shape[1]}열")
df.head()
```
* **🖥️ 실행 결과 화면:**
  ```text
  데이터 크기: 30행 x 14열
  ```

### 🟢 [Cell 3 ~ 4] 피처 엔지니어링 및 Lag 데이터 생성
```python
df['captured_at'] = pd.to_datetime(df['captured_at'])
df['day_of_week'] = df['captured_at'].dt.dayofweek
df['is_weekend'] = df['day_of_week'].apply(lambda x: 1 if x >= 5 else 0)
df['prev_day_count'] = df['total_count'].shift(1)
df_clean = df.dropna().reset_index(drop=True)
print(f"피처 가공 후 데이터 개수: {df_clean.shape[0]}개")
```
* **🖥️ 실행 결과 화면:**
  ```text
  피처 가공 후 데이터 개수: 29개
  ```

### 🟢 [Cell 5 ~ 6] 범주형 원핫 인코딩 & 예측 전용 피처 선택
```python
df_encoded = pd.get_dummies(df_clean, columns=['weather'], drop_first=True)
features = ['temperature', 'is_weekend', 'prev_day_count']
features += [col for col in df_encoded.columns if col.startswith('weather_')]
X = df_encoded[features]
y = df_encoded['total_count']
print("최종 선정된 입력 피처 (X):", features)
```
* **🖥️ 실행 결과 화면:**
  ```text
  최종 선정된 입력 피처 (X): ['temperature', 'is_weekend', 'prev_day_count', 'weather_RAINY', 'weather_SUNNY']
  ```

### 🟢 [Cell 7 ~ 8] 데이터 스케일링 및 분할
```python
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
print(f"학습용 샘플 수: {X_train.shape[0]}개 | 검증용 샘플 수: {X_test.shape[0]}개")
```
* **🖥️ 실행 결과 화면:**
  ```text
  학습용 샘플 수: 23개 | 검증용 샘플 수: 6개
  ```

### 🟢 [Cell 9 ~ 10] 릿지 회귀 모델 훈련 및 검증 (핵심 성능 지표)
```python
model = Ridge(alpha=1.0)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
```
* **🖥️ 실행 결과 화면:**
  ```text
  =================== 📊 모델 평가 결과 ===================
  평균 절대 오차 (MAE): 5.87 명
  결정계수 (R² Score): 0.6841
  ==========================================================

  =================== 🔑 피처별 가중치 (Coefficients) ===================
   - temperature: +0.8715
   - is_weekend: +8.3198
   - prev_day_count: -0.7832
   - weather_RAINY: +1.3463
   - weather_SUNNY: -0.1949
  ```

#### 💡 피처 가중치 해석 리포트
1. **`is_weekend` (+8.32)**: 삼겹살집 유입에 가장 기여도가 큽니다. 주말(토, 일)에는 평일보다 평균적으로 **약 8.3명**이 더 방문합니다.
2. **`weather_RAINY` (+1.35)**: 흥미롭게도 비(RAINY)가 오는 날씨 예보는 삼겹살집 방문객을 **약 1.35명** 증가시키는 요인으로 작동합니다. (비가 올 때 삼겹살에 소주를 찾는 한국인의 식문화가 목업 데이터에 적절히 반영되었습니다.)
3. **`temperature` (+0.87)**: 날씨 온도가 1표준편차 상승할수록 방문 고객수가 미세하게 증가합니다.
4. **`prev_day_count` (-0.78)**: 어제 손님이 매우 붐볐다면 오늘 방문자 수는 미세하게 줄어드는 경향성을 띱니다. 삼겹살 외식 소비의 단기 주기성을 보입니다.

---

### 🟢 [Cell 11 ~ 12] 실시간 내일 방문자수 예측 시뮬레이션
* **가상 시나리오**: 내일 예보 온도 `21.5 ℃`, 내일 날씨 예보 `RAINY` (비), 오늘 매장 최종 마감 정산 고객 수 `42 명`일 때 내일은 몇 명이 방문할 것인가?
* **🖥️ 실행 결과 화면:**
  ```text
  [시뮬레이션 1. 비 내리는 평일]
  🔮 [SPOTLINE AI 내일 예측 보고서]
    - 예측 기준 일자: 2026-06-01 (월요일)
    - 내일 최고 기온: 21.5 ℃
    - 내일 기상 상태: RAINY
    - 오늘 최종 방문자수: 42 명
  --------------------------------------------------
  👉 내일 삼겹살집 예상 방문객 수: 【 46 명 】
  ```

---

## 🚀 4. 로컬 구동 가이드

1. **가상환경 활성화 및 패키지 설치**
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # macOS/Linux 기준
   pip install pandas numpy scikit-learn jupyter
   ```
2. **파이썬 스크립트 실행 (즉각 예측)**
   ```bash
   python spotline_model.py
   ```
3. **주피터 노트북 실행 및 확인**
   ```bash
   jupyter notebook spotline.ipynb
   ```
