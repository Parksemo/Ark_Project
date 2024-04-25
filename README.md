# Ark_Project

## 스마트 공장의 제어 시스템 구축을 위한 제품 품질 분류 AI 모델 개발

[DACON 링크](https://dacon.io/competitions/official/236055/overview/description)
<br><br>
![alt text](image.png)
<br>

### 배경
- 펜데믹을 맞이한 최근 몇 년간 제조 운영과 공급망의 디지털 혁신은 관련 산업 분야에서 최우선 과제로 급부상했습니다.
- 스마트 공장은 공정 데이터에서 인사이트를 발굴하고 해석하여 추세를 예측하고, 스마트 제조 워크플로와 자동화된 프로세스를 구현합니다.
- LG에서는 제조 지능화를 통해 공정 과정에서 발생하는 이벤트에 신속하게 대응하고, 안정성과 효율을 극대화하기 위한 방안을 지속적으로 도모하고 있습니다.
- 품질 편차를 최소화해 생산 경제성과 안정성을 확보할 수 있도록 생산된 제품이 적정 기준을 충족하는지 판단하고 분류하는 AI 모델을 개발해 주세요.

<br>

### 데이터 정보

- train.csv [파일]
    - PRODUCT_ID : 제품의 고유 ID
    - Y_Class : 제품 품질 상태(Target) 
        - 0 : 적정 기준 미달 (부적합)
        - 1 : 적합
        - 2 : 적정 기준 초과 (부적합)
    - Y_Quality : 제품 품질 관련 정량적 수치
    - TIMESTAMP : 제품이 공정에 들어간 시각
    - LINE : 제품이 들어간 공정 LINE 종류 ('T050304', 'T050307', 'T100304', 'T100306', 'T010306', 'T010305' 존재)
    - PRODUCT_CODE : 제품의 CODE 번호 ('A_31', 'T_31', 'O_31' 존재)
    - X_1 ~ X_2875 : 공정 과정에서 추출되어 비식별화된 변수

- test.csv [파일]
    - PRODUCT_ID : 제품의 고유 ID
    - TIMESTAMP : 제품이 공정에 들어간 시각
    - LINE : 제품이 들어간 공정 LINE 종류 ('T050304', 'T050307', 'T100304', 'T100306', 'T010306', 'T010305' 존재)
    - PRODUCT_CODE : 제품의 CODE 번호 ('A_31', 'T_31', 'O_31' 존재)
    - X_1 ~ X_2875 : 공정 과정에서 추출되어 비식별화된 변수

- sample_submission.csv [파일] - 제출 양식
    - PRODUCT_ID : 제품의 고유 ID
    - Y_Class : 예측한 제품 품질 상태
        - 0 : 적정 기준 미달 (부적합)
        - 1 : 적합
        - 2 : 적정 기준 초과 (부적합)

- 실제 공정 과정에서의 데이터로, 보안상의 이유로 일부 변수가 비식별화 처리 되었습니다. (X변수)
- 'LINE', 'PRODUCT_CODE'는 Train / Test 모두 동일한 종류가 존재합니다.

![alt text](image-1.png)

<br>

### 데이터 분석 및 모델 학습 (링크)
- [CatBoost+Optuna를 확용한 품질 상태 분석](https://github.com/Parksemo/Ark_Project/blob/master/%ED%95%99%EC%8A%B5%EB%AA%A8%EB%8D%B8/catboost%20%2B%20optuna/catboost%20%2B%20optuna.md)

- [Lgbm+ExtraTree+RF를 활용한 분류모델 앙상블](https://github.com/Parksemo/Ark_Project/blob/master/%ED%95%99%EC%8A%B5%EB%AA%A8%EB%8D%B8/softvotin(lgbm%2Crf%2Cextratree)/Lgbm%2BRf%2BExtraTree.md)