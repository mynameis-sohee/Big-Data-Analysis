import pandas as pd
from sklearn import preprocessing, ensemble, model_selection, linear_model, feature_selection, pipeline

X_test = pd.read_csv("C:/Users/sohee/Downloads/[Dataset] 작업형 제2유형/X_test.csv", encoding='CP949')
X_train = pd.read_csv("C:/Users/sohee/Downloads/[Dataset] 작업형 제2유형/X_train.csv", encoding='CP949')
y_train = pd.read_csv("C:/Users/sohee/Downloads/[Dataset] 작업형 제2유형/y_train.csv", encoding='CP949')

# 1. 데이터 결측치 등 정보 확인
'''
print(X_test.info())
print(X_train.info())
결과: 결측치 존재하지 않음, 총 10개의 feature로 구성
'''

# 2. 전처리
# 2-1. cust_id

def del_cust_id(df):
    return df.drop(['cust_id'], axis=1)

X_train = del_cust_id(X_train)
X_test = del_cust_id(X_test)

# 2-2. 환불금액
'''
null값이 매우 많아 해당 feature는 삭제한다.
'''
def del_refund(df):
    return df.drop(['환불금액'], axis=1)

X_train = del_refund(X_train)
X_test = del_refund(X_test)

# 2-2. 주구매상품
'''
print(X_train['주구매상품'].value_counts())
print(X_train['주구매지점'].nunique())
결과: 42개 종류로 매우 많다. feature를 묶어 대분류화해야 한다.
'''

# 2-3. 주구매지점
'''
print(X_train['주구매지점'].value_counts())
print(X_train['주구매지점'].nunique())
결과: 24개 종류로 매우 많다. feature를 묶어 대분류화해야 한다.
'''

# 3. 인코딩
'''
범주형 데이터를 정수로 인코딩하는 OrdinalEncoder
'''
encoded = preprocessing.OrdinalEncoder()
encoded.fit(X_train[['주구매상품','주구매지점']])
X_train[['주구매상품','주구매지점']] = encoded.transform(X_train[['주구매상품','주구매지점']])
X_test[['주구매상품','주구매지점']] = encoded.transform(X_test[['주구매상품','주구매지점']])


# 4. 모델링
# 4-1. 로지스틱 회귀분석, 랜덤포레스트
from sklearn import linear_model
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest

pipe_cat = make_pipeline(
    StandardScaler(),
    SelectKBest(),
    RandomForestClassifier())

# 4-2. 튜닝
parameters = {'selectkbest__k':range(5,9), 'randomforestclassifier__max_depth':range(3,10), 'randomforestclassifier__min_samples_split':range(3,8)}

tuning_lr = model_selection.GridSearchCV(
    pipe_cat,
    parameters,
    cv = 3,
    scoring = 'roc_auc'
)


tuning_lr.fit(X_train, y_train['gender'])
pipe_cat = tuning_lr.best_estimator_

pred = pipe_cat.predict(X_test)
