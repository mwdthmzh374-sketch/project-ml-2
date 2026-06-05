
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split,cross_val_score
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestRegressor,GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score

df=pd.read_csv('data/cleaned_mobile_phone_sales_data.csv')

target='Quantity Sold'
X=df.drop(columns=[target])
y=df[target]

cat=X.select_dtypes(include='object').columns
num=X.select_dtypes(exclude='object').columns

preprocessor=ColumnTransformer([
('cat',OneHotEncoder(handle_unknown='ignore'),cat),
('num','passthrough',num)
])

models={
'Random Forest': RandomForestRegressor(
    n_estimators=20,
    random_state=42,
    n_jobs=-1
),
'Gradient Boosting':GradientBoostingRegressor(random_state=42)
}

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)

best_model=None
best_r2=-999

for name,model in models.items():
    pipe=Pipeline([
        ('preprocessor',preprocessor),
        ('model',model)
    ])

    pipe.fit(X_train,y_train)
    pred=pipe.predict(X_test)

    mae=mean_absolute_error(y_test,pred)
    mse=mean_squared_error(y_test,pred)
    rmse=np.sqrt(mse)
    r2=r2_score(y_test,pred)

    #cv=cross_val_score(pipe,X,y,cv=5,scoring='r2').mean()
    cv = 0
    print('\n'+'='*50)
    print(name)
    print('MAE =',mae)
    print('MSE =',mse)
    print('RMSE =',rmse)
    print('R2 =',r2)
    print('Cross Validation R2 =',cv)

    if r2>best_r2:
        best_r2=r2
        best_model=pipe

joblib.dump(best_model,'models/best_model.pkl')
print('\nBest model saved successfully.')
