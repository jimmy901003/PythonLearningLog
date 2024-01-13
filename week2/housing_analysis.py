import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import statsmodels.api as sm

def load_california_housing():
    diabetes = fetch_california_housing()
    X, y = diabetes.data, diabetes.target
    feature_names = diabetes.feature_names
    df = pd.DataFrame(data=X, columns=feature_names)
    df['target'] = y
    return df

def feature_engineering(df):
    df['TotalBedrooms'] = df['AveRooms'] * df['AveBedrms']
    df['PopulationDensity'] = df['Population'] / df['AveRooms']
    df['BedroomAgeRatio'] = df['AveBedrms'] / df['HouseAge']
    df['LocationFeature'] = df['Latitude'] + df['Longitude']
    return df

def plot_heatmap(df, cmap="YlGnBu"):
    plt.figure(figsize=(12, 8))
    df_corr = df.corr()
    sns.heatmap(df_corr, cmap=cmap, annot=True, fmt=".2f")
    plt.show()


def categorize_MedInc(df, target_col='target', medinc_col='MedInc'):
    df['MedInc_Category'] = pd.qcut(df[medinc_col], q=[0, 1/3, 2/3, 1], labels=[1, 2, 3])
    X = df.drop([target_col], axis=1)
    y = df[target_col]
    return X, y

def train_and_evaluate_model(model, X_train, y_train, X_test, y_test):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    return mse, r2

# 加載數據
df = load_california_housing()

# 特徵工程
df = feature_engineering(df)

# 繪製熱度圖
plot_heatmap(df)

# 視覺化特徵關係
plt.figure(figsize=(12, 6))
sns.jointplot(x='MedInc', y='target', data=df)
plt.show()

# 將MedInc分類
X, y = categorize_MedInc(df)

# 分割數據集為訓練集和測試集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 初始化線性回歸模型
lr_model = LinearRegression()

# 訓練並評估線性回歸模型
mse_lr, r2_lr = train_and_evaluate_model(lr_model, X_train, y_train, X_test, y_test)
print(f'線性回歸模型測試集均方誤差（MSE）：{mse_lr:.4f}')
print(f'線性回歸模型測試集 R^2 分數：{r2_lr:.4f}')

# 初始化隨機森林模型
rf_model = RandomForestRegressor()

# 訓練並評估隨機森林模型
mse_rf, r2_rf = train_and_evaluate_model(rf_model, X_train, y_train, X_test, y_test)
print(f'隨機森林模型測試集均方誤差（MSE）：{mse_rf:.4f}')
print(f'隨機森林模型測試集 R^2 分數：{r2_rf:.4f}')
