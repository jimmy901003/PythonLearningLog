import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
pd.set_option('display.max_columns', None)

from sklearn.datasets import load_wine

# 載入wine數據集
wine_data = load_wine()

# 將特徵數據轉換為DataFrame
df = pd.DataFrame(wine_data.data, columns=wine_data.feature_names)
df.columns
# 添加目標列
df['target'] = wine_data.target

def plot_heatmap(df, cmap="YlGnBu"):
    # 繪製熱力圖
    plt.figure(figsize=(12, 8))
    df_corr = df.corr()
    sns.heatmap(df_corr, cmap=cmap, annot=True, fmt=".2f")
    plt.show()

# 繪製特徵相關性熱力圖
plot_heatmap(df)

# 添加類別列
df['class'] = df['target'].map({0: 'class1', 1: 'class 2', 2: 'class3'})

# 繪製直方圖
df.hist(figsize=(14,10))
plt.show()

# 繪製目標類別分佈圖
plt.figure(figsize=(10, 8))
sns.countplot(x=df['target'])
plt.show()

# 繪製alcalinity_of_ash直方圖
plt.figure(figsize=(12, 8))
sns.histplot(x=df['alcalinity_of_ash'], kde=1)
plt.show()

# 繪製alcohol直方圖
plt.figure(figsize=(12, 8))
sns.histplot(x=df['alcohol'], kde=1)
plt.show()

# 繪製alcohol核密度估計曲線
plt.figure(figsize=(12, 8))
sns.kdeplot(df['alcohol'], label='total dist')
sns.kdeplot(df['alcohol'][df['target'] == 0], label='class 0')
sns.kdeplot(df['alcohol'][df['target'] == 1], label='class 1')
sns.kdeplot(df['alcohol'][df['target'] == 2], label='class 2')
plt.legend()
plt.show()

# 讀取csv檔案並進行資料前處理
df = pd.read_csv(r"E:\0\機器學習\adventures\wine quality\winequality-red.csv", sep=';')
thresholds = {3: 0, 4: 0, 5: 1, 6: 1, 7: 2, 8: 2}
df['quality'] = df['quality'].map(thresholds)

# 繪製資料集特徵相關性熱力圖
plot_heatmap(df)

# 繪製alcohol箱形圖
plt.figure(figsize=(12, 8))
sns.boxplot(x=df['alcohol'])
plt.show()

# 繪製quality分佈圖
plt.figure(figsize=(12, 8))
sns.countplot(x=df['quality'])
plt.show()

# 依quality類別繪製alcohol核密度估計曲線
plt.figure(figsize=(12, 8))
for i in np.sort(df['quality'].unique()):
    sns.kdeplot(df['alcohol'][df['quality']==i], label=f'class ={i}')
    
plt.legend()    
plt.show()

# 機器學習模型部分
plt.figure(figsize=(12, 8))

from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC, SVC
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
from sklearn.model_selection import cross_val_score

# 分割資料集
X = df.drop(['quality'], axis=1)
y = df['quality']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 特徵標準化
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.fit_transform(X_test)

# 使用線性支持向量機模型
clf = LinearSVC()
clf.fit(X_train, y_train)

# 預測並評估模型
y_hat = clf.predict(X_test)
rep = classification_report(y_test, y_hat)
print(rep)

# 使用多個機器學習模型進行評估
from sklearn.linear_model import SGDClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier, RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

classifiers = [
    LogisticRegression(),
    LinearSVC(),
    SVC(), 
    SGDClassifier(),
    MLPClassifier(),
    AdaBoostClassifier(),
    GradientBoostingClassifier(),  
    RandomForestClassifier(),
    GaussianNB(),
    KNeighborsClassifier(),
]

results = []
for classifier in classifiers:
    # 交叉驗證評估模型性能
    scores = cross_val_score(classifier, X_train, y_train, cv=5)
    cls_name = classifier.__class__.__name__
    accuracy = scores.mean()
    classifier.fit(X_train, y_train)
    std = scores.std()
    test_y_pred = classifier.predict(X_test)
    test_accuracy = np.mean(test_y_pred == y_test)
    train_y_pred = classifier.predict(X_train)
    train_accuracy = np.mean(train_y_pred == y_train)
    results.append({
        "Classifier": cls_name,
        "Accuracy": accuracy,
        "Std": std,
        "Test_Accuracy": test_accuracy,
        "Train_Accuracy": train_accuracy
    })

# 將評估結果轉換為DataFrame
df_results = pd.DataFrame(results)

# 定義隨機森林模型
rf_model = RandomForestClassifier(
    n_estimators=200,
    max_depth=None,
    min_samples_split=2,
    min_samples_leaf=2
)

# # 設定超參數的範圍
# param_grid = {
#     'n_estimators': [50, 100, 200],
#     'max_depth': [None, 10, 20],
#     'min_samples_split': [2, 5, 10],
#     'min_samples_leaf': [1, 2, 4]
# }

# # 創建 GridSearchCV 物件
# grid_search = GridSearchCV(estimator=rf_model, param_grid=param_grid, cv=5)

# grid_search.fit(X_train, y_train)

# # 印出最佳的超參數組合
# print("最佳超參數組合:", grid_search.best_params_)

# # 使用最佳模型進行預測
# best_model = grid_search.best_estimator_
# y_pred = best_model.predict(X_test)

# # 在測試集上評估模型性能
# accuracy = np.mean(y_pred == y_test)
# print("測試集準確率:", accuracy)

from yellowbrick.model_selection import LearningCurve

visualizer = LearningCurve(rf_model, cv=5, train_sizes=np.linspace(0.1, 1.0, 10), scoring='accuracy')

visualizer.fit(X_train, y_train)

# 繪製學習曲線
visualizer.show()


