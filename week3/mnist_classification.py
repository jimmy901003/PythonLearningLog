import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
mnist = fetch_openml('mnist_784', version=1)


X, y = mnist["data"], mnist["target"]

X_train, y_train= X[:60000], y[:60000]
X_test, y_test = X[60000:], y[60000:]

y_test.iloc[107]
d=X_train.loc[107].values

plt.imshow(d.reshape(28, 28), cmap='gray')

# 顯示指定索引的圖片及標籤。
def show_minst(xdata, ydata, index):
    d=xdata.iloc[index].values
    plt.imshow(d.reshape(28, 28), cmap='binary')
    plt.grid(visible=True, linestyle='--', alpha=0.8)
    plt.title(f'index{index} is {ydata.iloc[index]}')
    plt.show()
    
show_minst(X_train, y_train, 1012)
    
fac=0.99/255
 
X_train = X_train * fac + 0.01
X_test = X_test * fac + 0.01

cnt = pd.DataFrame()
cnt['class'] = y_train
plt.figure(figsize=(12 ,8)) 
sns.countplot(x=cnt['class'])
plt.show()

y_train = y_train.astype(int)
y_test = y_test.astype(int)

y_train_9 = (y_train == 9).astype(int)
y_test_9 = (y_test==9).astype(int)

from sklearn.linear_model import Perceptron
from sklearn.metrics import confusion_matrix, roc_curve, auc
pen = Perceptron()
pen.fit(X_train, y_train_9)

y_hat = pen.predict(X_test)

from sklearn.metrics import classification_report 
from sklearn.model_selection import cross_val_score

rep = classification_report(y_test_9, y_hat)
print(rep)

# 預測概率
y_scores = pen.decision_function(X_test)

# 混淆矩陣
conf_matrix = confusion_matrix(y_test_9, y_hat)

# 將混淆矩陣轉換為 DataFrame 以便使用 seaborn 繪製
conf_df = pd.DataFrame(conf_matrix, index=['True Negative', 'True Positive'], columns=['Predicted Negative', 'Predicted Positive'])
# 繪製熱圖
plt.figure(figsize=(8, 6))
sns.heatmap(conf_df, annot=True, fmt='d', cmap='Blues', cbar=False)
plt.title('Confusion Matrix')
plt.show()
# ROC 曲線
fpr, tpr, thresholds = roc_curve(y_test_9, y_scores)
roc_auc = auc(fpr, tpr)

# 繪製 ROC 曲線
plt.figure(figsize=(8, 8))
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (AUC = {:.2f})'.format(roc_auc))
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc='lower right')
plt.show()

score = cross_val_score(pen, X_train, y_train_9,cv=5)
print(score)

from sklearn.linear_model import SGDClassifier

clf = SGDClassifier(max_iter=1000, tol=1e-3, random_state=42)

clf.fit(X_train, y_train_9)

y_hat = clf.predict(X_test)

rep = classification_report(y_test_9, y_hat)
print(rep)

# 預測概率
y_scores = clf.decision_function(X_test)

# 混淆矩陣
conf_matrix = confusion_matrix(y_test_9, y_hat)

# 將混淆矩陣轉換為 DataFrame 以便使用 seaborn 繪製
conf_df = pd.DataFrame(conf_matrix, index=['True Negative', 'True Positive'], columns=['Predicted Negative', 'Predicted Positive'])
# 繪製熱圖
plt.figure(figsize=(8, 6))
sns.heatmap(conf_df, annot=True, fmt='d', cmap='Blues', cbar=False)
plt.title('Confusion Matrix')
plt.show()
# ROC 曲線
fpr, tpr, thresholds = roc_curve(y_test_9, y_scores)
roc_auc = auc(fpr, tpr)

# 繪製 ROC 曲線
plt.figure(figsize=(8, 8))
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (AUC = {:.2f})'.format(roc_auc))
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc='lower right')
plt.show()

score = cross_val_score(clf, X_train, y_train_9,cv=5)
print(score)

from sklearn.linear_model import LogisticRegression

lr = LogisticRegression(max_iter=500)

lr.fit(X_train, y_train)

y_hat = lr.predict(X_test)

print(lr.score(X_test, y_test))

rep = classification_report(y_test, y_hat)
print(score)

conf_matrix = confusion_matrix(y_test, y_hat)
print(conf_matrix)

# 將混淆矩陣轉換為 DataFrame 以便使用 seaborn 繪製
conf_df = pd.DataFrame(conf_matrix, index=range(10), columns=range(10))

# 繪製熱圖
plt.figure(figsize=(10, 8))
sns.heatmap(conf_df, annot=True, fmt='g', cmap='Blues', cbar=False)
plt.xlabel('Predicted Value')
plt.ylabel('Actual Value')
plt.title('Confusion Matrix Heatmap')
plt.show()

from matplotlib.gridspec import GridSpec

def plot_image_and_bar(model, x_data, iloc):

    probs = model.predict_proba([x_data.iloc[iloc]])[0]
    probs_df = pd.DataFrame({'Probability': probs})
    
    image_data = x_data.iloc[iloc].values
    
    plt.figure(figsize=(14, 8))
    
    gs = GridSpec(1, 2, width_ratios=[1, 1])

    plt.subplot(gs[0])
    plt.imshow(image_data.reshape(28, 28), cmap='binary')
    plt.title('Image')
    plt.grid(False)

    plt.subplot(gs[1])
    sns.barplot(x=probs_df.index, y='Probability', data=probs_df)
    plt.title('Class Probability Distribution')
    plt.grid(True)
    
    plt.tight_layout()
    
    plt.show()

plot_image_and_bar(lr, X_test, 10)

# yellowbrick套件繪製ClassificationReport
from yellowbrick.classifier import ClassificationReport

digits = ['%d' % i for i in range(10)]

plt.figure(figsize=(12, 8))
visualizer = ClassificationReport(lr, classes=digits, support=True, cmap='Blues')

visualizer.score(X_test, y_test)

visualizer.show()

# mlxtend繪製confusion_matrix
from mlxtend.plotting import plot_confusion_matrix as plot_cm3

ml_cm = confusion_matrix(y_test, y_hat)

plot_cm3(conf_mat=ml_cm, show_normed=True, colorbar=True, figsize=(12, 8))

from sklearn.neural_network import MLPClassifier

# 多層感知器模型建構
mlp = MLPClassifier(hidden_layer_sizes=(128,), 
                    max_iter=100, 
                    activation='tanh',
                    solver='adam', 
                    tol=1E-4, random_state=0, 
                    learning_rate_init=0.005,
                    shuffle=True, 
                    verbose=False, 
                    learning_rate='adaptive'
                    )

mlp.fit(X_train, y_train)

y_hat = mlp.predict(X_test)

rep = classification_report(y_test, y_hat)
print(rep)
plot_image_and_bar(mlp, X_test, 9986)

from sklearn.neighbors import KNeighborsClassifier

# KNeighborsClassifier 模型
knn = KNeighborsClassifier(n_neighbors=3, algorithm='auto')
knn.fit(X_train, y_train)

y_hat = knn.predict(X_test)

rep = classification_report(y_test, y_hat)
print(rep)


  
