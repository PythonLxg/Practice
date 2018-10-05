# -*- coding: utf-8 -*-
# C:\Users\lxg\Documents\Python\Practice
"""
Author:李小根
Time：2018/10/04
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
# from sklearn.model_selection import KFold
# from sklearn.linear_model import LinearRegression
# from sklearn.linear_model import LogisticRegression
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.feature_selection import SelectKBest, f_classif
# from sklearn.ensemble import GradientBoostingClassifier


"""
用决策树模型预测titanic幸存者
"""


def read_dataset(fname):
    # 指定第一列作为行索引
    data = pd.read_csv(fname, index_col=0)
    # 丢弃无用的数据
    data.drop(['Name', 'Ticket', 'Cabin'], axis=1, inplace=True)
    # 处理Sex数据
    data['Sex'] = (data['Sex'] == 'male').astype('int')
    # 处理Embarked数据
    labels = data['Embarked'].unique().tolist()
    data['Embarked'] = data['Embarked'].apply(lambda n: labels.index(n))
    # 处理缺失数据
    data = data.fillna(0)
    return data


train = read_dataset(r'C:\PythonDataSource\Titanic\train.csv')
print(train.head())

y = train['Survived'].values
X = train.drop(['Survived'], axis=1).values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
print('train dataset: {0}; test dataset: {1}'.format(X_train.shape, X_test.shape))
clf = DecisionTreeClassifier()
clf.fit(X_train, y_train)
train_score = clf.score(X_train, y_train)
test_score = clf.score(X_test, y_test)
print('train score: {0}; test score: {1}'.format(train_score, test_score))


# 参数选择 max_depth
def cv_score(d):
    df = DecisionTreeClassifier(max_depth=d)
    df.fit(X_train, y_train)
    tr_score = df.score(X_train, y_train)
    ts_score = df.score(X_test, y_test)
    return tr_score, ts_score


depths = range(2, 15)
scores = [cv_score(d) for d in depths]
tr_scores = [s[0] for s in scores]
cv_scores = [s[1] for s in scores]

# 找出交叉验证数据集评分最高的索引
best_score_index = np.argmax(cv_scores)
best_score = cv_scores[best_score_index]
best_param = depths[best_score_index]
print('best param:{0}; best score: {1}'.format(best_param, best_score))

# 画图分析
plt.figure(figsize=(6, 4), dpi=144)
plt.grid()
plt.xlabel('max depth of decision tree')
plt.ylabel('score')
plt.plot(depths, cv_scores, '.g-', label='cross-validation score')
plt.plot(depths, tr_scores, '.r--', label='training score')
plt.legend()

entropy_thresholds = np.linspace(0, 1, 50)
gini_thresholds = np.linspace(0, 0.5, 50)

# 设置参数矩阵
param_grid = [{'max_depth': range(2, 10)},
              {'min_samples_split': range(2, 30, 2)}]

clf = GridSearchCV(DecisionTreeClassifier(), param_grid, cv=5)
clf.fit(X, y)
print('best param: {0}\nbest score:{1}'.format(clf.best_params_, clf.best_score_))

"""
用线性回归模型预测titanic幸存者
"""
# titanic = pd.read_csv(r'C:\PythonDataSource\Titanic\train.csv')     # 读入数据
# titanic['Age'] = titanic['Age'].fillna(titanic['Age'].median())  # 填充缺失值
# print(titanic['Sex'].unique())
# print(titanic['Embarked'].unique())
# titanic['Embarked'] = titanic['Embarked'].fillna('S')
# titanic.loc[titanic['Sex'] == 'male', 'Sex'] = 0    # 将str类型特征转换成int类型
# titanic.loc[titanic['Sex'] == 'female', 'Sex'] = 1
# titanic.loc[titanic['Embarked'] == 'S', 'Embarked'] = 0
# titanic.loc[titanic['Embarked'] == 'C', 'Embarked'] = 1
# titanic.loc[titanic['Embarked'] == 'Q', 'Embarked'] = 2
# print(titanic['Embarked'].unique())

# category_vars = ['Sex', 'Embarked']
# for var in category_vars:   # 将str类型特征转换成int类型
#     pd.concat([titanic, pd.get_dummies(titanic[var])], 1)
#     titanic = titanic.drop(var, 1)

# predictors = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']
# alg = LinearRegression()
# kf = KFold(titanic.shape[0], n_splits=3)  # 将m个样本平均分成3份进行交叉验证
# predictions = []
# for train, test in kf:
#     train_predictors = titanic[predictors].iloc[train, :]
#     train_target = titanic['Survived'].iloc[train, :]
#     alg.fit(train_predictors, train_target)   # 训练模型
#     test_prediction = alg.predict(titanic[predictors].iloc[test, :])    # 预测数据
#     predictions.append(test_prediction)     # 预测结果
#
# predictions = np.concatenate(predictions, axis=0)   # 将横向的变成纵向的
# predictions[predictions > .5] = 1
# predictions[predictions <= .5] = 0
# accuracy = sum(predictions == titanic['Survived']) / len(predictions)   # 计算预测准确率
# print(accuracy)
