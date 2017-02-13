# -*- coding: utf-8 -*-
"""
Created on Sun Feb 12 14:44:08 2017

@author: luzhangqin
"""

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model.logistic import LogisticRegression
#网格搜索
#from sklearn.grid_search import GridSearchCV
from sklearn.model_selection import GridSearchCV
#pipeline的目的就是当设置不同的参数时组合几个可以一起交叉验证的步骤。
#所以可以使用组合这几个步骤的名字和它们的属性参数
#（不过需要在参数前面加_来连接）
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, accuracy_score

pipeline = Pipeline([
    ('vect', TfidfVectorizer(stop_words = 'english')),
    ('clf', LogisticRegression())
])

parameters = {
    #max_df：float in range [0.0，1.0]或int，default = 1.0
    #当构建词汇时忽略具有严格高于给定阈值的文档频率（语料库特定停止词）的词语。 
    #如果float，该参数表示文档的比例，整数绝对计数。 如果词汇不为None，则忽略此参数。
    'vect__max_df': (0.25, 0.5, 0.75),
    'vect__stop_words': ('english', None),
    #int或None，default = None
    #如果不是无，建立一个词汇，只考虑顶部的max_features按词频率排序的语料库。
    #如果词汇不为None，则忽略此参数。
    'vect__max_features': (2500, 5000, 10000, None),
    #元组（min_n，max_n）
    #要提取的不同n-gram的n值范围的下限和上限。 
    #将使用n的所有值，使得min_n <= n <= max_n。
    'vect__ngram_range': ((1, 1), (1, 2)),
    #boolean，default = True
    #启用反文档频率重新加权。
    'vect__use_idf': (True, False),
    #'l1'，'l2'或None，可选
    #Norm用于归一化词向量。 无无标准化。
    'vect__norm': ('l1', 'l2'),
    #str，'l1'或'l2'，default：'l2'
    #用于指定在惩罚中使用的规范。 'newton-cg'，'sag'和'lbfgs'求解器只支持l2的惩罚。
    'clf__penalty': ('l1', 'l1'),
    #float，默认值：1.0
    #反正规化强度; 必须是正浮点数。 
    #像支持向量机一样，较小的值指定更强的正则化。
    'clf__C': (0.01, 0.1, 1.0, 10.0),
}
#n_jibs并行运行的作业数。
#verbose控制详细程度：越高，消息越多。
grid_search = GridSearchCV(pipeline, parameters, n_jobs = 10, verbose = 1, scoring = 'accuracy', cv = 3)

df = pd.read_csv('smsspamcollection/SMSSpamCollection', sep = '\t', header = None, names=["label","message"]) 
df['label']=pd.factorize(df['label'])[0]
X, y = df['message'], df['label']
X_train, X_test, y_train, y_test =  train_test_split(X, y)
grid_search.fit(X_train, y_train)
print('最佳效果: %0.3f' % grid_search.best_score_)
print('最优参数组合： ')
best_parameters = grid_search.best_estimator_.get_params()

for param_name in sorted(parameters.keys()):
    print('\t%s: %r' %(param_name, best_parameters[param_name]))
predictions = grid_search.predict(X_test)
print('准确率: ', accuracy_score(y_test, predictions))
print('精确率: ', precision_score(y_test, predictions))
print('召回率: ', recall_score(y_test, predictions))

