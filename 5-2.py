# -*- coding: utf-8 -*-
"""
Created on Thu Feb 16 16:00:32 2017

@author: luzhangqin
"""

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV

'''
low_memory = False
在内部以块的方式处理文件，
导致解析时内存使用较少，但可能是混合类型推断。 
要确保没有混合类型，请设置False，或使用dtype参数指定类型。 
请注意，整个文件将读入单个DataFrame，无论如何，
请使用chunksize或iterator参数以块形式返回数据。 （只有C解析器有效）
'''
df = pd.read_csv('internet-ads/ad.data', sep = ',', header = None, low_memory = False)
#set
explanatory_variable_columns = set(df.columns.values)
response_variable_column = df[len(df.columns.values)-1]
explanatory_variable_columns.remove(len(df.columns.values)-1)
y = [1 if e == 'ad.' else 0 for e in response_variable_column]
#.loc
# a:b   提取数据a->b包括ab
#list(explanatory_variable_columns)选择的标签
X = df.loc[:, list(explanatory_variable_columns)]
#匹配to_replace的regexs将被替换为value
#inplace = True 则返回调用者。
#如果这是True，则to_replace必须是字符串。 否则，to_replace必须为None。
X.replace(to_replace=' *\?', value = -1, regex = True, inplace = True)
X_train, X_test, y_train, y_test =  train_test_split(X, y)
#测量分割质量的功能。 支持的标准是基尼杂质的“gini”和信息增益的“熵”。
pipeline = Pipeline([
    ('clf', RandomForestClassifier(criterion = 'entropy'))
])

parameters = {
    #树的数量
    'clf__n_estimators': (5, 10, 20, 50),
    'clf__max_depth': (150, 155, 160),
    #拆分内部节点所需的最小样本数：
    #最小是2
    'clf__min_samples_split': (2, 3, 4),
    #叶节点所需的最小样本数：
    'clf__min_samples_leaf': (1, 2, 3),
}

grid_search = GridSearchCV(pipeline, parameters, n_jobs = -1, verbose = 1, scoring = 'f1', cv = 3)
grid_search.fit(X_train, y_train)
print('最佳效果: %0.3f' % grid_search.best_score_)
print('最优参数组合： ')
best_parameters = grid_search.best_estimator_.get_params()

for param_name in sorted(parameters.keys()):
    print('\t%s: %r' %(param_name, best_parameters[param_name]))
predictions = grid_search.predict(X_test)
print('分类报告: ', classification_report(y_test, predictions))