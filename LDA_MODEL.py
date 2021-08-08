#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul 17 22:26:19 2021

@author: nick
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul 17 21:14:34 2021

@author: nick
"""

import pyLDAvis.sklearn
import pyLDAvis
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import pandas as pd
import re
import os

# 待做 LDA 的文本 csv 文件，可以是本地文件，也可以是远程文件，一定要保证它是存在的！！！！
source_csv_path = 'out.csv'
# 文本 csv 文件里面文本所处的列名,注意这里一定要填对，要不然会报错的！！！
document_column_name = 'comments'
# 输出主题词的文件路径
top_words_csv_path = 'top-topic-words-3.csv'
# 输出各文档所属主题的文件路径
predict_topic_csv_path = 'document-distribution-3.csv'
# 可视化 html 文件路径
html_path = 'document-lda-visualization-3.html'
# 选定的主题数
n_topics = 2
# 要输出的每个主题的前 n_top_words 个主题词数
n_top_words = 10
# 去除无意义字符的正则表达式
pattern = u'[\\s\\d,.<>/?:;\'\"[\\]{}()\\|~!\t"@#$%^&*\\-_=+a-zA-Z，。\n《》、？：；“”‘’｛｝【】（）…￥！—┄－]+'



d3=pd.read_csv(r'out.csv')

# 构造 tf-idf
tf_idf_vectorizer = TfidfVectorizer()
tf_idf = tf_idf_vectorizer.fit_transform(d3['clean_comments'])

lda = LatentDirichletAllocation(
    n_components=n_topics,
    max_iter=50,
    learning_method='online',
    learning_offset=50,
    random_state=0)

# 使用 tf_idf 语料训练 lda 模型
lda.fit(tf_idf)

# 计算 n_top_words 个主题词
top_words_df = top_words_data_frame(lda, tf_idf_vectorizer, n_top_words)

# 保存 n_top_words 个主题词到 csv 文件中
top_words_df.to_csv(top_words_csv_path, encoding='utf-8-sig', index=None)

# 转 tf_idf 为数组，以便后面使用它来对文本主题概率分布进行计算
X = tf_idf.toarray()

# 计算完毕主题概率分布情况
predict_df = predict_to_data_frame(lda, X)

# 保存文本主题概率分布到 csv 文件中
predict_df.to_csv(predict_topic_csv_path, encoding='utf-8-sig', index=None)

# 使用 pyLDAvis 进行可视化
data = pyLDAvis.sklearn.prepare(lda, tf_idf, tf_idf_vectorizer)
pyLDAvis.save_html(data, html_path)
# 清屏
os.system('clear')
# 浏览器打开 html 文件以查看可视化结果
os.system(f'start {html_path}')

print('本次生成了文件：',
      top_words_csv_path,
      predict_topic_csv_path,
      html_path)


d1=pd.read_csv(r'document-distribution.csv')
y1 = d3.iloc[:, 4]

d4=pd.concat([d1,y1],axis=1) 

X=d4.iloc[:,0:2].values
y=d4.iloc[:,2].values


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import linear_model

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)
from imblearn.over_sampling import SMOTE

oversample = SMOTE()
X_train, y_train = oversample.fit_resample(X_train, y_train)



from sklearn.model_selection import GridSearchCV
import warnings
warnings.filterwarnings("ignore")

# parameter setting
params = {'C':[0.0001, 1, 100, 1000],
          'max_iter':[1, 10, 100, 500],
          'class_weight':['balanced', None],
          'solver':['liblinear','sag','lbfgs','newton-cg']
         }
model=linear_model.LogisticRegression()
clf = GridSearchCV(model, param_grid=params, cv=10)
clf.fit(X_train,y_train)
clf.best_params_
clf.best_score_

# 根据以上绘图结果选择一个合理参数值
classifier = LogisticRegression(**clf.best_params_)
# 训练模型
classifier.fit(X_train, y_train)

# 预测测试集结果
y_pred = classifier.predict(X_test)
y_score = classifier.predict_proba(X_test)[:,1]

# roc/auc计算
y_score = classifier.predict_proba(X_test)[:,1]
fpr,tpr,threshold = metrics.roc_curve(y_test, y_score)
roc_auc = metrics.auc(fpr,tpr)


# 绘制roc曲线
f,ax = plt.subplots(figsize=(8,6))
plt.stackplot(fpr, tpr, color='#338DFF', alpha = 0.5, edgecolor = 'black')
plt.plot(fpr, tpr, color='black', lw = 1.5)
plt.plot([0,1],[0,1], color = 'red', linestyle = '--', lw = 2)
plt.text(0.5,0.3,'ROC curve (area = %0.2f)' % roc_auc)
plt.xlabel('1-Specificity')
plt.ylabel('Sensitivity')
plt.show()


# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
print(cm)
from sklearn.metrics import accuracy_score
p=accuracy_score(y_test, y_pred)
from sklearn.metrics import f1_score
f1=f1_score(y_test, y_pred)
from sklearn.metrics import recall_score
recall=recall_score(y_test, y_pred)
print(p)
print(f1)
print(recall)


from matplotlib.colors import ListedColormap
X_set,y_set=X_train,y_train
X1,X2=np.meshgrid(np.arange(start=X_set[:,0].min()-1, stop=X_set[:, 0].max()+1, step=0.01),
                   np.arange(start=X_set[:,1].min()-1, stop=X_set[:,1].max()+1, step=0.01))

plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(),X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('#C65749', '#338DFF')))
plt.xlim(X1.min(),X1.max())
plt.ylim(X2.min(),X2.max())
for i,j in enumerate(np. unique(y_set)):
    plt.scatter(X_set[y_set==j,0],X_set[y_set==j,1],
                c = ListedColormap(('red', 'blue'))(i), label=j)

plt. title(' LOGISTIC(Training set)')
plt. xlabel(' Age')
plt. ylabel(' Estimated Salary')
plt. legend()
plt. show()

X_set,y_set=X_test,y_test
X1,X2=np. meshgrid(np. arange(start=X_set[:,0].min()-1, stop=X_set[:, 0].max()+1, step=0.01),
                   np. arange(start=X_set[:,1].min()-1, stop=X_set[:,1].max()+1, step=0.01))

plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(),X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('#C65749', '#338DFF')))
plt.xlim(X1.min(),X1.max())
plt.ylim(X2.min(),X2.max())
for i,j in enumerate(np. unique(y_set)):
    plt.scatter(X_set[y_set==j,0],X_set[y_set==j,1],
                c = ListedColormap(('red', 'blue'))(i), label=j)

plt.title(' LOGISTIC(Test set)')
plt.xlabel(' Age')
plt.ylabel(' Estimated Salary')
plt.legend()
plt.show()





