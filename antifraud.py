# -*- coding: utf-8 -*-
"""
Created on Sun Mar 25 12:04:48 2018

@author: alex
"""

import matplotlib.pylab as plt
import numpy as np
import pandas as pd

data = pd.read_csv("/home/alex/Downloads/creditcard.csv")
# print(data.head())
#0 - positive，1 - negetive

count_classes = pd.value_counts(data['Class'], sort=True).sort_index()
count_classes.plot(kind = 'bar', alpha=0.5)
plt.title('Fraud class histogram')
plt.xlabel('Class')
plt.ylabel('Frequency')
# plt.show()


from sklearn.preprocessing import StandardScaler

#fit_transform()：data transformations,and add columes to the data
data['normAmount'] = StandardScaler().fit_transform(data['Amount'].reshape(-1, 1))
data = data.drop(['Time', 'Amount'], axis=1)
# print(data.head())

#oversample
X = data.ix[:, data.columns != 'Class']
y = data.ix[:, data.columns == 'Class']
# print(X)
# print(y)

#the number of samoles
number_records_fraud = len(data[data.Class == 1])
#index of samples which is 1
fraud_indices = np.array(data[data.Class == 1].index)
#index of samples which is 0
normal_indices = data[data.Class == 0].index

# print(number_records_fraud)
# print(fraud_indices)
# print(normal_indices)

#same less 
#choose the samples randomly(class=0)
#随机选取样本中的数据，随机选择：np.random.choice(Class为0的数据索引=>样本，选择多少数据量，是否选择代替=False)
random_normal_indeics = np.random.choice(normal_indices, number_records_fraud, replace=False)
random_normal_indeics = np.array(random_normal_indeics)

# connect the sample between class=0 and class =1
#连接Class=1的数据索引和随机选取的Class=0的数据索引
under_sample_indeice = np.concatenate([fraud_indices, random_normal_indeics])
# print(under_sample_indeice)

#undersample
#下采样（选取该索引下的数据）
under_sample_data = data.iloc[under_sample_indeice, :]

X_undersample = under_sample_data.ix[:, under_sample_data.columns != 'Class']
y_undersample = under_sample_data.ix[:, under_sample_data.columns == 'Class']

#Showing Ratio
# print('Percentage oif normal transactions:', len(under_sample_data[under_sample_data.Class == 0]) / len(under_sample_data))
# print('Percentage oif fraud transactions:', len(under_sample_data[under_sample_data.Class == 1]) / len(under_sample_data))
# print('Total number of transactions in resampled data:', len(under_sample_data))

#cross validate,split data
#交叉验证， train_test_split: 切分数据
from sklearn.cross_validation import  train_test_split

#divide  the data into train set and test set
#将数据集分成训练集0.7和测试集0.3
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

# print('Number transactions train dataset:', len(X_train))
# print('Number transactions test dataset:', len(X_test))
# print('Total number of transaction:', len(X_train) + len(X_test))

#same operation to the undersample set
#对下采样数据集进行相同的操作
#just use the undersample set to train and finally use the original data set to test
#只采用下采样数据集进行训练，最终用原始数据集测试
X_train_undersample, X_test_undersample, y_train_undersample, y_test_undersample = train_test_split(X_undersample, y_undersample, test_size=0.3, random_state=0)
# print('Number transactions train dataset:', len(X_train_undersample))
# print('Number transactions test dataset:', len(X_test_undersample))
# print('Total number of transaction:', len(X_train_undersample) + len(X_test_undersample))


#在类不平衡问题中，用精度来衡量指标是骗人的！没有意义(1000个人，全部预测为正常， 0个癌症) 精度 =  TP / Total
# 这里使用Recall(召回率, 查全率) =  TP / TP+FN , 1000个人(990个正常,10个癌症),如果检测出0个病人  0/10=0,检测出2个病人2/10=0.2
# 检测任务上，通常用Recall作为模型的评估标准
from sklearn.linear_model import LogisticRegression
#KFold->做几倍的交叉验证
from sklearn.cross_validation import KFold
# confusion_matrix:混淆矩阵
from sklearn.metrics import confusion_matrix, recall_score


def printing_Kfold_score(x_train_data, y_train_data):
    #切分成5部分，将原始数据集进行切分
    fold = KFold(len(y_train_data), 5, shuffle=False)

    #正则化惩罚项
    #希望当前模型的泛化能力更好，不仅满足训练数据，还要在测试数据上尽可能满足
    #浮动的差异小 ====> 过拟合的风险小
    c_param_range = [0.01, 0.1, 1, 10, 100]

    results_table = pd.DataFrame(index = range(len(c_param_range), 2), columns = ['C_parameter', 'Mean recall score'])
    results_table['C_parameter'] = c_param_range

    j = 0
    #查看哪一个c值比较好
    for c_param in c_param_range:
        print('------------------------------------')
        print('C paramter:', c_param)
        print('------------------------------------')
        print('')

        recall_accs = []
        for iteration, indices in enumerate(fold, start=1):
            #L1惩罚或者L2惩罚
            lr = LogisticRegression(C = c_param, penalty='l1')

            #进行训练，交叉验证数据 ---> 建立模型
            lr.fit(x_train_data.iloc[indices[0], :], y_train_data.iloc[indices[0], :].values.ravel())

            #进行预测，用交叉验证中的验证集 再进行一个验证测操作
            y_pred_undersameple = lr.predict(x_train_data.iloc[indices[1], :].values)

            #计算当前模型的recall(indices[1]:测试集)
            recall_acc = recall_score(y_train_data.iloc[indices[1], :].values, y_pred_undersameple)
            recall_accs.append(recall_acc)
            print("Iterator", iteration, ': recall score = ', recall_acc)


        results_table.ix[j, 'Mean recall score'] = np.mean(recall_accs)
        j += 1
        print('')
        print('Mean recall score', np.mean(recall_accs))
        print('')

    # print(results_table)

    # best_c = results_table.loc[results_table['Mean recall score'].idxmax()]['C_parameter']

    # Finally, we can check which C parameter is the best amongst the chosen.
    # print('*********************************************************************************')
    # print('Best model to choose from cross validation is with C parameter = ', best_c)
    # print('*********************************************************************************')
    # return best_c

printing_Kfold_score(X_train_undersample, y_train_undersample)


#混淆矩阵
def plot_confusion_matrix(cm, classes, title = 'Confusion matrix', cmap = plt.cm.Blues):
    plt.imshow(cm, interpolation='nearest', cmap = cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=0)
    plt.yticks(tick_marks, classes)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment='center',
                 color='white' if cm[i, j] > thresh else 'black')

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

import itertools
lr = LogisticRegression(C = 0.1, penalty='l1')
lr.fit(X_train_undersample, y_train_undersample.values.ravel())
y_pred_undersample = lr.predict(X_test_undersample)

#计算混淆矩阵
cnf_matrix = confusion_matrix(y_test_undersample, y_pred_undersample)
np.set_printoptions(precision=2)

print('Recal metric in the testing dataset: ', cnf_matrix[1, 1] / (cnf_matrix[1, 0] + cnf_matrix[1, 1]))

class_names = [0,1]
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=class_names, title = 'Confusion matrix')
# plt.show()



lr = LogisticRegression(C = 0.1, penalty='l1')
lr.fit(X_train_undersample, y_train_undersample.values.ravel())
y_pred = lr.predict(X_test.values)

#用原数据来生成混淆矩阵
cnf_matrix = confusion_matrix(y_test, y_pred)
np.set_printoptions(precision=2)

print('Recall metric in the testing dataset:', cnf_matrix[1, 1] / (cnf_matrix[1, 0] + cnf_matrix[1, 1]))

class_names = [0, 1]
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=class_names, title='Confusion matrix')

# plt.show()


#测试阈值对结果的影响
lr = LogisticRegression(C = 0.01, penalty='l1')
lr.fit(X_train_undersample, y_train_undersample.values.ravel())
y_pred_undersample_proba = lr.predict_proba(X_test_undersample.values)

#设置不同的阈值
thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

plt.figure(figsize=(10, 10))

j = 1
for i in thresholds:
    y_test_undersample_high_recall = y_pred_undersample_proba[:, 1] > i

    plt.subplot(3, 3, j)
    j += 1

    #计算混淆矩阵
    cnf_matrix = confusion_matrix(y_test_undersample, y_test_undersample_high_recall)
    np.set_printoptions(precision=2)

    print('Recal metrix in the testing dataset:', cnf_matrix[1, 1] / (cnf_matrix[1, 0] + cnf_matrix[1, 1]))

    class_names = [0, i]
    plot_confusion_matrix(cnf_matrix, classes=class_names, title='Threshold >= %s' %i)

# plt.show()



# #采取过采样的策略 ---- SMOTE样本生成策略
# from imblearn.over_sampling import SMOTE
# from sklearn.ensemble import RandomForestClassifier
# 
# 
# credit_cards = pd.read_csv('creditcard.csv')
# 
# columns = credit_cards.columns
# # The labels are in the last column ('Class'), Simply remove it to obtain features cloumns
# features_columns = columns.delete(len(columns) - 1)
# 
# features = credit_cards[features_columns]
# labels = credit_cards['Class']
# features_train, features_test, labels_train, labels_test = train_test_split(features,
#                                                                             labels,
#                                                                             test_size=0.2,
#                                                                             random_state=0)
# oversampler = SMOTE(random_state=0)
# os_features, os_labels = oversampler.fit_sample(features_train, labels_train)
# 
# len(os_labels[os_labels == 1])