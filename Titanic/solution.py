# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn import linear_model
from sklearn import svm
from matplotlib.font_manager import FontProperties
font = FontProperties(fname=r"c:\windows\fonts\simsun.ttc", size=14)    # 解决windows环境下画图汉字乱码问题



'''逻辑回归模型'''
def solution_logisticRegression():
    train_data = pd.read_csv(r"data/train.csv")
    print u"数据信息：\n",train_data.info()
    print u'数据描述：\n',train_data.describe()  
    display_data(train_data)  # 简单显示数据信息
    display_with_process(train_data) # 根据数据的理解，简单处理一下数据显示,验证猜想
    process_data = pre_processData(train_data,'process_train_data')  # 数据预处理，要训练的数据
    train_data = process_data.filter(regex='Survived|Age_.*|SibSp|Parch|Fare_.*|Cabin_.*|Embarked_.*|Sex_.*|Pclass_.*')  # 使用正则抽取想要的数据
    train_np = train_data.as_matrix()  # 转为矩阵
    '''训练model'''
    X = train_np[:,1:]
    y = train_np[:,0]
    model = linear_model.LogisticRegression(C=1.0,tol=1e-6).fit(X,y)
    '''测试集上预测'''
    test_data = pd.read_csv(r"data/test.csv")
    process_test_data = pre_processData(test_data,'process_test_data')  # 预处理数据
    test_data = process_test_data.filter(regex='Age_.*|SibSp|Parch|Fare_.*|Cabin_.*|Embarked_.*|Sex_.*|Pclass_.*')
    test_np = test_data.as_matrix()
    predict = model.predict(test_np)
    result = pd.DataFrame(data={'PassengerId':process_test_data['PassengerId'].as_matrix(),'Survived':predict.astype(np.int32)})
    result.to_csv(r'logisticRegression_result/prediction.csv',index=False)
    
'''SVM模型''' 
def solution_svm():
    train_data = pd.read_csv(r"data/train.csv")
    print u"数据信息：\n",train_data.info()
    print u'数据描述：\n',train_data.describe()  
    #display_data(train_data)  # 简单显示数据信息
    #display_with_process(train_data) # 根据数据的理解，简单处理一下数据显示,验证猜想
    process_data = pre_processData(train_data,'process_train_data')  # 数据预处理，要训练的数据
    train_data = process_data.filter(regex='Survived|Age_.*|SibSp|Parch|Fare_.*|Cabin_.*|Embarked_.*|Sex_.*|Pclass_.*')  # 使用正则抽取想要的数据
    train_np = train_data.as_matrix()  # 转为矩阵
    '''训练model'''
    X = train_np[:,1:]
    y = train_np[:,0]
    model = svm.SVC(tol=1e-6).fit(X,y)
    
    '''测试集上预测'''
    test_data = pd.read_csv(r"data/test.csv")
    process_test_data = pre_processData(test_data,'process_test_data')  # 预处理数据
    test_data = process_test_data.filter(regex='Age_.*|SibSp|Parch|Fare_.*|Cabin_.*|Embarked_.*|Sex_.*|Pclass_.*')
    test_np = test_data.as_matrix()
    predict = model.predict(test_np)
    result = pd.DataFrame(data={'PassengerId':process_test_data['PassengerId'].as_matrix(),'Survived':predict.astype(np.int32)})
    result.to_csv(r'svm_result/prediction.csv',index=False)
    
  
'''数据预处理'''  
def pre_processData(train_data,file_path):
    train_data.loc[(train_data.Age.isnull()), 'Age' ] = np.mean(train_data.Age)  # 为空的年龄补为平均年龄
    train_data.loc[(train_data.Cabin.isnull(),'Cabin')] = 'no'    # Cabin为空的设为no
    train_data.loc[(train_data.Cabin.notnull(),'Cabin')] = 'yes'  
    
    '''0/1对应处理'''
    dummies_cabin = pd.get_dummies(train_data['Cabin'],prefix='Cabin')  # get_dummies返回对应的0/1格式的数据，有几类返回几列，prefix指定为Cabin
    dummies_Embarked = pd.get_dummies(train_data['Embarked'], prefix='Embarked')
    dummies_Sex = pd.get_dummies(train_data['Sex'], prefix='Sex')
    dummies_Pclass = pd.get_dummies(train_data['Pclass'],prefix='Pclass')
    train_data = pd.concat([train_data,dummies_cabin,dummies_Embarked,dummies_Pclass,dummies_Sex], axis=1)  # 拼接dataframe,axis=1为列
    train_data.drop(['Pclass','Name','Sex','Embarked','Cabin','Ticket'],axis=1,inplace=True)   # 删除之前没有处理的数据列
    header_string = ','.join(train_data.columns.tolist())  # 将列名转为string，并用逗号隔开
    np.savetxt(file_path+r'/pre_processData1.csv', train_data, delimiter=',',header=header_string)  # 预处理数据保存到指定目录下    
    '''均值归一化处理(Age和Fare)'''
    scaler = StandardScaler()
    age_scaler = scaler.fit(train_data['Age'])
    train_data['Age'] = age_scaler.fit_transform(train_data['Age'])
    if np.sum(train_data.Fare.isnull()):  # 如果Fare中有为空的，就设为均值
        train_data.loc[(train_data.Fare.isnull(),'Fare')]=np.mean(train_data.Fare)
    fare_scaler = scaler.fit(train_data['Fare'])
    train_data['Fare'] = fare_scaler.transform(train_data['Fare'])
    
    header_string = ','.join(train_data.columns.tolist())  # 将列名转为string，并用逗号隔开
    np.savetxt(file_path+r'/pre_processData_scaled.csv', train_data, delimiter=',',header=header_string)  # 预处理数据保存到指定目录下    
    return train_data
    
    
# 简单显示数据
def display_data(train_data):
    plt.subplot(231)
    # plt.bar([train_data.Survived.value_counts().index],train_data.Survived.value_counts())
    train_data.Survived.value_counts().plot(kind='bar')  # 存活情况条形图，Survived里包含索引0/1
    '''kind : str
             ‘line’ : line plot (default)
             ‘bar’ : vertical bar plot
             ‘barh’ : horizontal bar plot
             ‘hist’ : histogram
             ‘box’ : boxplot
             ‘kde’ : Kernel Density Estimation plot
             ‘density’ : same as ‘kde’
             ‘area’ : area plot
             ‘pie’ : pie plot
             ‘scatter’ : scatter plot
             ‘hexbin’ : hexbin plot'''
    plt.title(u'存活情况(1为存活)',fontproperties=font)
    plt.grid()

    plt.subplot(232)
    train_data.Pclass.value_counts().plot(kind='bar')  # Pclass中包含索引1/2/3
    plt.grid()
    plt.title(u'3个等级存活情况',fontproperties=font)   

    plt.subplot(233)
    plt.scatter(train_data.Age,train_data.Survived)     # 年龄，是否存活，y坐标只有0/1
    plt.grid()
    plt.title(u'存活年龄分布',fontproperties=font)

    plt.subplot(234)
    plt.scatter(train_data.PassengerId,train_data.Age)  # 年龄的分布
    plt.grid()
    plt.title(u'年龄情况',fontproperties=font)

    plt.subplot(224)
    train_data.Age[train_data.Pclass == 1].plot(kind='kde',label=u'level 1')  # Pclass=1的年龄的密度图
    train_data.Age[train_data.Pclass == 2].plot(kind='kde',label=u'level 2')
    train_data.Age[train_data.Pclass == 3].plot(kind='kde',label=u'level 3')
    plt.grid()
    plt.xlabel(u'年龄',fontproperties=font)
    plt.ylabel(u'密度',fontproperties=font)
    plt.title(u'3个等级的年龄分布',fontproperties=font)
    plt.legend()
    plt.show()    
    
# 根据自己的理解简单处理显示一下数据
def display_with_process(train_data):
    '''显示（1）3个等级的死亡和存活的柱状对比图
           （2）死亡和存活的男女柱状对比图'''
    plt.subplot(1,2,1)
    survived_0 = train_data.Pclass[train_data.Survived == 0].value_counts().reindex([1,2,3]) # Pclass包含索引1/2/3，找到死亡的，注意重新索引一下，因为他会自动排序
    survived_1 = train_data.Pclass[train_data.Survived == 1].value_counts().reindex([1,2,3])

    index = np.array([1,2,3])
    bar_width=0.4
    plt.bar(index, survived_0,width=0.4,color='r',label=u'dead') # label对应查询的条件
    plt.bar(index+bar_width, survived_1,width=0.4,color='b',label=u'live')
    plt.xticks(index+bar_width,('Level 1','Level 2','Level 3'))
    plt.grid()
    plt.title(u'3个等级各自存活对比图',fontproperties=font)
    plt.legend(loc='best')
    
    plt.subplot(1,2,2)
    survived_male = train_data.Survived[train_data.Sex == 'male'].value_counts().reindex([0,1])  # Survived包含0/1,找到性别为male的
    survived_female = train_data.Survived[train_data.Sex == 'female'].value_counts().reindex([0,1])
    index = np.array([0,1])
    plt.bar(index, survived_male,width=0.4,color='r',label=u'male') # label对应查询的条件
    plt.bar(index+bar_width, survived_female,width=0.4,color='b',label=u'female')
    plt.grid()
    plt.xticks(index+bar_width,('dead','live'))
    plt.legend(loc='best')
    plt.show()
    '''显示（1）各个登录港口的死亡、存活柱状对比图'''
    plt.subplot(1,2,1)
    survived_0 = train_data.Embarked[train_data.Survived == 0].value_counts().reindex(['C','Q','S'])  #Embarked包含C/Q/S，
    survived_1 = train_data.Embarked[train_data.Survived == 1].value_counts().reindex(['C','Q','S'])
    index = np.array([1,2,3])
    plt.bar(index, survived_0, width=0.4,color='r',label='dead') # label对应查询的条件
    plt.bar(index+bar_width, survived_1, width=0.4,color='g',label='live')
    plt.grid()
    plt.xticks(index+bar_width,('C','Q','S'))
    plt.legend(loc='best')
    plt.title(u'登录港口存活情况',fontproperties=font)
    
    plt.subplot(1,2,2)
    survived_cabin = train_data.Survived[pd.notnull(train_data.Cabin)].value_counts().reindex([0,1])  # Survived包含所以0/1
    survived_nocabin = train_data.Survived[pd.isnull(train_data.Cabin)].value_counts().reindex([0,1])
    index = np.array([0,1])
    plt.bar(index, survived_cabin,width=0.4,color='g',label='cabin')  # label对应查询的条件
    plt.bar(index+bar_width, survived_nocabin, width=0.4,color='r',label='nocabin')
    plt.xticks(index+bar_width,('dead','live'))
    plt.grid()
    plt.title(u'有无cabin项的存活情况',fontproperties=font)
    plt.legend(loc='best')
    
    plt.show()
    
    
if __name__ == '__main__':
    solution_logisticRegression()
    #solution_svm()

