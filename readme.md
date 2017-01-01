Code for Kaggle competition problem
========================================

## 一、Titanic: Machine Learning from Disaster
- 问题地址：https://www.kaggle.com/c/titanic
- 全部代码：https://github.com/lawlite19/Kaggle/blob/master/Titanic/solution.py
- 使用了逻辑回归、和SVM两个模型，但是，观察完数据后会发现有的feature跟最后预测的结果可能关系并不是很大，所以使用线性模型进行预测个人感觉不会有太好的结果。

### 1、分析数据
- 使用pandas读取数据
 - data.info()函数查看基本的信息情况
 ```
 数据信息：
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 891 entries, 0 to 890
Data columns (total 12 columns):
PassengerId    891 non-null int64
Survived       891 non-null int64
Pclass         891 non-null int64
Name           891 non-null object
Sex            891 non-null object
Age            714 non-null float64
SibSp          891 non-null int64
Parch          891 non-null int64
Ticket         891 non-null object
Fare           891 non-null float64
Cabin          204 non-null object
Embarked       889 non-null object
dtypes: float64(2), int64(5), object(5)
memory usage: 66.2+ KB
None
 ```
 可以看到`Age`和`Cabin`项的数据缺失严重，特别是`Cabin`
 - data.describe()函数查看数据的描述
 ```
        PassengerId    Survived      Pclass         Age       SibSp  \
count   891.000000  891.000000  891.000000  714.000000  891.000000   
mean    446.000000    0.383838    2.308642   29.699118    0.523008   
std     257.353842    0.486592    0.836071   14.526497    1.102743   
min       1.000000    0.000000    1.000000    0.420000    0.000000   
25%     223.500000    0.000000    2.000000   20.125000    0.000000   
50%     446.000000    0.000000    3.000000   28.000000    0.000000   
75%     668.500000    1.000000    3.000000   38.000000    1.000000   
max     891.000000    1.000000    3.000000   80.000000    8.000000   

            Parch        Fare  
count  891.000000  891.000000  
mean     0.381594   32.204208  
std      0.806057   49.693429  
min      0.000000    0.000000  
25%      0.000000    7.910400  
50%      0.000000   14.454200  
75%      0.000000   31.000000  
max      6.000000  512.329200  
 ```
 可以看到平均年龄是`29.699118`，最大值是`80`，等等一些信息

- 简单作图显示
 - 图一：可以看到死亡的比较多
 - 图二：乘客等级为3的存活下来的比较多
 - 图三、四：乘客年龄的分布
 - 图五：不同等级年龄的密度曲线，比较集中在`20-40`之间
 ![enter description here][1]
- 根据自己的想法作图显示
 - 下图说明：三个等级中`等级一`中存活率是比较高的，`等级三`死亡率比较高
 - 性别为女的存活下来的比较多（也说明让孩子、女人先走的依据）
 ![enter description here][2]
 - 从S港口登录的死亡的人数较多
 - 从有无Cabin这个项来说（因为Cabin缺失比较严重），但是也没看出什么
 ![enter description here][3]

### 2、数据预处理
- 多个类别的就映射为多列的0/1值，如下图
![enter description here][4]
- 均值归一化`Age`和`Fare`
![enter description here][5]
- 缺失值采用该项的**平均值**填补的。
- 实现代码：
```
'''数据预处理'''  
def pre_processData(train_data,file_path):
    train_data.loc[(train_data.Age.isnull()), 'Age' ] = np.mean(train_data.Age)  # 为空的年龄补为平均年龄
    train_data.loc[(train_data.Cabin.notnull(),'Cabin')] = 'yes' # Cabin不为空的设为yes
    train_data.loc[(train_data.Cabin.isnull(),'Cabin')] = 'no'    
    
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
```
### 3、baseline model
- 逻辑回归模型
 - 实现代码：
 ```
    process_data = pre_processData(train_data,'process_train_data')  # 数据预处理，要训练的数据
    train_data = process_data.filter(regex='Survived|Age|SibSp|Parch|Fare|Cabin_.*|Embarked_.*|Sex_.*|Pclass_.*')  # 使用正则抽取想要的数据
    train_np = train_data.as_matrix()  # 转为矩阵
    '''训练model'''
    X = train_np[:,1:]
    y = train_np[:,0]
    model = linear_model.LogisticRegression(C=1.0,tol=1e-6).fit(X,y)
 ```
 - 进行预测（同时在测试集上也要预处理数据，和训练集处理方法一致）
 ```
     '''测试集上预测'''
    test_data = pd.read_csv(r"data/test.csv")
    process_test_data = pre_processData(test_data,'process_test_data')  # 预处理数据
    test_data = process_test_data.filter(regex='Age|SibSp|Parch|Fare|Cabin_.*|Embarked_.*|Sex_.*|Pclass_.*')
    test_np = test_data.as_matrix()
    predict = model.predict(test_np)
    result = pd.DataFrame(data={'PassengerId':process_test_data['PassengerId'].as_matrix(),'Survived':predict.astype(np.int32)})
    result.to_csv(r'logisticRegression_result/prediction.csv',index=False)
 ```

- SVM模型
 - `model = svm.SVC(tol=1e-6).fit(X,y)`
 - `predict = model.predict(test_np)`

### 4、baseline model提交结果
- 得分，还是可以的：    
![baseline model result][6]

### 5、优化-对于逻辑回归模型实验
- 查看各项对应的系数
 - `print pd.DataFrame({"columns":list(train_data.columns)[1:],"coef_":list(model.coef_.T)})`    
 - 对应数据
    ```
                     coef_     columns
    0    [-0.495832797784]         Age
    1    [-0.296800590035]       SibSp
    2   [-0.0914948195399]       Parch
    3    [-0.351233990575]    Cabin_no
    4     [0.620989192333]   Cabin_yes
    5     [0.224186492095]  Embarked_C
    6     [0.129390954718]  Embarked_Q
    7    [-0.287396447014]  Embarked_S
    8     [0.682623666835]    Pclass_1
    9     [0.376861367334]    Pclass_2
    10   [-0.789729832411]    Pclass_3
    11     [1.46093813976]  Sex_female
    12    [-1.19118293801]    Sex_male
    ```
 - `Age`对应系数是**负数**，呈负相关，说明年龄越小存活的机会越大
 - `Sex_female`对应系数是**正数**，呈正相关，而且值相对比较大，女性存活的机会也是比较大
 - `Sex_male`对应是**负数**，呈负相关
 - `Pclass_1`对应的系数也是**正数**，而且值相对也比较大，说明一等级的乘客存活的机会比较大
- 可以尝试组合多个feature产生新的feature训练和预测。
- 最后得分只是提高了一点点。


  [1]: ./images/Titanic_01.png "Titanic_01.png"
  [2]: ./images/Titanic_02.png "Titanic_02.png"
  [3]: ./images/Titanic_03.png "Titanic_03.png"
  [4]: ./images/Titanic_04.png "Titanic_04.png"
  [5]: ./images/Titanic_05.png "Titanic_05.png"
  [6]: ./images/Titanic_06.png "Titanic_06.png"