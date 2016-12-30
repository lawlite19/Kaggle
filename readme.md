Code for Kaggle competition problem
========================================

## 一、Titanic: Machine Learning from Disaster
- 问题地址：https://www.kaggle.com/c/titanic
- 全部代码：https://github.com/lawlite19/Kaggle/blob/master/Titanic/solution.py
- 只是使用了逻辑回归、和SVM两个模型

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
 - 可以看到`Age`和`Cabin`项的数据缺失严重，特别是`Cabin`


- 缺失值采用该项的**平均值**填补的。
- 得分：    
 - LogisticRegression    
 ![enter description here][1]
 - SVM    
 ![enter description here][2]


  [1]: ./images/Titanic_LogisticRegression_01.png "Titanic_LogisticRegression_01.png"
  [2]: ./images/Titanic_SVM_02.png "Titanic_SVM_02.png"