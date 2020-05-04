import os
import pylab
import calendar
import seaborn as sn
import missingno as msno
from datetime import datetime
import numpy as np  # linear algebra
import pandas as pd  #
from datetime import datetime
from scipy.stats import skew  # for some statistics
from scipy.special import boxcox1p
from scipy.stats import boxcox_normmax
from sklearn.linear_model import ElasticNetCV, LassoCV, RidgeCV
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import KFold, cross_val_score
from sklearn.metrics import mean_squared_error
from mlxtend.regressor import StackingCVRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
import matplotlib.pyplot as plt
import warnings
pd.options.mode.chained_assignment = None
warnings.filterwarnings("ignore", category=DeprecationWarning)
%matplotlib inline
dailyData = pd.read_csv("train.csv")
dailyDataTest = pd.read_csv("test.csv")
dailyData.drop(['row_id'], axis=1, inplace=True)
dailyDataTest.drop(['row_id'], axis=1, inplace=True)

# convert negative trip_distance to positive values
for index in dailyData.index:
    if dailyData['trip_distance'][index]<0:
        dailyData['trip_distance'][index] = -dailyData['trip_distance'][index]
# convert negative trip_distance to positive values
for index in dailyDataTest.index:
    if dailyDataTest['trip_distance'][index]<0:
        dailyDataTest['trip_distance'][index] = -dailyDataTest['trip_distance'][index]

# remove all data with trip_distance=0
dailyData = dailyData.query('trip_distance > 0.1')
# top speed for taxi 36s/mile
dailyData = dailyData.query('duration >= trip_distance*36')
# speed on foot 1200s/mile
dailyData1 = dailyData.query('duration <= trip_distance*1200')
# include some severe traffic jam and short trip distance
dailyData2 = dailyData.query('duration > trip_distance*1200 and trip_distance < 10 and duration < 7200')
dailyData = dailyData1.append(dailyData2)
dailyData.reset_index(inplace=True)
dailyData.drop('index',inplace=True,axis=1)
#remove outliers
# dailyData = dailyData.query('duration < 10000')

dailyData['duration']=np.log1p(dailyData['duration'])
y = dailyData['duration'].reset_index(drop=True) 
train_features = dailyData.drop(['duration'],axis=1)
test_features = dailyDataTest
features = pd.concat([train_features, test_features]).reset_index(drop=True)
print("剔除训练数据中的极端值后，将其特征矩阵和测试数据中的特征矩阵合并，维度为:",features.shape)
# fill nan with 2019-03-31 00:00:00 in datetime
dailyData = dailyData.rename(columns={'pickup_datetime':'datetime'})
dailyData['datetime'].fillna("2019-03-31 00:00:00",inplace=True)
dailyData[["datetime"]] = dailyData[["datetime"]].astype(str)
dailyData['hour'] = dailyData.datetime.apply(lambda x : x.split()[1].split(":")[0])
dailyData["weekday"] = dailyData.datetime.apply(lambda dateString : calendar.day_name[datetime.strptime(dateString.split()[0],"%Y-%m-%d").weekday()])

# calculate the duration mean of hour-weekday pair
hour_weekday_mean = dailyData.groupby(["hour","weekday"],sort=True)["duration"].mean()

# calculate duration mean in terms of hour
hour_mean = dailyData.groupby("hour").mean()['duration']

# fill nan with 2019-03-31 00:00:00 in datetime
features = features.rename(columns={'pickup_datetime':'datetime'})
features['datetime'].fillna("2019-03-31 00:00:00",inplace=True)
features[["datetime"]] = features[["datetime"]].astype(str)

# Remove the nan from dropoff_zone and pickup_zone
features['dropoff_zone'].fillna("NV",inplace=True)
features['pickup_zone'].fillna("NV",inplace=True)

# Fill nan with specific number
features['passenger_count'].fillna(1,inplace=True)
features['VendorID'].fillna(2,inplace=True)

# get weekday
features['year'] = features.datetime.apply(lambda x : x.split()[0].split("-")[0])
features['month'] = features.datetime.apply(lambda x : x.split()[0].split("-")[1])
features['day'] = features.datetime.apply(lambda x : x.split()[0].split("-")[2])
features['hour'] = features.datetime.apply(lambda x : x.split()[1].split(":")[0])
features["weekday"] = features.datetime.apply(lambda dateString : calendar.day_name[datetime.strptime(dateString.split()[0],"%Y-%m-%d").weekday()])

# combine pickup_zone & dropoff_zone, hour & weekkday
# features['zone'] = features.apply(lambda x: str(x['pickup_zone'])+"-"+str(x['dropoff_zone']),axis=1)
features['hour_weekday'] = features.apply(lambda x: str(x['hour'])+"-"+str(x['weekday']),axis=1)
features['borough_path'] = features.apply(lambda x: str(x['pickup_borough'])+"-"+str(x['dropoff_borough']),axis=1)

# calculate the duration mean of hour-weekday pair
weekday_order = {"Friday":0,"Monday":1,"Saturday":2,"Sunday":3,"Thursday":4,"Tuesday":5,"Wednesday":6}
features['hour_weekday_mean'] = features.apply(lambda x: hour_weekday_mean[7*int(x["hour"])+weekday_order[x["weekday"]]],axis=1)

# calculate duration mean in terms of hour
features['hour_mean'] = features.apply(lambda x: hour_mean[int(x['hour'])],axis=1)

# calculate log1p(trip_distance)
features['trip_distance_log'] = features.apply(lambda x: np.log1p(x['trip_distance']),axis=1)

features['same_borough'] = features.apply(lambda x: 1 if x['pickup_borough']==x['dropoff_borough']  else 0,axis=1)

# convert string to hash int
def convert_obj_to_int(self):
    object_list_columns = self.columns
    object_list_dtypes = self.dtypes
    new_col_suffix = '_int'
    for column in self.columns:
        if column in ["pickup_zone","dropoff_zone","pickup_borough","dropoff_borough","weekday","borough_path","hour_weekday"] :
            self[column+new_col_suffix] = self[column].map( lambda  x: hash(x) % 1000000)
            self.drop([column],inplace=True,axis=1)
    return self
features = convert_obj_to_int(features)
features = features.drop(['datetime','VendorID','year',], axis=1)
# categorical features
categoricalFeatureNames = ["pickup_borough_int","dropoff_borough_int","month","day","hour","weekday_int","same_borough","passenger_count"]

for var in categoricalFeatureNames:
    features[var] = features[var].astype("category")

######################数字型数据列偏度校正-【开始】#######################
#使用skew()方法，计算所有整型和浮点型数据列中，数据分布的偏度（skewness）。
#偏度是统计数据分布偏斜方向和程度的度量，是统计数据分布非对称程度的数字特征。亦称偏态、偏态系数。
numeric_dtypes = ['float64']
numerics2 = []
for i in features.columns:
    if features[i].dtype in numeric_dtypes:
        numerics2.append(i)
skew_features = features[numerics2].apply(lambda x: skew(x)).sort_values(ascending=False)

#以0.5作为基准，统计偏度超过此数值的高偏度分布数据列，获取这些数据列的index。
high_skew = skew_features[skew_features > 0.5]
skew_index = high_skew.index

#对高偏度数据进行处理，将其转化为正态分布。
#Box和Cox提出的变换可以使线性回归模型满足线性性、独立性、方差齐次以及正态性的同时，又不丢失信息。
for i in skew_index:
    features[i] = boxcox1p(features[i], boxcox_normmax(features[i] + 1))#这是boxcox1p的使用方法，参数的具体意义暂时不解释
######################数字型数据列偏度校正-【结束】#######################

X = features.iloc[:len(y), :]	#y是列向量，存储了训练数据中的房价列信息。截取后得到的X阵的维度是len(y)*(final_features的列数)。
X_sub = features.iloc[len(y):, :]#使用len命令，求矩阵X的长度，得到的是矩阵对象的长度，即有矩阵中有多少列，而不是每列上有多少行。
outliers = [273599, 274032, 275431]
X = X.drop(X.index[outliers])#因为X阵是经过对特征矩阵进行类似“坐标投影”操作后得到的，列向量y中的行号对应着X阵中的列号。
y = y.drop(y.index[outliers])

overfit = []#用来记录产生过拟合的数据列的序号
for i in X.columns:#遍历截取后特征矩阵的每一列
    counts = X[i].value_counts()#使用.value_counts()方法，查看在X矩阵的第i列中，不同的取值分别出现了多少次，默认按次数最高到最低做降序排列。返回一个df。
    zeros = counts.iloc[0]#通过行号索引行数据，取出counts列中第一个元素，即出现次数最多的取值到底是出现了多少次，存入zeros
    if zeros / len(X) * 100 > 99.94:
#判断某一列是否将产生过拟合的条件：
#截取后的特征矩阵有len(X)列，如果某一列中的某个值出现的次数除以特征矩阵的列数超过99.94%，即其几乎在被投影的各个维度上都有着同样的取值，并不具有“主成分”的性质，则记为过拟合列。
        overfit.append(i)
overfit = list(overfit)
#overfit.append('MSZoning_C (all)')#这条语句有用吗？是要把训练数据特征矩阵X中的列标签为'MSZoning_C (all)'的列也删除吗？但是训练数据中并没有任何一个列标签名称为MSZoning_C (all)。
X = X.drop(overfit, axis=1)#.copy()#删除截取后特征矩阵X中的过拟合列。因为drop并不影响原数据，所以使用copy。直接覆值应该也可以。
X_sub = X_sub.drop(overfit, axis=1)#.copy()

# split train and val
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)
def rmsle(y, y_,convertExp=True):
    if convertExp:
        y = np.exp(y),
        y_ = np.exp(y_)
    log1 = np.nan_to_num(np.array([np.log(v + 1) for v in y]))
    log2 = np.nan_to_num(np.array([np.log(v + 1) for v in y_]))
    calc = (log1 - log2) ** 2
    return np.sqrt(np.mean(calc))
def rmse(y, y_, convertExp=True):
    calc = (y - y_)**2
    return np.sqrt(np.mean(calc))
# #打印lightgbm轻梯度提升模型的得分
from sklearn.ensemble import GradientBoostingRegressor
#定义GB梯度提升模型（展开到一阶导数）									
gbr = GradientBoostingRegressor(n_estimators=3000, learning_rate=0.05, max_depth=4, min_samples_leaf=15, min_samples_split=10, loss='huber', random_state =42)                             
gbr.fit(x_train,y_train)
y_pred = gbr.predict(X=x_test)
print(rmse(np.exp(y_test),np.exp(y_pred)))

gbr = GradientBoostingRegressor(n_estimators=3000, learning_rate=0.05, max_depth=4, max_features='sqrt', min_samples_leaf=15, min_samples_split=10, loss='huber', random_state =42)                             
gbr.fit(X,y)
submission = pd.read_csv("submission.csv")
#函数注释：.iloc[:,1]是基于索引位来选取数据集，[索引1:索引2]，左闭右开。
submission.iloc[:,1] = np.floor(np.expm1(gbr.predict(X_sub)))
########将测试集的特征矩阵作为输入，传入训练好的模型，得出的输出写入.csv文件的第2列-【结束】########
q1 = submission['duration'].quantile(0.005)
q2 = submission['duration'].quantile(0.995)
submission['duration'] = submission['duration'].apply(lambda x: x if x > q1 else x*0.77)
submission['duration'] = submission['duration'].apply(lambda x: x if x < q2 else x*1.1)
submission.duration=submission.duration.astype(int)
submission.head(2)

submission.to_csv("submission1.csv", index=False)