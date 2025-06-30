import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from lightgbm.sklearn import LGBMRegressor
from sklearn.metrics import mean_absolute_error
from tqdm import tqdm
from lightgbm.callback import early_stopping
from datetime import datetime
import os

# 自定义mad函数
def mad(x):
    return np.median(np.abs(x - np.median(x)))

# 1. 数据加载
print('开始加载数据...')
Train_data = pd.read_csv(r'D:\AI_Learning\python\05_compitition\tianchi_competition\project_426\data\used_car_train_20200313.csv', sep=' ')
Test_data = pd.read_csv(r'D:\AI_Learning\python\05_compitition\tianchi_competition\project_426\data\used_car_testB_20200421.csv', sep=' ')

# 2. 数据预处理
print('开始数据预处理...')
# 合并数据
df = pd.concat([Train_data, Test_data], ignore_index=True)

# 目标值处理
df['price'] = np.log1p(df['price'])

# 处理异常数据
df.drop(df[df['seller'] == 1].index, inplace=True)
del df['offerType']     
del df['seller']

# 处理缺失值
df['fuelType'] = df['fuelType'].fillna(0)
df['gearbox'] = df['gearbox'].fillna(0)
df['bodyType'] = df['bodyType'].fillna(0)
df['model'] = df['model'].fillna(0)

# 处理异常值
df['power'] = df['power'].map(lambda x: 600 if x > 600 else x)
df['notRepairedDamage'] = df['notRepairedDamage'].astype('str').apply(lambda x: x if x != '-' else None).astype('float32')

# 3. 特征工程
print('开始特征工程...')
# 3.1 基础特征处理
# 对name进行挖掘
df['name_count'] = df.groupby(['name'])['SaleID'].transform('count')
del df['name']

# 地区特征
df['regionCode_count'] = df.groupby(['regionCode'])['SaleID'].transform('count')
df['city'] = df['regionCode'].apply(lambda x : str(x)[:2])

# 3.2 时间特征处理
def date_process(x):
    if pd.isna(x):  # 处理NaN值
        return pd.NaT  # 返回NaT (Not a Time)
    
    try:
        year = int(str(x)[:4])
        month = int(str(x)[4:6])
        day = int(str(x)[6:8])

        if month < 1:
            month = 1

        date = datetime(year, month, day)
        return date
    except (ValueError, TypeError):
        return pd.NaT  # 如果转换失败，返回NaT

df['regDates'] = df['regDate'].apply(date_process)
df['creatDates'] = df['creatDate'].apply(date_process)
df['regDate_year'] = df['regDates'].dt.year
df['regDate_month'] = df['regDates'].dt.month
df['regDate_day'] = df['regDates'].dt.day
df['creatDate_year'] = df['creatDates'].dt.year
df['creatDate_month'] = df['creatDates'].dt.month
df['creatDate_day'] = df['creatDates'].dt.day
df['car_age_day'] = (df['creatDates'] - df['regDates']).dt.days
df['car_age_year'] = round(df['car_age_day'] / 365, 1)

# 3.3 分桶特征
bin = [i*10 for i in range(31)]
df['power_bin'] = pd.cut(df['power'], bin, labels=False)

bin = [i*10 for i in range(24)]
df['model_bin'] = pd.cut(df['model'], bin, labels=False)

# 3.4 统计特征生成
print('生成统计特征...')
for col in ['regionCode', 'brand', 'model', 'kilometer', 'bodyType', 'fuelType']:
    Train_gb = Train_data.groupby(col)
    all_info = {}
    for kind, kind_data in Train_gb:
        info = {}
        kind_data = kind_data[kind_data['price'] > 0]
        info[f'{col}_amount'] = len(kind_data)
        info[f'{col}_price_max'] = kind_data.price.max()
        info[f'{col}_price_median'] = kind_data.price.median()
        info[f'{col}_price_min'] = kind_data.price.min()
        info[f'{col}_price_sum'] = kind_data.price.sum()
        info[f'{col}_price_std'] = kind_data.price.std()
        info[f'{col}_price_mean'] = kind_data.price.mean()
        info[f'{col}_price_skew'] = kind_data.price.skew()
        info[f'{col}_price_kurt'] = kind_data.price.kurt()
        info[f'{col}_price_mad'] = mad(kind_data.price)
        all_info[kind] = info
    brand_fe = pd.DataFrame(all_info).T.reset_index().rename(columns={"index": col})
    df = df.merge(brand_fe, how='left', on=col)

# 3.5 regionCode相关统计特征
print('生成regionCode相关统计特征...')
kk = "regionCode"
# 按car_age_day统计
Train_gb = df.groupby(kk)
all_info = {}
for kind, kind_data in Train_gb:
    info = {}
    kind_data = kind_data[kind_data['car_age_day'] > 0]
    info[f'{kk}_days_max'] = kind_data.car_age_day.max()
    info[f'{kk}_days_min'] = kind_data.car_age_day.min()
    info[f'{kk}_days_std'] = kind_data.car_age_day.std()
    info[f'{kk}_days_mean'] = kind_data.car_age_day.mean()
    info[f'{kk}_days_median'] = kind_data.car_age_day.median()
    info[f'{kk}_days_sum'] = kind_data.car_age_day.sum()
    all_info[kind] = info
brand_fe = pd.DataFrame(all_info).T.reset_index().rename(columns={"index": kk})
df = df.merge(brand_fe, how='left', on=kk)

# 按power统计
Train_gb = df.groupby(kk)
all_info = {}
for kind, kind_data in Train_gb:
    info = {}
    kind_data = kind_data[kind_data['power'] > 0]
    info[f'{kk}_power_max'] = kind_data.power.max()
    info[f'{kk}_power_min'] = kind_data.power.min()
    info[f'{kk}_power_std'] = kind_data.power.std()
    info[f'{kk}_power_mean'] = kind_data.power.mean()
    info[f'{kk}_power_median'] = kind_data.power.median()
    info[f'{kk}_power_sum'] = kind_data.power.sum()
    all_info[kind] = info
brand_fe = pd.DataFrame(all_info).T.reset_index().rename(columns={"index": kk})
df = df.merge(brand_fe, how='left', on=kk)

# 3.6 特征组合
print('生成特征组合...')
for i in range(15):
    for j in range(15):
        df[f'new{i}*{j}'] = df[f'v_{i}'] * df[f'v_{j}']

for i in range(15):
    for j in range(15):
        df[f'new{i}+{j}'] = df[f'v_{i}'] + df[f'v_{j}']

for i in range(15):
    df[f'new{i}*power'] = df[f'v_{i}'] * df['power']

for i in range(15):
    df[f'new{i}*day'] = df[f'v_{i}'] * df['car_age_day']

for i in range(15):
    df[f'new{i}*year'] = df[f'v_{i}'] * df['car_age_year']

for i in range(15):
    for j in range(15):
        df[f'new{i}-{j}'] = df[f'v_{i}'] - df[f'v_{j}']

# 4. 特征选择
print('开始特征选择...')
numerical_cols = df.select_dtypes(exclude='object').columns
list_tree = ['model_power_sum', 'model_power_std', 'model_power_median', 'model_power_max',
             'brand_price_max', 'brand_price_median', 'brand_price_sum', 'brand_price_std',
             'model_days_sum', 'model_days_std', 'model_days_median', 'model_days_max', 
             'model_bin', 'model_amount', 'model_price_max', 'model_price_median',
             'model_price_min', 'model_price_sum', 'model_price_std', 'model_price_mean',
             'bodyType', 'model', 'brand', 'fuelType', 'gearbox', 'power', 'kilometer',
             'notRepairedDamage', 'v_0', 'v_1', 'v_2', 'v_3', 'v_4', 'v_5', 'v_6', 'v_7', 
             'v_8', 'v_9', 'v_10', 'v_11', 'v_12', 'v_13', 'v_14', 'name_count', 
             'regDate_year', 'car_age_day', 'car_age_year', 'power_bin', 'price', 'SaleID', 'regionCode',
             'regDate', 'creatDate']

# 添加所有加法组合特征
for i in range(15):
    for j in range(15):
        list_tree.append(f'new{i}+{j}')

feature_cols = [col for col in numerical_cols if col in list_tree]

# 移除不需要的特征
exclude_features = ['new14+6', 'new13+6', 'new0+12', 'new9+11', 'v_3', 'new11+10', 'new10+14', 
                   'new12+4', 'new3+4', 'new11+11', 'new13+3', 'new8+1', 'new1+7', 'new11+14', 
                   'new8+13', 'v_8', 'v_0', 'new3+5', 'new2+9', 'new9+2', 'new0+11', 'new13+7', 
                   'new8+11', 'new5+12', 'new10+10', 'new13+8', 'new11+13', 'new7+9', 'v_1', 
                   'new7+4', 'new13+4', 'v_7', 'new5+6', 'new7+3', 'new9+10', 'new11+12', 
                   'new0+5', 'new4+13', 'new8+0', 'new0+7', 'new12+8', 'new10+8', 'new13+14', 
                   'new5+7', 'new2+7', 'v_4', 'v_10', 'new4+8', 'new8+14', 'new5+9', 'new9+13', 
                   'new2+12', 'new5+8', 'new3+12', 'new0+10', 'new9+0', 'new1+11', 'new8+4', 
                   'new11+8', 'new1+1', 'new10+5', 'new8+2', 'new6+1', 'new2+1', 'new1+12', 
                   'new2+5', 'new0+14', 'new4+7', 'new14+9', 'new0+2', 'new4+1', 'new7+11', 
                   'new13+10', 'new6+3', 'new1+10', 'v_9', 'new3+6', 'new12+1', 'new9+3', 
                   'new4+5', 'new12+9', 'new3+8', 'new0+8', 'new1+8', 'new1+6', 'new10+9', 
                   'new5+4', 'new13+1', 'new3+7', 'new6+4', 'new6+7', 'new13+0', 'new1+14', 
                   'new3+11', 'new6+8', 'new0+9', 'new2+14', 'new6+2', 'new12+12', 'new7+12', 
                   'new12+6', 'new12+14', 'new4+10', 'new2+4', 'new6+0', 'new3+9', 'new2+8', 
                   'new6+11', 'new3+10', 'new7+0', 'v_11', 'new1+3', 'new8+3', 'new12+13', 
                   'new1+9', 'new10+13', 'new5+10', 'new2+2', 'new6+9', 'new7+10', 'new0+0', 
                   'new11+7', 'new2+13', 'new11+1', 'new5+11', 'new4+6', 'new12+2', 'new4+4', 
                   'new6+14', 'new0+1', 'new4+14', 'v_5', 'new4+11', 'v_6', 'new0+4', 'new1+5', 
                   'new3+14', 'new2+10', 'new9+4', 'new2+6', 'new14+14', 'new11+6', 'new9+1', 
                   'new3+13', 'new13+13', 'new10+6', 'new2+3', 'new2+11', 'new1+4', 'v_2', 
                   'new5+13', 'new4+2', 'new0+6', 'new7+13', 'new8+9', 'new9+12', 'new0+13', 
                   'new10+12', 'new5+14', 'new6+10', 'new10+7', 'v_13', 'new5+2', 'new6+13', 
                   'new9+14', 'new13+9', 'new14+7', 'new8+12', 'new3+3', 'new6+12', 'v_12', 
                   'new14+4', 'new11+9', 'new12+7', 'new4+9', 'new4+12', 'new1+13', 'new0+3', 
                   'new8+10', 'new13+11', 'new7+8', 'new7+14', 'v_14', 'new10+11', 'new14+8', 
                   'new1+2']

feature_cols = [col for col in feature_cols if col not in exclude_features]
df = df[feature_cols]

# 5. 模型训练
print('开始模型训练...')
df1 = df.copy()
test = df1[df1['price'].isnull()]
X_train = df1[df1['price'].notnull()].drop(['price', 'regDate', 'creatDate', 'SaleID', 'regionCode'], axis=1)
Y_train = df1[df1['price'].notnull()]['price']
X_test = df1[df1['price'].isnull()].drop(['price', 'regDate', 'creatDate', 'SaleID', 'regionCode'], axis=1)

# 初始化
cols = list(X_train)
oof = np.zeros(X_train.shape[0])
sub = test[['SaleID']].copy()
sub['price'] = 0
feat_df = pd.DataFrame({'feat': cols, 'imp': 0})

# 设置交叉验证
skf = KFold(n_splits=100, shuffle=True, random_state=2025)

# 设置模型参数
clf = LGBMRegressor(
    n_estimators=500000,
    learning_rate=0.02,
    boosting_type='gbdt',
    objective='regression_l1',
    max_depth=-1,
    num_leaves=31,
    min_child_samples=20,
    feature_fraction=0.8,
    bagging_freq=1,
    bagging_fraction=0.8,
    lambda_l2=2,
    random_state=2025,
    metric='mae'
)

# 训练模型
mae = 0
for i, (trn_idx, val_idx) in enumerate(tqdm(list(skf.split(X_train, Y_train)), desc="交叉验证进度")):
    print('--------------------- 第 {} 折 ---------------------'.format(i + 1))
    trn_x, trn_y = X_train.iloc[trn_idx].reset_index(drop=True), Y_train[trn_idx]
    val_x, val_y = X_train.iloc[val_idx].reset_index(drop=True), Y_train[val_idx]
    
    clf.fit(
        trn_x, trn_y,
        eval_set=[(val_x, val_y)],
        eval_metric='mae',
        callbacks=[early_stopping(300)],
    )

    sub['price'] += np.expm1(clf.predict(X_test)) / skf.n_splits 
    oof[val_idx] = clf.predict(val_x)
    mae += mean_absolute_error(np.expm1(val_y), np.expm1(oof[val_idx])) / skf.n_splits

print('交叉验证 MAE:', mae)

# 6. 生成提交文件
print('生成提交文件...')
sub.to_csv(r'D:\AI_Learning\python\05_compitition\tianchi_competition\project_426\submit_lgb_repair2.csv', index=False)
print('完成！')