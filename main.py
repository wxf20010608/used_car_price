import warnings
warnings.filterwarnings("ignore")
import os
import pandas as pd
import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.callbacks import LearningRateScheduler
import lightgbm as lgb
from catboost import CatBoostRegressor
from sklearn.model_selection import KFold, RepeatedKFold
from sklearn.metrics import mean_absolute_error
from sklearn import linear_model
from sklearn.preprocessing import StandardScaler

## 读取树模型数据
base_path = os.path.dirname(os.path.abspath(__file__))
tree_data_path = os.path.join(base_path, "user_data")
data_path = os.path.join(base_path, "data") # Re-defining data_path here

if not os.path.exists(tree_data_path):
    os.makedirs(tree_data_path)
    print(f"Created directory: {tree_data_path}")

train_file = os.path.join(data_path, "used_car_train_20200313.csv")
test_file = os.path.join(data_path, "used_car_testB_20200421.csv")

try:
    Train_data = pd.read_csv(train_file, sep=" ")
    TestA_data = pd.read_csv(test_file, sep=" ")
    print("Successfully loaded original data files")
    print("Train_data columns after initial load:", Train_data.columns.tolist())
    print("TestA_data columns after initial load:", TestA_data.columns.tolist())
except FileNotFoundError:
    print(f"Error: Could not find the required data files.")
    print(f"Please ensure the following files exist:")
    print(f"  - {train_file}")
    print(f"  - {test_file}")
    exit(1)

# Combine train and test data for consistent preprocessing
combined_data = pd.concat([Train_data.drop('price', axis=1), TestA_data], ignore_index=True)

# Handle 'notRepairedDamage' first, as it's a mix of numeric and string
if 'notRepairedDamage' in combined_data.columns:
    combined_data['notRepairedDamage'] = combined_data['notRepairedDamage'].astype('str').replace('-', np.nan).astype('float32')

# 将日期列转换为 datetime 对象以便提取更多特征
combined_data['regDate'] = pd.to_datetime(combined_data['regDate'].astype(str), errors='coerce')
combined_data['creatDate'] = pd.to_datetime(combined_data['creatDate'].astype(str), errors='coerce')

# Re-split data after combined preprocessing (and before general fillna)
Train_data_processed = combined_data.iloc[:len(Train_data)].copy()
TestA_data_processed = combined_data.iloc[len(Train_data):].copy()
Train_data_processed['price'] = Train_data['price'] # Add price back to training data

print("Train_data_processed columns after initial processing:", Train_data_processed.columns.tolist())
print("TestA_data_processed columns after initial processing:", TestA_data_processed.columns.tolist())
print("Train_data_processed['regDate'] dtype:", Train_data_processed['regDate'].dtype)
print("TestA_data_processed['regDate'] dtype:", TestA_data_processed['regDate'].dtype)
print("Train_data_processed['creatDate'] dtype:", Train_data_processed['creatDate'].dtype)
print("TestA_data_processed['creatDate'] dtype:", TestA_data_processed['creatDate'].dtype)

# --- Feature Engineering and Preprocessing Start ---
# 提取日期特征
Train_data_processed['regDate_month'] = Train_data_processed['regDate'].dt.month
Train_data_processed['regDate_dayofweek'] = Train_data_processed['regDate'].dt.dayofweek
TestA_data_processed['regDate_month'] = TestA_data_processed['regDate'].dt.month
TestA_data_processed['regDate_dayofweek'] = TestA_data_processed['regDate'].dt.dayofweek

Train_data_processed['creatDate_month'] = Train_data_processed['creatDate'].dt.month
Train_data_processed['creatDate_dayofweek'] = Train_data_processed['creatDate'].dt.dayofweek
TestA_data_processed['creatDate_month'] = TestA_data_processed['creatDate'].dt.month
TestA_data_processed['creatDate_dayofweek'] = TestA_data_processed['creatDate'].dt.dayofweek

# 更多时间差异特征
Train_data_processed['age_on_sale_days'] = (Train_data_processed['creatDate'] - Train_data_processed['regDate']).dt.days
TestA_data_processed['age_on_sale_days'] = (TestA_data_processed['creatDate'] - TestA_data_processed['regDate']).dt.days

# Identify categorical columns after date feature engineering
categorical_cols = ['model', 'brand', 'bodyType', 'fuelType', 'gearbox', 'notRepairedDamage']

# 首先处理类别特征
for col in categorical_cols:
    if col in combined_data.columns:
        # 获取所有唯一的类别值
        unique_categories = combined_data[col].unique()
        # 创建类别映射字典
        category_map = {cat: i for i, cat in enumerate(unique_categories)}
        # 应用映射到训练和测试数据
        Train_data_processed[col] = Train_data_processed[col].map(category_map).fillna(-1).astype('int32')
        TestA_data_processed[col] = TestA_data_processed[col].map(category_map).fillna(-1).astype('int32')

# 确保所有数值特征都是 float32 类型
numeric_cols = Train_data_processed.select_dtypes(include=['float64', 'int64']).columns
numeric_cols = [col for col in numeric_cols if col != 'price']  # Exclude price column
for col in numeric_cols:
    Train_data_processed[col] = Train_data_processed[col].astype('float32')
    TestA_data_processed[col] = TestA_data_processed[col].astype('float32')

# Fill all remaining NaN values with 0 after all feature engineering
Train_data_processed = Train_data_processed.fillna(0)
TestA_data_processed = TestA_data_processed.fillna(0)

# 聚合特征增强
# 针对 'brand' 和 'model' 更多的统计量
for col in ['brand', 'model']:
    for stat in ['mean', 'std', 'max', 'min', 'median']:
        for feature in ['power', 'kilometer']:
            group_col_name = f'{col}_{feature}_{stat}'
            temp_df = Train_data_processed.groupby([col])[feature].agg([stat]).reset_index()
            temp_df.columns = [col, group_col_name]
            Train_data_processed = pd.merge(Train_data_processed, temp_df, on=col, how='left')
            TestA_data_processed = pd.merge(TestA_data_processed, temp_df, on=col, how='left')

# 简单的交互特征
Train_data_processed['power_per_kilometer'] = Train_data_processed['power'] / (Train_data_processed['kilometer'] + 1e-6) # 避免除以零
TestA_data_processed['power_per_kilometer'] = TestA_data_processed['power'] / (TestA_data_processed['kilometer'] + 1e-6)

# --- Feature Engineering and Preprocessing End ---

# Now, explicitly check if 'v_' columns exist and uncomment if they do
v_cols_exist = any(col.startswith('v_') for col in combined_data.columns) # Use original combined_data to check for v_cols

if v_cols_exist:
    # Define num_cols based on existing v_ columns
    v_columns = [col for col in combined_data.columns if col.startswith('v_')]
    num_cols_indices = [int(col.split('_')[1]) for col in v_columns if col.startswith('v_') and col.split('_')[1].isdigit()]
    num_cols_indices = sorted(list(set(num_cols_indices)))

    print(f"Detected v_ columns: {v_columns}")
    print(f"Corresponding v_ indices: {num_cols_indices}")

    for i in num_cols_indices:
        for j in num_cols_indices:
            if f'v_{i}' in Train_data_processed.columns and f'v_{j}' in Train_data_processed.columns:
                Train_data_processed[f'new{i}*{j}'] = Train_data_processed[f'v_{i}'] * Train_data_processed[f'v_{j}']
                TestA_data_processed[f'new{i}*{j}'] = TestA_data_processed[f'v_{i}'] * TestA_data_processed[f'v_{j}']

                Train_data_processed[f'new{i}+{j}'] = Train_data_processed[f'v_{i}'] + Train_data_processed[f'v_{j}']
                TestA_data_processed[f'new{i}+{j}'] = TestA_data_processed[f'v_{i}'] + TestA_data_processed[f'v_{j}']

                Train_data_processed[f'new{i}-{j}'] = Train_data_processed[f'v_{i}'] - Train_data_processed[f'v_{j}']
                TestA_data_processed[f'new{i}-{j}'] = TestA_data_processed[f'v_{i}'] - TestA_data_processed[f'v_{j}']

    if 'car_age_year' in Train_data_processed.columns:
        for i in num_cols_indices:
            if f'v_{i}' in Train_data_processed.columns:
                Train_data_processed[f'new{i}*year'] = Train_data_processed[f'v_{i}'] * Train_data_processed['car_age_year']
                TestA_data_processed[f'new{i}*year'] = TestA_data_processed[f'v_{i}'] * TestA_data_processed['car_age_year']

# 重新定义 feature_cols，排除 'price' 和 'SaleID'，并包含新生成的特征
feature_cols = [col for col in Train_data_processed.columns if col not in ["price", "SaleID", 'regDate', 'creatDate']]

# 确保训练和测试数据使用相同的列
Train_data_processed = Train_data_processed[feature_cols + ['price', 'SaleID']]
TestA_data_processed = TestA_data_processed[feature_cols + ['SaleID']]

# 特征缩放
scaler = StandardScaler()
numeric_feature_cols = Train_data_processed[feature_cols].select_dtypes(include=['float32']).columns.tolist()
Train_data_processed[numeric_feature_cols] = scaler.fit_transform(Train_data_processed[numeric_feature_cols])
TestA_data_processed[numeric_feature_cols] = scaler.transform(TestA_data_processed[numeric_feature_cols])

# Update X_data and X_test for tree models after scaling
X_data_tree = Train_data_processed[feature_cols]
X_test_tree = TestA_data_processed[feature_cols]

# 确保数据类型一致
X_data_tree = X_data_tree.astype('float32')
X_test_tree = X_test_tree.astype('float32')

# 将类别特征转换为整数类型
for col in categorical_cols:
    if col in X_data_tree.columns:
        X_data_tree[col] = X_data_tree[col].astype('int32')
        X_test_tree[col] = X_test_tree[col].astype('int32')

print("X_data_tree dtypes:", X_data_tree.dtypes)
print("X_test_tree dtypes:", X_test_tree.dtypes)
print("X_data_tree shape:", X_data_tree.shape)
print("X_test_tree shape:", X_test_tree.shape)

# Extract target variable for training
Y_data = Train_data_processed['price'].values

"""
lightgbm
"""

# 自定义损失函数
def myFeval(preds, xgbtrain):
    label = xgbtrain.get_label()
    # Clip values to avoid infinity
    max_value = 1e10  # Set a reasonable maximum value
    preds_exp = np.clip(np.expm1(preds), -max_value, max_value)
    label_exp = np.clip(np.expm1(label), -max_value, max_value)
    score = mean_absolute_error(label_exp, preds_exp)
    return "myFeval", score, False

param = {
    "boosting_type": "gbdt",         # 使用 GBDT（梯度提升树）作为提升方法，是 LightGBM 默认的方式
    "num_leaves": 31,                # 每棵树的最大叶子节点数，值越大模型越复杂，越容易过拟合
    "max_depth": -1,                 # 不限制树的最大深度，通常与 num_leaves 联合控制复杂度
    "lambda_l2": 2,                  # L2 正则化系数，用于防止模型过拟合
    "min_data_in_leaf": 20,          # 每个叶子节点最少的数据量，防止分裂出只有少量样本的叶子，减少过拟合风险
    "objective": "regression_l1",    # 目标函数为 L1 回归（最小绝对误差 MAE），对异常值更稳健
    "learning_rate": 0.02,           # 学习率，值小代表每棵树学习得更慢，训练更稳定但耗时更长
    "min_child_samples": 20,         # 与 min_data_in_leaf 类似，叶子节点所需的最小样本数，防止过拟合
    "feature_fraction": 0.8,         # 每棵树训练时，随机使用 80% 的特征，有助于减少特征间的共线性和过拟合
    "bagging_freq": 1,               # 每 1 次迭代执行一次数据采样（bagging）
    "bagging_fraction": 0.8,         # 每次训练时只用 80% 的训练数据进行采样，防止模型过拟合
    "bagging_seed": 11,              # 控制 bagging 的随机性，使得结果可复现
    "metric": "mae",                 # 使用 MAE（平均绝对误差）作为验证集评估指标，衡量预测误差
    "n_estimators": 10000,           # 最大迭代次数
    "verbose": -1                    # 不显示训练过程
}

folds = KFold(n_splits=2, shuffle=True)
oof_lgb = np.zeros(len(X_data_tree))
predictions_lgb = np.zeros(len(X_test_tree))
predictions_train_lgb = np.zeros(len(X_data_tree))

for fold_, (trn_idx, val_idx) in enumerate(folds.split(X_data_tree, Y_data)):
    print("fold n°{}".format(fold_ + 1))
    
    # 使用 scikit-learn API
    model = lgb.LGBMRegressor(**param)
    
    # 训练模型
    model.fit(
        X_data_tree.iloc[trn_idx],
        Y_data[trn_idx],
        eval_set=[(X_data_tree.iloc[val_idx], Y_data[val_idx])],
        eval_metric='mae',
        callbacks=[lgb.early_stopping(stopping_rounds=300)]
    )
    
    # 预测
    oof_lgb[val_idx] = model.predict(X_data_tree.iloc[val_idx])
    predictions_lgb += model.predict(X_test_tree) / folds.n_splits
    predictions_train_lgb += model.predict(X_data_tree) / folds.n_splits

# 修改LightGBM评分部分
print(
    "lightgbm score: {:<8.8f}".format(
        mean_absolute_error(
            np.clip(np.expm1(oof_lgb), 0, 1e10),  # 限制最大值为1e10
            np.clip(np.expm1(Y_data), 0, 1e10)    # 限制最大值为1e10
        )
    )
)

# 保存预测结果
output_path = os.path.join(base_path, "user_data")
# 测试集输出
predictions = np.clip(np.expm1(predictions_lgb), 0, 1e10)
predictions[predictions < 0] = 0
sub = pd.DataFrame()
sub["SaleID"] = TestA_data_processed.SaleID
sub["price"] = predictions
sub.to_csv(os.path.join(output_path, "test_lgb.csv"), index=False)

# 验证集输出
oof_lgb_clipped = np.clip(np.expm1(oof_lgb), 0, 1e10)
oof_lgb_clipped[oof_lgb_clipped < 0] = 0
sub = pd.DataFrame()
sub["SaleID"] = Train_data_processed.SaleID
sub["price"] = oof_lgb_clipped
sub.to_csv(os.path.join(output_path, "train_lgb.csv"), index=False)

"""
catboost
"""

kfolder = KFold(n_splits=2, shuffle=True)
oof_cb = np.zeros(len(X_data_tree))
predictions_cb = np.zeros(len(X_test_tree))
predictions_train_cb = np.zeros(len(X_data_tree))
kfold = kfolder.split(X_data_tree, Y_data)
fold_ = 0
for train_index, vali_index in kfold:
    fold_ = fold_ + 1
    print("fold n°{}".format(fold_))
    k_x_train = X_data_tree[train_index]
    k_y_train = Y_data[train_index]
    k_x_vali = X_data_tree[vali_index]
    k_y_vali = Y_data[vali_index]
    cb_params = {
        "n_estimators": 1000000,          # 最多训练的迭代次数（树的数量），非常大，通常配合 early_stopping 使用
        "loss_function": "MAE",           # 损失函数使用 MAE（Mean Absolute Error）—— 绝对值误差，更鲁棒于离群值
        "eval_metric": "MAE",             # 验证时的评估指标也是 MAE（与 loss_function 一致）
        "learning_rate": 0.02,            # 学习率，小学习率配合大 n_estimators，训练更稳定
        "depth": 6,                       # 每棵树的最大深度，控制模型复杂度（一般 6~10）
        "use_best_model": True,           # 使用验证集找到最佳模型（用于 early stopping）
        "subsample": 0.6,                 # 每次训练使用 60% 的样本，防止过拟合
        "bootstrap_type": "Bernoulli",   # 使用 Bernoulli 采样方法来做子样本（和 subsample 一起使用）
        "reg_lambda": 3,                  # L2 正则化系数，防止过拟合
        "one_hot_max_size": 2,           # 如果类别变量的唯一值数量 ≤ 2，则使用 One-Hot 编码
    }
    model_cb = CatBoostRegressor(**cb_params)
    # train the model
    model_cb.fit(
        k_x_train,
        k_y_train,
        eval_set=[(k_x_vali, k_y_vali)],
        verbose=100,  # 设置为较小的值，如100，以显示更频繁的进度更新
        early_stopping_rounds=300,
        plot=True  # 如果在支持可视化的环境中运行，这将显示训练进度图
    )
    oof_cb[vali_index] = model_cb.predict(k_x_vali, ntree_end=model_cb.best_iteration_)
    predictions_cb += (
        model_cb.predict(X_test_tree, ntree_end=model_cb.best_iteration_) / kfolder.n_splits
    )
    predictions_train_cb += (
        model_cb.predict(X_data_tree, ntree_end=model_cb.best_iteration_) / kfolder.n_splits
    )

print(
    "catboost score: {:<8.8f}".format(
        mean_absolute_error(np.expm1(oof_cb), np.expm1(Y_data))
    )
)

output_path = os.path.join(base_path, "user_data") # Correct output path for catboost results
# 测试集输出
predictions = predictions_cb
predictions[predictions < 0] = 0
sub = pd.DataFrame()
sub["SaleID"] = TestA_data_processed.SaleID
sub["price"] = predictions
sub.to_csv(os.path.join(output_path, "test_cab.csv"), index=False)

# 验证集输出
oof_cb[oof_cb < 0] = 0
sub = pd.DataFrame()
sub["SaleID"] = Train_data_processed.SaleID  # Train_data.SaleID的长度是149999，与oof_cb一致
sub["price"] = oof_cb
sub.to_csv(os.path.join(output_path, "train_cab.csv"), index=False)

## 读取神经网络模型数据
# Now, prepare data for NN model with one-hot encoding
# Combine train and test data again for one-hot encoding
combined_data_nn = pd.concat([Train_data.drop('price', axis=1), TestA_data], ignore_index=True)

# One-hot encode categorical features
combined_data_nn = pd.get_dummies(combined_data_nn, columns=categorical_cols, dummy_na=False)

# Fill remaining NaN values with 0 if any (after one-hot encoding, some might remain)
combined_data_nn = combined_data_nn.fillna(0)

# Re-split data for NN model
x = combined_data_nn.iloc[:len(Train_data)][feature_cols].values
y = np.array(Train_data['price']) # Use original Y_data (log1p transformed) for NN training
x_test = combined_data_nn.iloc[len(Train_data):][feature_cols].values

print("NN input X shape:", x.shape)
print("NN input X_test shape:", x_test.shape)
print("NN input Y shape:", y.shape)

# Normalize numerical features for NN after one-hot encoding
scaler_nn = StandardScaler()
# Get feature names after one-hot encoding
feature_cols_nn = [col for col in combined_data_nn.columns if col not in ["SaleID", 'regDate', 'creatDate']]
x = scaler_nn.fit_transform(combined_data_nn.iloc[:len(Train_data)][feature_cols_nn])
x_test = scaler_nn.transform(combined_data_nn.iloc[len(Train_data):][feature_cols_nn])

# Log transform y for NN
y = np.log1p(y) # Ensure y is log-transformed for NN as well

print(x)
print(x_test)
print(y)

# Adjust the training process learning rate
def scheduler(epoch):
    # At specified epochs, learning rate is reduced to 1/10 of its original value
    if epoch == 1400:
        lr = float(model.optimizer.learning_rate.numpy())
        model.optimizer.learning_rate.assign(lr * 0.1)
        print("lr changed to {}".format(lr * 0.1))
    if epoch == 1700:
        lr = float(model.optimizer.learning_rate.numpy())
        model.optimizer.learning_rate.assign(lr * 0.1)
        print("lr changed to {}".format(lr * 0.1))
    if epoch == 1900:
        lr = float(model.optimizer.learning_rate.numpy())
        model.optimizer.learning_rate.assign(lr * 0.1)
        print("lr changed to {}".format(lr * 0.1))
    return float(model.optimizer.learning_rate.numpy())

reduce_lr = LearningRateScheduler(scheduler)

kfolder = KFold(n_splits=2, shuffle=True)
oof_nn = np.zeros(len(x))
predictions_nn = np.zeros(len(x_test))
predictions_train_nn = np.zeros(len(x))
kfold = kfolder.split(x, y)
fold_ = 0
for train_index, vali_index in kfold:
    k_x_train = x[train_index]
    k_y_train = y[train_index]
    k_x_vali = x[vali_index]
    k_y_vali = y[vali_index]

    model = tf.keras.Sequential()
    model.add(
        tf.keras.layers.Dense(
            512, activation="relu", kernel_regularizer=tf.keras.regularizers.l2(0.02)
        )
    )
    model.add(
        tf.keras.layers.Dense(
            256, activation="relu", kernel_regularizer=tf.keras.regularizers.l2(0.02)
        )
    )
    model.add(
        tf.keras.layers.Dense(
            128, activation="relu", kernel_regularizer=tf.keras.regularizers.l2(0.02)
        )
    )
    model.add(
        tf.keras.layers.Dense(
            64, activation="relu", kernel_regularizer=tf.keras.regularizers.l2(0.02)
        )
    )
    model.add(
        tf.keras.layers.Dense(1, kernel_regularizer=tf.keras.regularizers.l2(0.02))
    )

    model.compile(
        loss="mean_absolute_error",
        optimizer=tf.keras.optimizers.Adam(),
        metrics=["mae"],
    )

    model.fit(
        k_x_train,
        k_y_train,
        batch_size=512,
        epochs=2000,
        validation_data=(k_x_vali, k_y_vali),
        callbacks=[reduce_lr],
    )  # callbacks=callbacks,
    oof_nn[vali_index] = model.predict(k_x_vali).reshape(
        (model.predict(k_x_vali).shape[0],)
    )
    predictions_nn += (
        model.predict(x_test).reshape((model.predict(x_test).shape[0],))
        / kfolder.n_splits
    )
    predictions_train_nn += (
        model.predict(x).reshape((model.predict(x).shape[0],)) / kfolder.n_splits
    )

print("NN score: {:<8.8f}".format(mean_absolute_error(oof_nn, y)))


output_path = os.path.join(base_path, "user_data") # Correct output path for nn results
# 测试集输出
predictions = predictions_nn
predictions[predictions < 0] = 0
sub = pd.DataFrame()
sub["SaleID"] = TestA_data_processed.SaleID # Use TestA_data_processed for consistency
sub["price"] = predictions
sub.to_csv(os.path.join(output_path, "test_nn.csv"), index=False)

# 验证集输出
oof_nn[oof_nn < 0] = 0
sub = pd.DataFrame()
sub["SaleID"] = Train_data_processed.SaleID # Use Train_data_processed for consistency
sub["price"] = oof_nn
sub.to_csv(os.path.join(output_path, "train_nn.csv"), index=False)


tree_data_path = os.path.join(base_path, "user_data") # Ensure tree_data_path is consistent for stack model

# 导入树模型lgb预测数据
predictions_lgb = np.array(
    pd.read_csv(os.path.join(tree_data_path, "test_lgb.csv"))["price"]
)
oof_lgb = np.array(pd.read_csv(os.path.join(tree_data_path, "train_lgb.csv"))["price"])

# 导入树模型cab预测数据
predictions_cb = np.array(
    pd.read_csv(os.path.join(tree_data_path, "test_cab.csv"))["price"]
)
oof_cb = np.array(pd.read_csv(os.path.join(tree_data_path, "train_cab.csv"))["price"])

# 读取price，对验证集进行评估
Train_data_original = pd.read_csv(train_file, sep=" ") # Reload original Train_data for Y_data
Y_data_original = Train_data_original["price"]

train_stack = np.vstack([oof_lgb, oof_cb]).transpose()
test_stack = np.vstack([predictions_lgb, predictions_cb]).transpose()
folds_stack = RepeatedKFold(n_splits=2, n_repeats=2)
tree_stack = np.zeros(train_stack.shape[0])
predictions = np.zeros(test_stack.shape[0])

# 二层贝叶斯回归stack
for fold_, (trn_idx, val_idx) in enumerate(folds_stack.split(train_stack, Y_data_original)):
    print("fold {}".format(fold_))
    trn_data, trn_y = train_stack[trn_idx], Y_data_original[trn_idx]
    val_data, val_y = train_stack[val_idx], Y_data_original[val_idx]

    Bayes = linear_model.BayesianRidge()
    Bayes.fit(trn_data, trn_y)
    tree_stack[val_idx] = Bayes.predict(val_data)
    predictions += Bayes.predict(test_stack) / 4

tree_predictions = np.expm1(predictions)
tree_stack = np.expm1(tree_stack)
tree_point = mean_absolute_error(tree_stack, np.expm1(Y_data_original)) # Use Y_data_original
print("树模型：二层贝叶斯: {:<8.8f}".format(tree_point))

# 导入神经网络模型预测训练集数据，进行三层融合

mix_nn = True
if mix_nn:
    oof_nn = np.array(pd.read_csv(os.path.join(tree_data_path, "train_nn.csv"))["price"]) # Use tree_data_path
    oof = (oof_nn + tree_stack) / 2
    predictions_nn = np.array(pd.read_csv(os.path.join(tree_data_path, "test_nn.csv"))["price"]) # Use tree_data_path
    predictions = (tree_predictions + predictions_nn) / 2
else:
    oof = tree_stack
    predictions = tree_predictions
all_point = mean_absolute_error(oof, np.expm1(Y_data_original)) # Use Y_data_original
print("总输出：三层融合: {:<8.8f}".format(all_point))


output_path = os.path.join(base_path, "prediction_result") # Correct output path for final predictions
# 测试集输出
sub = pd.DataFrame()
sub["SaleID"] = TestA_data_processed.SaleID # Use TestA_data_processed for final output
predictions[predictions < 0] = 0
# sub["price"] = predictions
sub["price"] = np.expm1(predictions)  # 反变换回真实价格
import random

x = random.randint(1, 10000)

sub.to_csv(os.path.join(output_path, f"predictions_{x}.csv"), index=False)
