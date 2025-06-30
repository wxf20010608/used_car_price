import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


def analyze_correlation(train):
    # 计算特征与目标变量（价格）之间的相关性
    corr_with_price = train[['price'] + [col for col in train.columns if col not in ['price', 'SaleID']]].corr()['price']
    print("特征与价格的相关性:")
    print(corr_with_price)

    # 计算所有特征之间的相关性矩阵
    corr_matrix = train[train.columns.difference(['SaleID'])].corr()

    # 绘制相关性热力图
    plt.figure(figsize=(12, 10))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
    plt.title('特征之间的相关性热力图')
    plt.show()

    return corr_with_price, corr_matrix
