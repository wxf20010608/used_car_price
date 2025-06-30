import os
import pandas as pd
import numpy as np

# 设置文件路径
data_path = "d:/AI_Learning/python/Neural_Networks/tianchi_competition/project/user_data/"
output_path = "d:/AI_Learning/python/Neural_Networks/tianchi_competition/project/user_data/original_price/"

# 创建输出目录（如果不存在）
os.makedirs(output_path, exist_ok=True)

# 获取所有CSV文件
files = [f for f in os.listdir(data_path) if f.endswith('.csv')]

for file in files:
    print(f"处理文件: {file}")
    # 读取CSV文件
    df = pd.read_csv(os.path.join(data_path, file))
    
    # 检查是否有price列
    if 'price' in df.columns:
        # 将对数空间的价格转换为原始价格空间
        df['price'] = np.expm1(df['price'])
        
        # 保存转换后的文件
        output_file = "original_" + file
        df.to_csv(os.path.join(output_path, output_file), index=False)
        print(f"已保存转换后的文件: {output_file}")
    else:
        print(f"文件 {file} 没有price列，跳过")

print("所有文件处理完成！")