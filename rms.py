import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import glob
import re

# 设置文件夹路径
input_dir = 'output_data'
scale_dir = 'scalefile'

# 获取所有文件路径
file_paths = glob.glob(os.path.join(input_dir, '*.dat'))
scale_paths = glob.glob(os.path.join(scale_dir, '*.txt'))

# 用于存储数据的字典
data = {}
scale_factors = {}

# 读取所有数据文件
for file_path in file_paths:
    file_name = os.path.basename(file_path)
    prefix = file_name.split('_')[0] + '_' + file_name.split('_')[1]  # 例如 ru-1
    if prefix not in data:
        data[prefix] = []
    df = pd.read_csv(file_path, delim_whitespace=True, header=None, names=['binCenter', 'binContent', 'binError'])
    data[prefix].append((file_name, df))

# 读取所有缩放因子文件
for scale_path in scale_paths:
    scale_file_name = os.path.basename(scale_path)
    df = pd.read_csv(scale_path, delim_whitespace=True, header=None, names=['scaleFactor'])
    scale_factors[scale_file_name] = df['scaleFactor'].tolist()

# 处理数据并计算RMS宽度
result_data = {}
rms_results = {}

def calculate_rms_width(df):
    pi = np.pi
    delta_phi_m = pi  # Δφm 设置为 π
    low_limit = 1.5  # 远侧区域的下限
    high_limit = 2 * pi - 1.5  # 远侧区域的上限

    numerator = 0.0  # 分子初始化
    denominator = 0.0  # 分母初始化

    for index, row in df.iterrows():
        delta_phi = row['binCenter']  # 获取 Δφ 的值
        dN_dDelta_phi = row['new_binContent']  # 获取 dN/dΔφ 的值

        # 只考虑远侧区域内的 Δφ
        if low_limit <= delta_phi <= high_limit:
            weight = (delta_phi - delta_phi_m) ** 2 * dN_dDelta_phi
            numerator += weight
            denominator += dN_dDelta_phi

    rms_value = np.sqrt(numerator / denominator) if denominator != 0 else 0
    return rms_value

# 遍历数据
for prefix, files in data.items():
    print(f"Prefix: {prefix}")
    
    # 创建一个字典来存储same和mix文件
    file_dict = {'same': {}, 'mix': {}}
    
    # 分配文件到字典
    for file_name, df in files:
        if 'same' in file_name:
            file_dict['same'][file_name] = df
        elif 'mix' in file_name:
            file_dict['mix'][file_name] = df
    
    # 获取相应的乘法因子文件名
    scale_file_name = f"{prefix}_200_scale_A_values.txt"
    if scale_file_name in scale_factors:
        scale_factors_list = scale_factors[scale_file_name]

    # 对配对文件进行操作
    for same_file, same_df in file_dict['same'].items():
        mix_file = same_file.replace('same', 'mix')
        if mix_file in file_dict['mix']:
            mix_df = file_dict['mix'][mix_file]

            same_num = int(re.search(r'azimuthal_(\d+)', same_file).group(1))
            print("same_num",same_file, same_num)

            # 确保数据长度相同
            if len(same_df) == len(mix_df) and same_num < len(scale_factors_list):
                # 计算新的第二列数据（减法）
                new_binContent = same_df['binContent'] -( mix_df['binContent']* scale_factors_list[same_num])
                print(f"prefix:{prefix}, same_num: {same_num}, binContent:{new_binContent}")

                # 计算新的第三列误差
                new_binError = (same_df['binError']**2 + mix_df['binError']**2)**0.5
                
                # 生成新的DataFrame
                new_df = pd.DataFrame({
                    'binCenter': same_df['binCenter'],
                    'new_binContent': new_binContent,
                    'new_binError': new_binError
                })
                
                # 将结果存储到result_data字典中
                result_data[f"{prefix}_azimuthal_{same_num}"] = new_df
                print(f"this is :{prefix}_azimuthal_{same_num}")

                # 计算并存储RMS值
                rms_value = calculate_rms_width(new_df)
                rms_results[f"{prefix}_azimuthal_{same_num}"] = rms_value
                print(f"RMS for {prefix}_azimuthal_{same_num}: {rms_value}")

# 只绘制same_num = 0的结果
plt.figure(figsize=(12, 8))
keys = [key for key in rms_results.keys() if key.endswith('_azimuthal_0')]

# 提取prefix和case
cases = ['1', '3', '4', '8']
case_labels = [f'case{case}' for case in cases]
case_indices = {f'ru-{case}': idx for idx, case in enumerate(cases)}
case_indices.update({f'zr-{case}': idx for idx, case in enumerate(cases)})

# 数据存储到对应位置
case_data = {case: {'ru': None, 'zr': None} for case in cases}
for key in keys:
    case_match = re.search(r'(ru|zr)-(\d+)', key)
    if case_match:
        system = case_match.group(1)
        case = case_match.group(2)
        if case in case_data:
            case_data[case][system] = rms_results[key]

# 绘制图形
bar_width = 0.35
positions = np.arange(len(cases))

ru_values = [case_data[case]['ru'] for case in cases]
zr_values = [case_data[case]['zr'] for case in cases]

plt.bar(positions, ru_values, bar_width, color='blue', label='ru')
plt.bar(positions + bar_width, zr_values, bar_width, color='green', label='zr')

plt.xlabel('Case')
plt.ylabel('RMS Width')
plt.title('RMS Width for different systems (same_num = 0)')
plt.xticks(positions + bar_width / 2, case_labels)
plt.legend()
plt.tight_layout()
plt.savefig('rms_widths.png')
plt.show()
