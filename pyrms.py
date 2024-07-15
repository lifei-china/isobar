import pandas as pd
import numpy as np
import os
import glob
import re
import matplotlib.pyplot as plt


# 定义RMS宽度计算函数
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



def calculate_kurt_width(df, rms_value):
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
            weight = (delta_phi - delta_phi_m) ** 4 * dN_dDelta_phi
            numerator += weight
            denominator += dN_dDelta_phi

    kurt_value = numerator / (rms_value**4 * denominator) -3 if denominator != 0 else 0
    return kurt_value


# 设置文件夹路径
input_dir = 'output_data'

# 获取所有文件路径
file_paths = glob.glob(os.path.join(input_dir, '*.dat')) + glob.glob(os.path.join(input_dir, '*.txt'))

# 用于存储数据的字典
data = {}
scale_factors = {}

# 读取所有数据文件
for file_path in file_paths:
    file_name = os.path.basename(file_path)
    prefix = file_name.split('_')[0]
    
    if prefix not in data:
        data[prefix] = []
    
    # 读取文件
    if file_name.endswith('.dat'):
        df = pd.read_csv(file_path, delim_whitespace=True, header=None, names=['binCenter', 'binContent', 'binError'])
        data[prefix].append((file_name, df))
    elif file_name.endswith('_scale_A_values.txt'):
        scale_factors[file_name] = pd.read_csv(file_path, delim_whitespace=True, header=None, names=['scale']).values.flatten()



# 处理和输出数据
result_data = {}

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


    # 对配对文件进行操作
    for same_file, same_df in file_dict['same'].items():
        mix_file = same_file.replace('same', 'mix')
        if mix_file in file_dict['mix']:
            mix_df = file_dict['mix'][mix_file]
            

            # 获取相应的乘法因子文件名
            scale_file_name = f"{prefix}_200_scale_A_values.txt"

            if scale_file_name in scale_factors:
                scale_factors_list = scale_factors[scale_file_name]

            same_num = int(re.search(r'azimuthal_(\d+)', same_file).group(1))
            print("same_num",same_file, same_num)


            # 确保数据长度相同
            if len(same_df) == len(mix_df) and same_num < len(scale_factors_list):
                # 计算新的第二列数据（减法）
                # new_binContent = same_df['binContent'] - mix_df['binContent']
                new_binContent = same_df['binContent'] -( mix_df['binContent']* scale_factors_list[same_num])
                print(f"prefix:{prefix}, same_num: {same_num}, binContent:{new_binContent}")

                # 计算新的第三列误差
                new_binError = (same_df['binError']**2 + (scale_factors_list[same_num] * mix_df['binError'])**2)**0.5

                if prefix == "zr-8":
                    new_binContent = new_binContent - 0.04


                # 生成新的DataFrame
                new_df = pd.DataFrame({
                    'binCenter': same_df['binCenter'],
                    'new_binContent': new_binContent,
                    'new_binError': new_binError
                })
                
                # 将结果存储到result_data字典中
                result_data[f"{prefix}_{same_file.split('_')[2]}_{same_num}"] = new_df
                print(f"this is :{prefix}_{same_file.split('_')[2]}_{same_num}")

# 计算每个系统的RMS宽度
rms_results = {}
kurt_results = {}   #kurtosis
kurt_rms_results = {}   # |kurtosis|/rms

for key, df in result_data.items():
    same_num = int(re.search(r'azimuthal_(\d+)', key).group(1))
    if same_num == 0:
        rms_value = calculate_rms_width(df)
        rms_results[key] = rms_value
        kurt_value = calculate_kurt_width(df, rms_value)
        kurt_results[key] = kurt_value
        kurt_rms_results[key] = np.abs(kurt_value)/rms_value
        print(f"RMS and kurt for {key}: {rms_value} {kurt_value}")

keys = list(rms_results.keys())

# 提取prefix和case
cases = ['1', '3', '4', '8']
case_labels = [f'case{case}' for case in cases]
case_indices = {f'ru-{case}': idx for idx, case in enumerate(cases)}
case_indices.update({f'zr-{case}': idx for idx, case in enumerate(cases)})

# 数据存储到对应位置
case_data_rms = {case: {'ru': None, 'zr': None} for case in cases}
case_data_kurt = {case: {'ru': None, 'zr': None} for case in cases}
case_data_kurt_rms = {case: {'ru': None, 'zr': None} for case in cases}

for key in keys:
    case_match = re.search(r'(ru|zr)-(\d+)', key)
    if case_match:
        system = case_match.group(1)
        case = case_match.group(2)
        if case in case_data_rms:
            case_data_rms[case][system] = rms_results[key]
        if case in case_data_kurt:
            case_data_kurt[case][system] = kurt_results[key]
        if case in case_data_kurt_rms:
            case_data_kurt_rms[case][system] = kurt_rms_results[key]

# 绘制图形
bar_width = 0.35
positions = np.arange(len(cases))

ru_rms_values = [case_data_rms[case]['ru'] for case in cases]
zr_rms_values = [case_data_rms[case]['zr'] for case in cases]

ru_kurt_values = [case_data_kurt[case]['ru'] for case in cases]
zr_kurt_values = [case_data_kurt[case]['zr'] for case in cases]

ru_kurt_rms_values = [case_data_kurt_rms[case]['ru'] for case in cases]
zr_kurt_rms_values = [case_data_kurt_rms[case]['zr'] for case in cases]

fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(9, 4))

# 绘制RMS值
ax1.bar(positions, ru_rms_values, bar_width, color='blue', label='ru RMS')
ax1.bar(positions + bar_width, zr_rms_values, bar_width, color='green', label='zr RMS')
ax1.set_xlabel('Case')
ax1.set_ylabel('RMS')
# ax1.set_title('RMS Width for different systems (same_num = 0)')
ax1.set_xticks(positions + bar_width / 2)
ax1.set_xticklabels(case_labels)
ax1.set_ylim(-0.1, 1.0)
ax1.legend()

# 绘制Kurtosis值
ax2.plot(positions + bar_width / 2, ru_kurt_values, color='blue', marker='o', linestyle='dashed', label='ru Kurtosis')
ax2.plot(positions + bar_width / 2, zr_kurt_values, color='green', marker='o', linestyle='dashed', label='zr Kurtosis')
ax2.set_xlabel('Case')
ax2.set_ylabel('Kurtosis')
# ax2.set_title('Kurtosis for different systems (same_num = 0)')
ax2.set_xticks(positions + bar_width / 2)
ax2.set_xticklabels(case_labels)
ax2.set_ylim(-1.0, -0.80)
ax2.legend()


# 绘制Kurtosis值
ax3.plot(positions + bar_width / 2, ru_kurt_rms_values, color='blue', marker='o', linestyle='dashed', label='ru')
ax3.plot(positions + bar_width / 2, zr_kurt_rms_values, color='green', marker='o', linestyle='dashed', label='zr')
ax3.set_xlabel('Case')
ax3.set_ylabel(r'$\frac{|Kurtosis|}{RMS}$')
# ax3.set_title('Kurtosis for different systems (same_num = 0)')
ax3.set_xticks(positions + bar_width / 2)
ax3.set_xticklabels(case_labels)
# ax3.set_ylim(-1.0, -0.80)
ax3.legend()



plt.tight_layout()
plt.savefig("rms_kurt.jpg")
plt.show()


