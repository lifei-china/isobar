import pandas as pd
import os
import glob
import re
import matplotlib.pyplot as plt

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
                # result_data[f"{prefix}_{same_file.split('_')[2]}"] = new_df
                # print(f"this is :{prefix}_{same_file.split('_')[2]}")

# 创建画布
fig, axes = plt.subplots(2, 2, figsize=(8, 6))

# 绘制结果
plot_mapping = {
    'ru-1': (0, 0),
    'zr-1': (0, 0),
    'ru-3': (0, 1),
    'zr-3': (0, 1),
    'ru-4': (1, 0),
    'zr-4': (1, 0),
    'ru-8': (1, 1),
    'zr-8': (1, 1)
}

for key, df in result_data.items():
    prefix = key.split('_')[0]
    same_num = int(re.search(r'azimuthal_(\d+)', key).group(1))
    if prefix in plot_mapping and same_num == 0: 
        ax = axes[plot_mapping[prefix]]
        ax.errorbar(df['binCenter'], df['new_binContent'], yerr=df['new_binError'], label=key)
        ax.set_xlabel('$\Delta \phi$')
        ax.set_ylabel('$1/N_{trig}dN_{pair}/d\Delta\phi$')
        ax.set_ylim(-0.1,1.7 )
        ax.legend()
        ax.grid(True)

# 设置图表属性
plt.tight_layout()
plt.savefig('isobar_azimuthal.png')
plt.show()