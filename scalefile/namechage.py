import os

# 定义文件夹路径
directory = './'

# 获取所有文件名
file_names = os.listdir(directory)

# 遍历文件名并进行重命名
for file_name in file_names:
    if 'mix' in file_name:
        new_file_name = file_name.replace('mix', 'scale')
        old_file_path = os.path.join(directory, file_name)
        new_file_path = os.path.join(directory, new_file_name)
        os.rename(old_file_path, new_file_path)
        print(f'Renamed: {file_name} -> {new_file_name}')
