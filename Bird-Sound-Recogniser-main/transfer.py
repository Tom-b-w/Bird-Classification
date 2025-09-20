import os

# 文件夹路径
folder_path = './audio_origin/birds'  # 替换为实际路径

# 获取文件夹中的所有文件名
bird_dict = set()

# 遍历文件夹中的文件
for filename in os.listdir(folder_path):
    if filename.endswith('.wav'):  # 只选择.wav文件
        bird_name = filename.split('.')[0]  # 去掉文件扩展名
        bird_dict.add(bird_name)  # 添加到字典中

# 输出字典格式
print(bird_dict)
