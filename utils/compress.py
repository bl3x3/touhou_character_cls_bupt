# 每次执行程序，从data/touhou中随机抽取10个文件夹，压缩到data/touhou，并记录压缩的文件夹名
import os
import shutil
import random

# 定义要压缩的文件夹路径
source_dir = "data/touhou"

# 定义压缩后的文件夹路径
target_dir = "data/touhou"

# 获取所有文件夹名
folder_names = os.listdir(source_dir)

# 获取已压缩的文件夹名并过滤掉已压缩的文件夹名以及非目录
with open("data/touhou/compressed_folders.txt", "r") as f:
    compressed_folders = f.read().splitlines()
folder_names = [
    folder_name
    for folder_name in folder_names
    if folder_name not in compressed_folders
    and os.path.isdir(os.path.join(source_dir, folder_name))
]

# 随机选择10个文件夹名
selected_folders = random.sample(folder_names, min(20, len(folder_names)))

# 遍历选择的文件夹名，将它们压缩到目标文件夹中
for folder_name in selected_folders:
    # 定义源文件夹路径
    source_folder = os.path.join(source_dir, folder_name)

    # 定义目标文件夹路径
    target_folder = os.path.join(target_dir, folder_name)

    # 压缩文件夹
    shutil.make_archive(target_folder, "zip", source_folder)
    print(f"压缩文件夹 {folder_name} 成功！")
    # # 删除源文件夹
    # shutil.rmtree(source_folder)
    # print(f'删除文件夹 {folder_name} 成功！')
    # 记录压缩的文件夹名
    with open("data/touhou/compressed_folders.txt", "a") as f:
        f.write(folder_name + "\n")
    print(f"记录文件夹 {folder_name} 成功！")
    print("--------------------------------------------------")
