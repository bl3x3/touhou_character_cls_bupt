import os
import shutil
import random

def split_dataset(src_dir, dest_dir, train_ratio=0.7, val_ratio=0.15):
    """
    将图像数据集切分为训练集、验证集和测试集。

    Args:
        src_dir (str): 原始数据集的根目录路径。
        dest_dir (str): 目标数据集的根目录路径。
        train_ratio (float): 训练集比例。
        val_ratio (float): 验证集比例。
    """

    # 创建目标目录
    os.makedirs(os.path.join(dest_dir, 'train'), exist_ok=True)
    os.makedirs(os.path.join(dest_dir, 'test'), exist_ok=True)
    if val_ratio > 0:
        os.makedirs(os.path.join(dest_dir, 'val'), exist_ok=True)

    # 遍历每个类别
    for class_name in os.listdir(src_dir):
        class_src_dir = os.path.join(src_dir, class_name)
        if not os.path.isdir(class_src_dir):
            continue

        # 获取所有图像文件
        image_files = [f for f in os.listdir(class_src_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        random.shuffle(image_files)  # 随机打乱文件列表

        # 计算切分点
        num_images = len(image_files)
        train_split = int(num_images * train_ratio)
        val_split = int(num_images * (train_ratio + val_ratio))

        # 切分数据集
        train_images = image_files[:train_split]
        val_images = image_files[train_split:val_split]
        test_images = image_files[val_split:]

        # 创建类别目录并复制文件
        # def copy_images(images, subdir):
        #     if not images:
        #         return
        #     class_dest_dir = os.path.join(dest_dir, subdir, class_name)
        #     os.makedirs(class_dest_dir, exist_ok=True)
        #     for image in images:
        #         src_path = os.path.join(class_src_dir, image)
        #         dest_path = os.path.join(class_dest_dir, image)
        #         ####################
        #         shutil.copy(src_path, dest_path)    # 复制文件
        #         ####################

        def move_images(images, subdir):
            if not images:
                return
            class_dest_dir = os.path.join(dest_dir, subdir, class_name)
            os.makedirs(class_dest_dir, exist_ok=True)
            for image in images:
                src_path = os.path.join(class_src_dir, image)
                dest_path = os.path.join(class_dest_dir, image)
                ####################
                shutil.move(src_path, dest_path)    # 移动文件
                ####################

        # copy_images(train_images, 'train')
        # copy_images(test_images, 'test')
        move_images(train_images, 'train')
        move_images(test_images, 'test')
        if val_ratio > 0:
            # copy_images(val_images, 'val')
            move_images(val_images, 'val')

        print(f'Processed class: {class_name}')

# 示例用法
src_directory = 'data/touhou'  # 替换为您的原始数据集路径
dest_directory = 'data/touhou_split'  # 替换为您想要创建的目标数据集路径

split_dataset(src_directory, dest_directory, train_ratio=0.8, val_ratio=0.1)