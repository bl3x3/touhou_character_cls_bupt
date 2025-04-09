import os
import imagehash
from PIL import Image, UnidentifiedImageError
import shutil
import argparse
import numpy as np
import time

# 设置阈值
threshold = 5
# 设置文件格式
file_format = (".png", ".jpeg", ".jpg")


def compare_images(image1, image2):
    try:
        # 计算图片的hash值
        hash1 = imagehash.average_hash(Image.open(image1))
        hash2 = imagehash.average_hash(Image.open(image2))
        # 计算图片的相似度
        difference = hash1 - hash2
        # 如果相似度大于阈值，则认为图片相似
        return difference <= threshold
    except (FileNotFoundError, UnidentifiedImageError) as e:
        print(f"Error comparing images {image1} and {image2}: {e}")
        return False


def process_directories(input_dir, target_dir, output_dir):
    try:
        # 获取输入目录和目标目录中的一级子目录
        input_subdirs = [
            d
            for d in os.listdir(input_dir)
            if os.path.isdir(os.path.join(input_dir, d))
        ]
        # target_subdirs = [d for d in os.listdir(target_dir) if os.path.isdir(os.path.join(target_dir, d))]

        for input_subdir in input_subdirs:
            # 构建输入子目录和目标子目录的路径
            input_subdir_path = os.path.join(input_dir, input_subdir)
            target_subdir_path = os.path.join(target_dir, input_subdir)

            # 检查目标子目录是否存在
            if not os.path.exists(target_subdir_path):
                print(
                    f"Target subdirectory {target_subdir_path} does not exist. Skipping."
                )
                continue

            # 构建输出子目录的路径
            output_subdir = os.path.join(output_dir, f"new_{input_subdir}")
            os.makedirs(output_subdir, exist_ok=True)

            # 获取输入子目录和目标子目录中的所有图片
            input_image_files = [
                f
                for f in os.listdir(input_subdir_path)
                if f.endswith(file_format)
            ]
            target_image_files = [
                f
                for f in os.listdir(target_subdir_path)
                if f.endswith(file_format)
            ]

            # 初始化统计变量
            total_images = len(input_image_files)
            non_duplicate_images = 0
            duplicate_images = []
            error_images = []

            # 遍历目标子目录中的所有图片，计算得到hash值
            target_hashes = []
            for target_file in target_image_files:
                target_image_path = os.path.join(target_subdir_path, target_file)
                try:
                    target_hash = imagehash.average_hash(Image.open(target_image_path))
                    target_hashes.append(target_hash)
                except (FileNotFoundError, UnidentifiedImageError) as e:
                    print(f"Error processing image {target_image_path}: {e}")
            target_hashes = np.array(target_hashes)

            # 遍历输入子目录中的所有图片，计算得到hash值
            for image_file in input_image_files:
                image_path = os.path.join(input_subdir_path, image_file)
                try:
                    image_hash = imagehash.average_hash(Image.open(image_path))
                    # 比较输入图片和目标图片的hash值，如果相似度大于阈值，则认为图片相似
                    if np.any(np.abs(target_hashes - image_hash) <= threshold):
                        # 找到重复图片
                        similarity = np.min(np.abs(target_hashes - image_hash))
                        duplicate_images.append((image_file, similarity))
                        continue
                    else:
                        # 将非重复图片移动到输出目录
                        shutil.move(image_path, os.path.join(output_subdir, image_file))
                        non_duplicate_images += 1
                except (FileNotFoundError, UnidentifiedImageError) as e:
                    error_images.append(image_file)
                    print(f"Error processing image {image_path}: {e}")
            # 输出统计信息
            with open(os.path.join(output_subdir, f"{time.strftime('%Y%m%d%H%M%S')}_result.txt"), "w") as f:
                f.write(f"Input directory: {input_subdir}\n")
                f.write(f"Total images: {total_images}\n")
                f.write(f"Non-duplicate images: {non_duplicate_images}\n")
                f.write(f"Duplicate images: {len(duplicate_images)}\n")
                f.write("Duplicate images with similarity:\n")
                for image_file, similarity in duplicate_images:
                    f.write(f"{image_file}: {similarity}\n")
                f.write(f"Error images: {len(error_images)}\n")
    except Exception as e:
        print(f"An error occurred: {e}")


def main():
    # 获取命令行参数
    parser = argparse.ArgumentParser(description="Compare images and remove duplicates")
    parser.add_argument(
        "-i", "--input_dir", help="Input directory containing images", required=True
    )
    parser.add_argument(
        "-t",
        "--target_dir",
        help="Target directory containing images to compare against",
        required=True,
    )
    parser.add_argument(
        "-o",
        "--output_dir",
        help="Output directory to save non-duplicate images",
        required=True,
    )
    args = parser.parse_args()

    try:
        # 获取输入目录和目标目录的路径
        input_dir = os.path.abspath(args.input_dir)
        target_dir = os.path.abspath(args.target_dir)
        output_dir = os.path.abspath(args.output_dir)

        # 检查输入目录和目标目录是否存在
        if not os.path.exists(input_dir):
            print(f"Input directory {input_dir} does not exist.")
            return
        if not os.path.exists(target_dir):
            print(f"Target directory {target_dir} does not exist.")
            return

        # 处理目录
        process_directories(input_dir, target_dir, output_dir)
    except Exception as e:
        print(f"An error occurred: {e}")


if __name__ == "__main__":
    main()
