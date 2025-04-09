import os
import imghdr
import shutil
from collections import Counter

# imgpaths=os.listdir()
# desPath="others"
# os.makedirs(desPath)
# for imgPath in imgpaths:
#     flag=False
#     if imghdr.what(imgPath) not in ["jpg","jpeg","png"]:
#         shutil.move(imgPath, os.path.join(desPath,imgPath))


def ifOK(imgPath):
    if imghdr.what(imgPath) in ["jpg", "jpeg", "png"]:
        return True
    else:
        return False

"""
环境要求：小于等于python3.12
示例用法：
文件夹结构如下：
data/
    touhou_split/
        class1/
            img1.jpg
    touhou_split_none/
        class1_img1.jpg
findNoneImg.py
设置路径参数
input_dir = "data/touhou_split"
output_dir = "data/touhou_split_none"
随后在当前目录下打开终端运行 findNoneImg.py
"""
if __name__ == "__main__":
    input_dir = "data/touhou_split"
    output_dir = "data/touhou_split_none"
    os.makedirs(output_dir, exist_ok=True)

    count_dict = Counter()
    for root, dirs, files in os.walk(input_dir):
        for file in files:
            class_name = root.split(os.sep)[-1]
            # print(root, dirs, files)
            if not ifOK(os.path.join(root, file)):
                count_dict[class_name] += 1
                shutil.move(
                    os.path.join(root, file),
                    os.path.join(output_dir, "_".join([class_name ,file])),
                )
print(count_dict)