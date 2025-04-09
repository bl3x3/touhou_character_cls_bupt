import os
import shutil
import imghdr
from collections import Counter

from PIL import Image


call_times = 0

def ifOK(imgPath: str):

    # 只是为了掌握进度
    global call_times

    # 细节玄学优化
    if call_times & 0x3ff == 0:
        print(f"ifOK() function has been called {call_times} times.")
    call_times += 1

    # imghdr 检查图片类型
    if (img_type := imghdr.what(imgPath)) not in ["jpg", "jpeg", "png"]:
        print(f"[found] file {imgPath}")
        print(f"image type is {img_type}")
        return False

    # PIL 检查文件完整性
    try:
        with Image.open(imgPath) as img:
            img.verify()
    except Exception as e:
        print(f"[found] file {imgPath}")
        print(f"exception: {e}")
        return False

    return True

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
    output_dir = "data/touhou_split_none_pil_ver"
    os.makedirs(output_dir, exist_ok=True)

    count_dict = Counter()
    for root, dirs, files in os.walk(input_dir):
        for file in files:
            class_name = root.split(os.sep)[-1]
            # print(root, dirs, files)
            if not ifOK(os.path.join(root, file)):
                count_dict[class_name] += 1
                '''
                shutil.move(
                    os.path.join(root, file),
                    os.path.join(output_dir, "_".join([class_name ,file])),
                )
                '''
                pass
print(count_dict)