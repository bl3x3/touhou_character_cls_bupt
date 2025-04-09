import os
from PIL import Image
import numpy as np


# # 1.重命名
image_dir = 'touhou_classification/static/images'
# for i, filename in enumerate(os.listdir(image_dir)):
#     if not filename.endswith('large.png'):
#         new_filename = f'{filename[:-4]}_large.png'
#         os.rename(os.path.join(image_dir, filename), os.path.join(image_dir, new_filename))

# 2.裁剪缩放图片
images = os.listdir(image_dir)
for image_file in images:
    # print(f"Processing image: {image_file}")
    image = Image.open(os.path.join(image_dir, image_file))
    if not image:
        print(f'Error opening image: {image_file}')
        continue
    if not image_file.endswith('large.png'):
        print(f'Skipping non-large image: {image_file}')
        continue
    image = np.array(image)
    height, width, _ = image.shape
    w_h = width / height
    changed = False
    if w_h < 0.75:
        new_width = width
        new_height = int(width / 0.75)
        new_image = image[:new_height,:,:]
        changed = True
    elif w_h > 1:
        new_width = int(height * 0.75)
        new_width_start = int((width - new_width) / 2)
        new_image = image[:, new_width_start:(new_width_start + new_width), :]
        changed = True
    if changed:
        Image.fromarray(new_image).save(os.path.join(image_dir, image_file))

        small_filename = f'{image_file[:-9]}small.png'
        # print(small_filename)
        small_image = new_image[:new_width, :, :]
        Image.fromarray(small_image).save(os.path.join(image_dir, small_filename))
    else:
        small_filename = f'{image_file[:-9]}small.png'
        # print(small_filename)
        small_image = image[:width, :, :]
        Image.fromarray(small_image).save(os.path.join(image_dir, small_filename))
