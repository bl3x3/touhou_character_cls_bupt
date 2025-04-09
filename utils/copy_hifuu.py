import os
import shutil
import numpy as np
import tensorflow as tf
import cv2
import matplotlib.pyplot as plt
import torch

# # 加载TFLite模型并分配张量
# print("Loading model...")
# interpreter = tf.lite.Interpreter(
#     model_path=r"D:\PycharmProjects\deeplearning_25spr\touhou_classification\model\nas_net_mobile.tflite"
# )
# interpreter.allocate_tensors()

# input_details = interpreter.get_input_details()  # 输入
# output_details = interpreter.get_output_details()  # 输出

# input_shape = input_details[0]["shape"]  # 获取输入的shape
# input_dtype = input_details[0]["dtype"]  # 获取输入的数据类型


# def preprocess_image(image_path):
#     try:
#         input_data = np.fromfile(image_path, dtype=np.uint8)
#         if input_data is None:
#             return None
#         input_data = cv2.imdecode(input_data, cv2.IMREAD_COLOR)
#         input_data = cv2.resize(input_data, (input_shape[1], input_shape[2]))
#         input_data = cv2.cvtColor(input_data, cv2.COLOR_BGR2RGB)
#         input_data = np.expand_dims(input_data, axis=0)
#         input_data = input_data.astype(input_dtype)
#         input_data /= 255.0

#         interpreter.set_tensor(input_details[0]["index"], input_data)  # 输入给模型
#         interpreter.invoke()  # 运行
#         # 函数`get_tensor()`返回张量数据的副本
#         # 使用`tensor()`来获得一个指向这个张量的指针
#         output_data = interpreter.get_tensor(output_details[0]["index"])
#         # print(output_data)
#         return output_data
#     except Exception as e:
#         print(f"Error processing image {image_path}: {e}")
#         return None


# # print(np.argmax(output_data))

# with open(r"D:\PycharmProjects\deeplearning_25spr\touhou_classification\assets\characters.txt", "r", encoding="utf-8") as f:
#     characters = f.readlines()
#     characters = [character.strip() for character in characters]

# # print(characters[np.argmax(output_data)])

src_dir = r"D:\PycharmProjects\nonebot\hifuu-bot\img"
dst_dir = r"D:\PycharmProjects\deeplearning_25spr\data"

# classes = ["usami_renko", "maribel_hearn", "usami_sumireko"]
# classes_cn = ["宇佐见莲子", "玛艾露贝莉·赫恩", "宇佐见堇子"]
# classes_idx = [characters.index(character) for character in classes_cn]
# idx2classes = {idx: character for idx, character in zip(classes_idx, classes)}
# print(classes_idx)
# print(idx2classes)

# # result = preprocess_image(r"D:\PycharmProjects\nonebot\hifuu-bot\img\Aolaite_\0004.png")
# # print(result)
# # print(np.argmax(result), characters[np.argmax(result)])

# # result = torch.from_numpy(result)
# # val, idx = torch.topk(result, k=3)
# # print(val, idx)

# def judge(image_path):
#     result = preprocess_image(image_path)
#     if result is None:
#         return None
#     result = torch.from_numpy(result).squeeze()
#     val, idx = torch.topk(result, k=2)
#     # print(val, idx)
#     if idx[0] not in classes_idx or (idx[1] in classes_idx and val[1] > 0.5) or val[0] < 0.5:
#         # 如果识别为其它角色，或有多个角色，或第一个角色置信度较低，则返回None
#         return None
#     else:
#         return idx[0].data.item()


# for char in classes:
#     char_dir = os.path.join(dst_dir, char)
#     if not os.path.exists(char_dir):
#         os.makedirs(char_dir)

# authors = os.listdir(src_dir)
# renko_cnt, merry_cnt, sumireko_cnt = 0, 0, 0
# max_count = 1000
# for author in authors:
#     author_dir = os.path.join(src_dir, author)
#     if not os.path.isdir(author_dir):
#         continue
#     print(f"Processing {author}...")
#     images = os.listdir(author_dir)
#     for image in images:
#         image_path = os.path.join(author_dir, image)
#         result = judge(image_path)
#         if result is not None:
#             # 如果识别为指定角色，则复制到目标目录
#             # char/author_image
#             match result:
#                 case 104:
#                     if renko_cnt < max_count:
#                         renko_cnt += 1
#                     else:
#                         continue
#                 case 60:
#                     if merry_cnt < max_count:
#                         merry_cnt += 1
#                     else:
#                         continue
#                 case 105:
#                     if sumireko_cnt < max_count:
#                         sumireko_cnt += 1
#                     else:
#                         continue
#             dst_path = os.path.join(dst_dir, idx2classes[result], f"{author}_{image}")
#             if not os.path.exists(dst_path):
#                 shutil.copy(image_path, dst_path)

# print(f"Renko count: {renko_cnt}")
# print(f"Merry count: {merry_cnt}")
# print(f"Sumireko count: {sumireko_cnt}")


charas = ['renko', 'merry', 'sumireko']
final_charas = ['usami_renko', 'maribel_hearn', 'usami_sumireko']
cnt = {chara: 0 for chara in charas}
for chara, final_chara in zip(charas, final_charas):
    root_dir = os.path.join(dst_dir, chara)
    authors = os.listdir(root_dir)

    exist_images = set(os.listdir(os.path.join(dst_dir, final_chara)))

    for author in authors:
        author_dir = os.path.join(root_dir, author)
        images = os.listdir(author_dir)
        for image in images:
            image_path = os.path.join(author_dir, image)
            if author + '_' + image not in exist_images:
                shutil.move(image_path, os.path.join(dst_dir, final_chara, author + '_' + image))
                cnt[chara] += 1

print(cnt)
