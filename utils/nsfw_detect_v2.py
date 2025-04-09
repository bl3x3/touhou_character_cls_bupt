# requirement: transformers, torch, Pillow
### pip install transformers torch Pillow
# 此外需要挂梯子，因为transformers需要从huggingface下载模型
from transformers import pipeline
from PIL import Image
import os


model = pipeline("image-classification", model="LukeJacob2023/nsfw-image-detector-384")


def nsfw_detect(img_path):
    """
    判断图片是否为nsfw图片
    :param img_path: 图片路径
    :return: True for nsfw, or False for not
    """
    result = model(Image.open(img_path))
    if result[0]["label"] == "drawings" or (
        result[1]["label"] == "drawings" and result[1]["score"] > 0.0001
    ):
        return False
    return True


if __name__ == "__main__":
    # tests = ["chen.png", "daiyousei.png", "cirno2.png", "flan.png", "mokou.png", "mokou2.png", "mokou3.png", "medicine.jpg", "image.png", "medicine2.jpg"]
    # for test in tests:
    #     # print(test, model(os.path.join("touhou_classification", "assets", test)))
    #     print(test, nsfw_detect(os.path.join("touhou_classification", "assets", test)))
    # # chen.png True
    # # daiyousei.png True
    # # cirno2.png False
    # # flan.png False

    # 处理一个子目录的图片
    chara_path = "hakurei_reimu"  # 替换为实际的子目录路径
    nsfw_buffer_dir = "nsfw_buffer/hakurei_reimu"  # 替换为实际的nsfw图片缓存目录
    if os.path.isdir(chara_path):
        for image in os.listdir(chara_path):
            if nsfw_detect(os.path.join(chara_path, image)):
                os.rename(
                    os.path.join(chara_path, image),
                    os.path.join(nsfw_buffer_dir, image),
                )
    # 在这之后，子目录中的nsfw图片会被移动到nsfw_buffer_dir目录中，可以手动检查并删除这些图片
    # 或者根据需要进一步处理这些图片
