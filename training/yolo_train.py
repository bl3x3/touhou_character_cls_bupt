from ultralytics import YOLO
from torch.utils.tensorboard import SummaryWriter
import os
import shutil
import time


# 从命令行中启动训练
if __name__ == "__main__":
    model = YOLO("model/yolo11m-cls.pt")

    experiment_name = "yolo11m_freeze8_new_01"    # 修改实验名称
    model.train(
        data="data/touhou_split",    # 在这里指定数据集路径
        pretrained=True,
        epochs=200,
        patience=16, 
        batch=32, # batch=16
        imgsz=224,
        val=True,
        save=True,
        # cache='disk',
        # workers=4,
        amp=True,   # 混合精度
        optimizer='AdamW',
        freeze=8, # 冻结前8层参数
        cos_lr=True,
        lr0=0.001,
        project="touhou_cls_yolo11",
        name=experiment_name,     # 在这里指定实验名称
        exist_ok=True,

        ############# 数据增强
        # 第2组将这里全部删掉以使用默认参数
        hsv_h=0,
        hsv_s=0,
        hsv_v=0.1,
        degrees=30,
        mosaic=0,
        # crop_fraction=0.1,    # fatal error
        erasing=0.1,
        scale=0.2, 
        #############

        # resume=True,  
        device="0"    # 修改为 device="0"
    )
    model.save(f"model/{experiment_name}_end.pt")    # 修改保存模型路径
