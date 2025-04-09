
if __name__ == '__main__':
    print("---------------------- 开始 import ----------------------")

    # 导入必要的库

    import numpy as np
    import evaluate
    from datasets import load_dataset, Dataset
    from torchvision.transforms import (
        Compose,
        Resize,
        Normalize,
        RandomHorizontalFlip,
        RandomResizedCrop,
        RandomRotation,
        ColorJitter,
        ToTensor,
    )
    from transformers import (
        ViTForImageClassification,
        ViTImageProcessor,
        TrainingArguments,
        Trainer,
        DefaultDataCollator,
        EarlyStoppingCallback,
    )
    from torch.utils.data import DataLoader

    print("---------------------- import 完成 ----------------------")

    print("---------------------- 超参数设置 ----------------------")
    model_name = "google/vit-base-patch16-224-in21k"
    dataset_path = "data/touhou_split"  # 本地图片数据集的根路径
    output_dir = "output/vit_fine_tuned"  # 模型输出和保存的路径
    logging_dir = "logs/vit_fine_tuned"  # 日志文件保存路径
    logging_steps = 10  # 每多少步打印一次日志

    per_device_train_batch_size = 32  # 每个GPU的训练批次大小
    per_device_eval_batch_size = 64  # 每个GPU的评估批次大小
    gradient_accumulation_steps = 2  # 梯度累积步数，用于在有限的GPU内存下增加有效批次大小
    learning_rate = 1e-3  # 学习率
    num_train_epochs = 50  # 训练的总轮数
    weight_decay = 0.01  # 权重衰减
    evaluation_strategy = "epoch"  # 每个epoch结束后进行评估
    save_strategy = "epoch"  # 每个epoch结束后保存模型
    early_stopping_patience = 5  # 早停的耐心度，即连续多少个epoch没有改善就停止训练

    load_best_model_at_end = True  # 训练结束后加载最佳模型（基于评估指标）
    metric_for_best_model = "loss"  # 用于确定最佳模型的评估指标

    remove_unused_columns = False  # 保留数据集中的所有列，因为我们可能需要图像路径等信息

    print("---------------------- 数据集加载和预处理 ----------------------")
    print(f"加载本地图片数据集：{dataset_path}")
    # 使用 datasets 库加载本地图片数据集，假设数据集遵循 ImageNet 文件结构
    # 即在 dataset_path 下有 train 和 val 文件夹，每个文件夹下是按类别命名的子文件夹
    dataset = load_dataset("imagefolder", data_dir=dataset_path)

    print("获取数据集的类别标签")
    label2id = {
        label: id for id, label in enumerate(dataset["train"].features["label"].names)
    }
    id2label = {id: label for label, id in label2id.items()}
    num_labels = len(label2id)
    print(f"数据集包含 {num_labels} 个类别：{label2id}")

    print("加载 ViT 的图像处理器，用于将图像转换为模型所需的输入格式")
    processor = ViTImageProcessor.from_pretrained(model_name)

    print("定义图像预处理流程")
    image_size = processor.size["height"]  # 获取模型期望的图像尺寸
    train_transforms = Compose(
        [
            RandomResizedCrop(image_size),  # 随机裁剪并调整大小到目标尺寸
            RandomHorizontalFlip(),  # 随机水平翻转
            ColorJitter(brightness=(0.9, 1.1)),
            RandomRotation(degrees=45),  # 随机旋转
            ToTensor(),  # 将PIL图像或NumPy数组转换为PyTorch张量
        ]
    )
    val_transforms = Compose(
        [
            Resize(image_size),  # 调整大小到目标尺寸
            ToTensor(),  # 将PIL图像或NumPy数组转换为PyTorch张量
        ]
    )


    # 定义一个函数来对数据集中的每个样本进行预处理
    def train_preprocess_function(examples):
        images = examples["image"]
        transforms = train_transforms

        images = [transforms(image.convert("RGB")) for image in images]
        inputs = processor(images=images, return_tensors="pt")
        inputs["labels"] = examples["label"]
        return inputs


    def val_preprocess_function(examples):
        images = examples["image"]
        transforms = val_transforms

        images = [transforms(image.convert("RGB")) for image in images]
        inputs = processor(images=images, return_tensors="pt")
        inputs["labels"] = examples["label"]
        return inputs




    # 这是还没预处理的时候，进行预处理，然后存进磁盘。
    # 注意数据集文件夹是 data 还是 data_tiny_ver

    print("对训练集和验证集应用预处理函数")
    # `batched=True` 可以加速处理过程
    processed_train_dataset = dataset["train"].map(
        train_preprocess_function, batched=True, remove_columns=["image"]
    )
    print("训练集预处理完成！")
    processed_train_dataset.save_to_disk("./processed_train_dataset_tiny_ver")
    print("processed_train_dataset 已保存")

    if "validation" in dataset:
        processed_val_dataset = dataset["validation"].map(
            val_preprocess_function, batched=True, remove_columns=["image"]
        )
        print("验证集预处理完成！")
        processed_val_dataset.save_to_disk("./processed_val_dataset_tiny_ver")
        print("processed_val_dataset 已保存")
    else:
        processed_val_dataset = None
        print("警告：未找到验证集！")

    exit()  # 注意这里直接退出脚本了

    # 这是之前预训练过了，从磁盘把已经预处理成功的数据集直接加载过来

    '''
    processed_train_dataset = Dataset.load_from_disk("./processed_train_dataset")
    print("已加载预训练成功的数据集！")
    processed_val_dataset = Dataset.load_from_disk("./processed_val_dataset")
    print("已加载预训练成功的验证集！")
    '''

    print("---------------------- 模型加载 ----------------------")
    print(f"加载预训练模型：{model_name}")
    # 加载预训练的 ViT 模型，并指定分类头的输出类别数量
    model = ViTForImageClassification.from_pretrained(
        model_name,
        num_labels=num_labels,
        label2id=label2id,
        id2label=id2label,
    )

    print("---------------------- 训练参数设置 ----------------------")
    print("设置训练参数...")
    training_args = TrainingArguments(
        output_dir=output_dir,  # 模型输出目录
        logging_dir=logging_dir,  # 日志目录
        logging_steps=logging_steps,  # 日志步骤
        per_device_train_batch_size=per_device_train_batch_size,  # 每个设备的训练批次大小
        per_device_eval_batch_size=per_device_eval_batch_size,  # 每个设备的评估批次大小
        gradient_accumulation_steps=gradient_accumulation_steps,  # 梯度累积步数
        learning_rate=learning_rate,  # 学习率
        lr_scheduler_type="cosine",  # 学习率调度器类型
        num_train_epochs=num_train_epochs,  # 训练轮数
        weight_decay=weight_decay,  # 权重衰减
        eval_strategy=evaluation_strategy,  # 评估策略
        save_strategy=save_strategy,  # 保存策略
        load_best_model_at_end=load_best_model_at_end,  # 加载最佳模型
        metric_for_best_model=metric_for_best_model,  # 最佳模型评估指标
        fp16=True,  # 是否使用混合精度训练（需要GPU支持）
        report_to="tensorboard",  # 日志报告到 TensorBoard
        remove_unused_columns=remove_unused_columns,  # 保留未使用的列

        evaluation_strategy = "epoch",
        save_total_limit = 5,

        dataloader_num_workers=4,
        dataloader_drop_last=True,

        fp16_opt_level = "O2",  # 混合精度训练的优化级别
        fp16_full_eval = True,
    )

    print("---------------------- 数据整理器 ----------------------")
    # 使用默认的数据整理器，它会将预处理后的数据整理成模型可以接受的格式
    data_collator = DefaultDataCollator()

    print("---------------------- 评估指标计算 ----------------------")



    accuracy_metric = evaluate.load("accuracy")


    def compute_metrics(eval_pred):
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=1)
        return accuracy_metric.compute(predictions=predictions, references=labels)


    print("---------------------- 训练器初始化 ----------------------")
    print("初始化训练器...")
    trainer = Trainer(
        model=model,  # 要训练的模型
        args=training_args,  # 训练参数
        train_dataset=processed_train_dataset,  # 训练数据集
        eval_dataset=processed_val_dataset,  # 评估数据集
        data_collator=data_collator,  # 数据整理器
        compute_metrics=compute_metrics,  # 评估指标计算函数
        processing_class=processor,  # 使用图像处理器作为 tokenizer (虽然概念上不同，但在 Trainer 中需要)
        #callbacks=[EarlyStoppingCallback(early_stopping_patience=early_stopping_patience)],  # 早停回调
    )

    print("---------------------- 模型训练 ----------------------")
    print("开始模型训练...")
    try:
        train_results = trainer.train()
        trainer.save_model()
        trainer.log_metrics("train", train_results.metrics)
        trainer.save_state()
        print("训练脚本执行完毕！")
    except Exception as e:
        print(f"训练过程中发生错误：{e}")

