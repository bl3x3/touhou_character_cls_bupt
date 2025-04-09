from datasets import load_dataset
from transformers import (
    AutoImageProcessor,
    AutoFeatureExtractor,
    AutoModelForImageClassification,
    Trainer,
    TrainingArguments,
)
import torch
from torchvision import transforms as T
import matplotlib.pyplot as plt


# 加载数据集
print("Loading dataset...")
dataset_path = "../../data/touhou_split"
dataset = load_dataset("imagefolder", data_dir=dataset_path)
# print(dataset)
# print(dataset["train"][0].keys())


# 转换label
labels = dataset["train"].features["label"].names
label2id, id2label = dict(), dict()
for i, label in enumerate(labels):
    label2id[label] = i
    id2label[i] = label


# 加载特征提取器和模型
image_processor = AutoImageProcessor.from_pretrained(
    "google/vit-base-patch16-224", use_fast=True
)
# feature_extractor = AutoFeatureExtractor.from_pretrained("google/vit-base-patch16-224")
train_transform = T.Compose(
    [
        T.ToTensor(),
        T.RandomResizedCrop(
            size=(256, 256),
            scale=(0.75, 1.0),
            ratio=(0.75, 1.33),
            interpolation=T.InterpolationMode.BICUBIC,
        ),
        T.RandomHorizontalFlip(),
        T.ColorJitter(brightness=(0.9, 1.1)),
        T.RandomRotation(degrees=45),
        T.RandomErasing(p=0.05, value="random"),
        # T.Normalize(mean=data_config["mean"], std=data_config["std"])
    ]
)
normalize = T.Normalize(mean=image_processor.image_mean, std=image_processor.image_std)
size = (
    image_processor.size["shortest_edge"]
    if "shortest_edge" in image_processor.size
    else (image_processor.size["height"], image_processor.size["width"])
)
_transforms = T.Compose([T.RandomResizedCrop(size), T.ToTensor(), normalize])


# 定义图片预处理函数
def transforms(examples):
    examples["pixel_values"] = [
        _transforms(img.convert("RGB")) for img in examples["image"]
    ]
    del examples["image"]
    return examples

def train_transforms(examples):
    examples["pixel_values"] = [
        train_transform(img.convert("RGB")) for img in examples["image"]
    ]
    del examples["image"]
    return examples

# precessed_dataset = dataset.map(transforms, batched=True, batch_size=32)
processed_dataset = dataset.with_transform(transforms)
# processed_train_dataset = dataset['train'].map(train_transforms, batched=True, batch_size=32, remove_columns=['image'])
# processed_validation_dataset = dataset['validation'].map(transforms, batched=True, batch_size=32, remove_columns=['image'])
# # processed_test_dataset = dataset['test'].map(transforms, batched=True, batch_size=32, remove_columns=['image'])


from transformers import DefaultDataCollator

data_collator = DefaultDataCollator(return_tensors="pt")


import evaluate
import numpy as np

accuracy = evaluate.load("accuracy")


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return accuracy.compute(predictions=predictions, references=labels)


model = AutoModelForImageClassification.from_pretrained(
    "google/vit-base-patch16-224-in21k",
    num_labels=len(labels),
    id2label=id2label,
    label2id=label2id,
)


training_args = TrainingArguments(
    output_dir="touhou_cls_vit",
    remove_unused_columns=False,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    learning_rate=1e-4,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=50,
    warmup_ratio=0.1,
    logging_steps=10,
    metric_for_best_model="accuracy",
    report_to="tensorboard",
    load_best_model_at_end=True,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=processed_dataset["train"],
    eval_dataset=processed_dataset["validation"],
    processing_class=image_processor,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)


train_results = trainer.train()
trainer.save_model()
trainer.log_metrics("train", train_results.metrics)
trainer.save_metrics("train", train_results.metrics)
trainer.save_state()
