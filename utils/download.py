import os
import zipfile
import re

from huggingface_hub import HfApi, hf_hub_download


def extract_string(s):
    # 使用正则表达式匹配所需的模式
    match = re.search(r"/(.+)_touhou$", s)
    if match:
        # 如果匹配成功，返回匹配到的第一个分组
        return match.group(1)
    else:
        # 如果没有匹配到，返回None或者适当的提示信息
        return None


api = HfApi()
datasets = api.list_datasets(author="CyberHarem", search="touhou")
downloaded = os.listdir("data/touhou")
download_count = 0
for dataset in datasets:
    # print(dataset)
    repo_id = dataset.id
    name = extract_string(dataset.id)
    if not name or name in downloaded:
        print(f"Skipping {name}")
        continue

    # download raw archive file
    print(f"Downloading {name}...")
    try:
        zip_file = hf_hub_download(
            repo_id=repo_id,
            repo_type="dataset",
            filename="dataset-raw.zip",
        )
    except Exception as e:
        print(f"Error downloading {name}: {e}")
        continue
    print(f"Downloaded {name}")

    # extract files to your directory
    print(f"Extracting {name}...")
    dataset_dir = f"data/touhou/{name}"
    os.makedirs(dataset_dir, exist_ok=True)
    with zipfile.ZipFile(zip_file, "r") as zf:
        zf.extractall(dataset_dir)

    download_count += 1

# print(f"Total datasets: {len(datasets)}")
print(f"Before download: {len(downloaded)}")
print(f"After download: {len(downloaded) + download_count}")
print(f"Downloaded {download_count} datasets.")

downloaded = os.listdir("data/touhou")
print(f"Downloaded datasets: {downloaded}")
