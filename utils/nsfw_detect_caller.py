import os

def call_script_with_folders(script_name):
    """
    以当前目录中所有一级文件夹名称作为参数，多次调用指定的 Python 脚本
    参数:
        script_name: 要调用的 Python 脚本文件名（例如 "target.py"）
    """
    # 确保目标脚本存在
    if not os.path.isfile(script_name):
        print(f"错误: 目标脚本 {script_name} 不存在")
        return
    
    # 获取当前目录中的一级文件夹
    folders = [item for item in os.listdir('.') if os.path.isdir(item)]
    
    if not folders:
        print("当前目录中没有文件夹")
        return
    
    print(f"找到的文件夹: {folders}")
    
    # 对每个文件夹名称调用目标脚本
    for folder in folders:
        if folder == 'nsfw_buffer' or folder == 'seiran':
            print(f"==== 已跳过 `{folder}`. ====")
            continue
    
        try:
            print(f"\n调用 {script_name}，参数: {folder}")
            # 调用外部 Python 脚本
            os.system('py {} {}'.format(script_name, folder))

        except Exception as e:
            print(f"调用 {script_name} 时出错: {e}")

def main():
    # 指定要调用的目标脚本名称
    target_script = "nsfw_detect_v2.py"  # 可替换为你的实际脚本名
    
    print(f"开始调用 {target_script}...")
    call_script_with_folders(target_script)
    print("调用完成！")

if __name__ == "__main__":
    main()