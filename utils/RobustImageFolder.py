class RobustImageFolder(Dataset):
    """健壮的图片数据集加载器，可以自动跳过损坏的图片"""
    
    def __init__(self, root, transform=None):
        self.root = Path(root)
        self.transform = transform
        self.samples = []
        self.class_to_idx = {}
        self.classes = []
        
        # 设置日志
        logging.basicConfig(
            filename=f'{self.root.name}_dataset_errors.log',
            level=logging.WARNING,
            format='%(asctime)s - %(message)s'
        )
        
        # 构建类别索引
        self.classes = sorted([d.name for d in self.root.iterdir() if d.is_dir()])
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}
        
        # 收集所有有效的图片
        self._collect_valid_samples()
        
        if len(self.samples) == 0:
            raise RuntimeError(f"在{root}中没有找到有效的图片")
            
    def _collect_valid_samples(self):
        """收集所有可用的图片样本，跳过损坏的文件"""
        valid_extensions = {'.jpg', '.jpeg', '.png'}
        skipped_files = 0
        total_files = 0
        
        # 首先计算总文件数
        for class_path in self.root.iterdir():
            if not class_path.is_dir():
                continue
            total_files += sum(1 for f in class_path.iterdir() if f.suffix.lower() in valid_extensions)
        
        # 使用tqdm显示进度
        pbar = tqdm(total=total_files, desc="验证图片文件")
        
        for class_path in self.root.iterdir():
            if not class_path.is_dir():
                continue
                
            class_idx = self.class_to_idx[class_path.name]
            
            for img_path in class_path.iterdir():
                if img_path.suffix.lower() not in valid_extensions:
                    continue
                    
                try:
                    # 尝试打开图片验证是否完整
                    with warnings.catch_warnings():
                        warnings.simplefilter("error")
                        with Image.open(img_path) as img:
                            img.verify()
                    # 再次打开以确保可以正确读取
                    with Image.open(img_path) as img:
                        img.load()
                    # 确保图片有正确的模式
                    if img.mode not in ['RGB', 'L']:
                        img = img.convert('RGB')
                        
                    self.samples.append((str(img_path), class_idx))
                    
                except Exception as e:
                    skipped_files += 1
                    logging.warning(f"跳过文件 {img_path}: {str(e)}")
                
                pbar.update(1)
        
        pbar.close()
        print(f"已加载 {len(self.samples)} 个有效图片，跳过 {skipped_files} 个无效文件")
        
    def __len__(self):
        return len(self.samples)
        
    def __getitem__(self, idx):
        img_path, class_idx = self.samples[idx]
        try:
            with Image.open(img_path) as img:
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                if self.transform is not None:
                    img = self.transform(img)
                return img, class_idx
        except Exception as e:
            # 如果在获取时出现问题，返回数据集中的下一个有效样本
            logging.error(f"加载 {img_path} 时出错: {str(e)}")
            new_idx = (idx + 1) % len(self)
            return self.__getitem__(new_idx)