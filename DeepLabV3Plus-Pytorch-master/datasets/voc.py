import os
import torch.utils.data as data
import numpy as np
from PIL import Image


def voc_cmap(N=256, normalized=False):
    def bitget(byteval, idx):
        return ((byteval & (1 << idx)) != 0)

    dtype = 'float32' if normalized else 'uint8'
    cmap = np.zeros((N, 3), dtype=dtype)
    for i in range(N):
        r = g = b = 0
        c = i
        for j in range(8):
            r = r | (bitget(c, 0) << 7 - j)
            g = g | (bitget(c, 1) << 7 - j)
            b = b | (bitget(c, 2) << 7 - j)
            c = c >> 3

        cmap[i] = np.array([r, g, b])

    cmap = cmap / 255 if normalized else cmap
    return cmap


class CRACKSegmentation(data.Dataset):
    """CRACK500 Segmentation Dataset.
    Args:
        root (string): Root directory of the dataset (包含CRACK500文件夹的目录)
        image_set (string): Select the image_set to use, ``train``, ``val`` or ``test``
        transform (callable, optional): A function/transform that takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
    """
    cmap = voc_cmap()

    def __init__(self,
                 root,
                 image_set='train',
                 transform=None):

        self.root = os.path.expanduser(root)
        self.transform = transform

        # 定义数据集路径
        dataset_root = os.path.join(self.root, 'QCrackSeg')
        image_dir = os.path.join(dataset_root, 'JPEGImages')
        mask_dir = os.path.join(dataset_root, 'SegmentationClass')
        split_dir = os.path.join(dataset_root, 'imageSet', 'Segmentation')

        # 检查目录是否存在
        if not os.path.isdir(dataset_root):
            raise RuntimeError(f'Dataset not found at {dataset_root}')

        # 获取划分文件路径
        split_file = os.path.join(split_dir, f'{image_set}.txt')
        if not os.path.exists(split_file):
            raise ValueError(f'Split file not found: {split_file}')

        # 读取文件列表
        with open(split_file, "r") as f:
            file_names = [x.strip() for x in f.readlines()]

        # 构建图片和mask路径
        self.images = [os.path.join(image_dir, x + ".jpg") for x in file_names]
        # 处理_mask后缀的mask文件
        self.masks = [os.path.join(mask_dir, x + ".png") for x in file_names]

        # 验证文件是否存在
        assert len(self.images) == len(self.masks), "图片和标注数量不匹配"
        for img_path, mask_path in zip(self.images, self.masks):
            if not os.path.exists(img_path):
                raise FileNotFoundError(f"图片不存在: {img_path}")
            if not os.path.exists(mask_path):
                raise FileNotFoundError(f"标注不存在: {mask_path}")

    def __getitem__(self, index):
        img = Image.open(self.images[index]).convert('RGB')
        target = Image.open(self.masks[index]).convert('L')  # 转换为灰度

        # 关键修复：将像素值255转换为1
        target = np.array(target)
        target = np.where(target > 0, 1, target)  # 所有非零值设为1
        target = Image.fromarray(target.astype(np.uint8))

        if self.transform is not None:
            img, target = self.transform(img, target)

        return img, target

    def __len__(self):
        return len(self.images)

    @classmethod
    def decode_target(cls, mask):
        """decode semantic mask to RGB image"""
        return cls.cmap[mask]