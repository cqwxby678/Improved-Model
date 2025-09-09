import os
import argparse
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
from torchvision import transforms as T
from PIL import Image

import network
import utils


def get_argparser():
    """设置命令行参数解析器"""
    parser = argparse.ArgumentParser()

    # 数据输入选项
    parser.add_argument("--test_dir", type=str, required=True,
                        default='D:/111/test/images',
                        help="测试图像目录路径")

    # 模型选项
    parser.add_argument("--model", type=str, default='deeplabv3plus_mobilenet',
                        help="模型名称，例如 deeplabv3plus_mobilenet")
    parser.add_argument("--output_stride", type=int, default=16, choices=[8, 16],
                        help="模型输出步幅，可选 8 或 16")
    parser.add_argument("--num_classes", type=int, default=2,
                        help="类别数量，裂缝分割为二元分割")

    # 输出选项
    parser.add_argument("--save_dir", type=str, required=True,
                        help="保存预测结果的目录")
    parser.add_argument("--crop_size", type=int, default=513,
                        help="裁剪或调整图像的目标尺寸")
    parser.add_argument("--ckpt", type=str, required=True,
                        help="加载的检查点文件路径")
    parser.add_argument("--gpu_id", type=str, default='0',
                        help="使用的 GPU ID")

    return parser


def decode_target(prediction):
    """将预测结果解码为彩色图像
    0: 背景 -> 黑色 [0, 0, 0]
    1: 裂缝 -> 红色 [255, 0, 0]
    """
    # 创建全黑的RGB图像
    h, w = prediction.shape
    color_output = np.zeros((h, w, 3), dtype=np.uint8)

    # 将裂缝区域标记为红色
    color_output[prediction == 1] = [255, 0, 0]

    return color_output


def main():
    """主函数：执行图像分割推理"""
    # 解析命令行参数
    opts = get_argparser().parse_args()
    opts.num_classes = 2  # 固定为二元分割

    # 设置设备
    os.environ['CUDA_VISIBLE_DEVICES'] = opts.gpu_id
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用的设备: {device}")

    # 获取测试图像列表
    if not os.path.isdir(opts.test_dir):
        raise NotADirectoryError(f"测试目录不存在: {opts.test_dir}")

    # 支持的图像格式
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
    image_files = [
        os.path.join(opts.test_dir, f)
        for f in os.listdir(opts.test_dir)
        if os.path.splitext(f)[1].lower() in image_extensions
    ]

    if not image_files:
        raise FileNotFoundError(f"在目录 {opts.test_dir} 中未找到图像文件")
    print(f"找到 {len(image_files)} 张图像进行推理")

    # 初始化模型
    model = network.modeling.__dict__[opts.model](num_classes=opts.num_classes, output_stride=opts.output_stride)
    # 如果模型名包含 'plus' 且启用 separable_conv，则转换卷积层
    if 'plus' in opts.model and hasattr(opts, 'separable_conv') and opts.separable_conv:
        network.convert_to_separable_conv(model.classifier)
    utils.set_bn_momentum(model.backbone, momentum=0.01)  # 设置 BatchNorm 动量

    # 加载检查点
    if not os.path.isfile(opts.ckpt):
        raise FileNotFoundError(f"检查点文件不存在: {opts.ckpt}")

    checkpoint = torch.load(opts.ckpt, map_location=torch.device('cpu'))
    model.load_state_dict(checkpoint["model_state"])
    model = nn.DataParallel(model)  # 支持多 GPU
    model.to(device)
    print(f"从 {opts.ckpt} 加载模型")
    del checkpoint  # 释放内存

    # 创建保存结果的目录
    os.makedirs(opts.save_dir, exist_ok=True)
    print(f"预测结果将保存到: {opts.save_dir}")

    # 定义图像预处理变换
    transform = T.Compose([
        T.Resize((opts.crop_size, opts.crop_size)),  # 调整图像到指定尺寸
        T.ToTensor(),  # 转换为 Tensor
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # 归一化
    ])

    # 进入推理模式
    with torch.no_grad():
        model.eval()  # 设置模型为评估模式
        for img_path in tqdm(image_files, desc="处理图像"):
            # 获取图像文件名
            img_name = os.path.splitext(os.path.basename(img_path))[0]

            # 读取并预处理图像
            img = Image.open(img_path).convert('RGB')
            img_tensor = transform(img).unsqueeze(0)  # 转换为 NCHW 格式
            img_tensor = img_tensor.to(device)

            # 模型推理
            pred = model(img_tensor)
            pred = pred.max(1)[1].cpu().numpy()[0]  # 获取预测结果 (H, W)

            # 将预测结果转换为彩色图像
            colorized_pred = decode_target(pred)
            colorized_pred = Image.fromarray(colorized_pred)

            # 保存结果
            save_path = os.path.join(opts.save_dir, f'{img_name}_pred.png')
            colorized_pred.save(save_path)

    print("推理完成！所有预测结果已保存。")


if __name__ == '__main__':
    main()