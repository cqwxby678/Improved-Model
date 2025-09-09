import os
import random
import argparse
import numpy as np
from tqdm import tqdm
import csv
import matplotlib.pyplot as plt
import time

import torch
import torch.nn as nn
from torchvision import transforms as T
from PIL import Image

import network
import utils
from datasets import CRACKSegmentation
from metrics import StreamSegMetrics


def get_argparser():
    """设置命令行参数解析器"""
    parser = argparse.ArgumentParser()

    # 数据输入选项
    parser.add_argument("--input", type=str, required=True,
                        default='D:/111/segmentionModel/DeepLabV3Plus-Pytorch-master/datasets/data',
                        help="CRACK500 数据集根目录路径，包含 imageSet 和 JPEGImages 文件夹")

    # 模型选项
    parser.add_argument("--model", type=str, default='deeplabv3plus_mobilenet',
                        help="模型名称，例如 deeplabv3plus_mobilenet")
    parser.add_argument("--output_stride", type=int, default=16, choices=[8, 16],
                        help="模型输出步幅，可选 8 或 16")
    parser.add_argument("--num_classes", type=int, default=2,
                        help="类别数量，CRACK500 为二元分割（裂缝 vs 背景）")

    # 推理和评估选项
    parser.add_argument("--crop_size", type=int, default=513,
                        help="裁剪或调整图像的目标尺寸")
    parser.add_argument("--ckpt", type=str, required=True,
                        help="加载的检查点文件路径，必须指定")
    parser.add_argument("--gpu_id", type=str, default='0',
                        help="使用的 GPU ID")
    parser.add_argument("--csv_output", type=str, default='test_metrics.csv',
                        help="保存指标的 CSV 文件名（将存放在 testResult 中）")
    parser.add_argument("--plot_output", type=str, default ='test_metrics_plot.png',
                        help="保存指标图表的 PNG 文件名（将存放在 testResult 中）")
    parser.add_argument("--save_val_results_to", type=str, default=None,
                            help="保存分割预测图的目录路径")

    return parser


def main():
    """主函数：对测试集图像进行分割推理并评估指标，包括 FPS"""
    # 解析命令行参数
    opts = get_argparser().parse_args()
    opts.num_classes = 2  # 固定为 CRACK500 的二元分割

    # 设置设备
    os.environ['CUDA_VISIBLE_DEVICES'] = opts.gpu_id
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用的设备: {device}")

    # 创建 testResult 目录
    save_dir = 'datasets/testResult'
    os.makedirs(save_dir, exist_ok=True)

    # 从 test.txt 读取测试图像文件名
    test_file = os.path.join(opts.input, 'imageSet', 'Segmentation', 'test.txt')
    if not os.path.isfile(test_file):
        raise FileNotFoundError(f"未找到 test.txt 文件: {test_file}")

    with open(test_file, 'r') as f:
        file_names = [x.strip() for x in f.readlines() if x.strip()]

    # 加载图像和掩码路径
    image_dir = os.path.join(opts.input, 'JPEGImages')
    mask_dir = os.path.join(opts.input, 'SegmentationClass')
    image_files = [os.path.join(image_dir, f"{name}.jpg") for name in file_names]
    mask_files = [os.path.join(mask_dir, f"{name}.png") for name in file_names]

    # 检查文件是否存在
    missing_images = [f for f in image_files if not os.path.isfile(f)]
    missing_masks = [f for f in mask_files if not os.path.isfile(f)]
    if missing_images:
        raise FileNotFoundError(f"以下图像文件不存在: {missing_images[:5]} (共 {len(missing_images)} 个)")
    if missing_masks:
        raise FileNotFoundError(f"以下掩码文件不存在: {missing_masks[:5]} (共 {len(missing_masks)} 个)")
    print(f"从 test.txt 找到 {len(image_files)} 张图像进行推理和评估")

    # 初始化模型
    model = network.modeling.__dict__[opts.model](num_classes=opts.num_classes, output_stride=opts.output_stride)
    if 'plus' in opts.model and hasattr(opts, 'separable_conv') and opts.separable_conv:
        network.convert_to_separable_conv(model.classifier)
    utils.set_bn_momentum(model.backbone, momentum=0.01)

    # 加载检查点
    if not os.path.isfile(opts.ckpt):
        raise FileNotFoundError(f"检查点文件不存在: {opts.ckpt}")
    checkpoint = torch.load(opts.ckpt, map_location=torch.device('cpu'))
    model.load_state_dict(checkpoint["model_state"])
    model = nn.DataParallel(model)
    model.to(device)
    print(f"从 {opts.ckpt} 加载模型")

    # 定义图像和掩码预处理变换
    transform = T.Compose([
        T.Resize((opts.crop_size, opts.crop_size)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    mask_transform = T.Compose([
        T.Resize((opts.crop_size, opts.crop_size)),
        T.ToTensor(),
    ])

    # 初始化评价指标
    metrics = StreamSegMetrics(opts.num_classes)
    metrics.reset()

    # 存储每张图片的指标（包括 FPS）
    image_metrics = {
        'image': [],
        'overall_acc': [],
        'mean_acc': [],
        'mean_iou': [],
        'fps': [],

    }

    # 进入推理和评估模式
    with torch.no_grad():
        model.eval()
        for img_path, mask_path in tqdm(zip(image_files, mask_files), desc="评估测试集"):
            # 获取图像文件名
            img_name = os.path.basename(img_path).split('.')[0]

            # 读取并预处理图像和掩码
            img = Image.open(img_path).convert('RGB')
            mask = Image.open(mask_path).convert('L')
            img_tensor = transform(img).unsqueeze(0).to(device)
            mask_tensor = mask_transform(mask).squeeze(0).to(device, dtype=torch.long)

            # 测量推理时间
            start_time = time.time()
            pred = model(img_tensor).max(1)[1].cpu().numpy()[0]  # 推理
            end_time = time.time()
            inference_time = end_time - start_time
            fps = 1.0 / inference_time if inference_time > 0 else float('inf')  # 计算 FPS

            # 获取目标掩码
            target = mask_tensor.cpu().numpy()

            # 更新指标
            metrics.update(target[None], pred[None])
            batch_score = metrics.get_results()

            # 记录当前图像的指标
            image_metrics['image'].append(img_name)
            image_metrics['overall_acc'].append(batch_score['Overall Acc'])
            image_metrics['mean_acc'].append(batch_score['Mean Acc'])
            image_metrics['mean_iou'].append(batch_score['Mean IoU'])
            image_metrics['fps'].append(fps)

    # 保存指标到 CSV 文件
    csv_path = os.path.join(save_dir, opts.csv_output)
    with open(csv_path, 'w', newline='') as csvfile:
        fieldnames = ['Image', 'Overall Accuracy', 'Mean Accuracy', 'Mean IoU', 'FPS']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for i in range(len(image_metrics['image'])):
            writer.writerow({
                'Image': image_metrics['image'][i],
                'Overall Accuracy': image_metrics['overall_acc'][i],
                'Mean Accuracy': image_metrics['mean_acc'][i],
                'Mean IoU': image_metrics['mean_iou'][i],
                'FPS': image_metrics['fps'][i]
            })
    print(f"指标已保存至: {csv_path}")

    # 绘制指标折线图（包括 FPS）
    plt.figure(figsize=(12, 6))
    plt.plot(range(1, len(image_metrics['image']) + 1), image_metrics['overall_acc'], label='Overall Accuracy', marker='o')
    plt.plot(range(1, len(image_metrics['image']) + 1), image_metrics['mean_acc'], label='Mean Accuracy', marker='s')
    plt.plot(range(1, len(image_metrics['image']) + 1), image_metrics['mean_iou'], label='Mean IoU', marker='^')
    plt.plot(range(1, len(image_metrics['image']) + 1), image_metrics['fps'], label='FPS', marker='x')
    plt.xlabel('Image Index')
    plt.ylabel('Metric Value / FPS')
    plt.title('Evaluation Metrics and FPS per Image on Test Set')
    plt.legend()
    plt.grid(True)
    plot_path = os.path.join(save_dir, opts.plot_output)
    plt.savefig(plot_path)
    plt.close()
    print(f"指标图表已保存至: {plot_path}")

    # 打印最终测试集指标和平均 FPS
    final_score = metrics.get_results()
    avg_fps = np.mean(image_metrics['fps'])
    print("测试集最终评价指标:")
    print(metrics.to_str(final_score))
    print(f"平均 FPS: {avg_fps:.2f}")


if __name__ == '__main__':
    main()
