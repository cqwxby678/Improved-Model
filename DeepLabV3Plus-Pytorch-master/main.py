import os
import random
import argparse
import numpy as np
from tqdm import tqdm
import csv
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils import data

from datasets import CRACKSegmentation
from utils import ext_transforms as et
from metrics import StreamSegMetrics
import network
import utils
from thop import profile
from thop import clever_format


def get_argparser():
    parser = argparse.ArgumentParser()
    # 数据集选项
    parser.add_argument("--data_root", type=str,
                        default='D:/111/segmentionModel/DeepLabV3Plus-Pytorch-master/datasets/data',
                        help="CRACK500数据集的路径")
    parser.add_argument("--num_classes", type=int, default=2,
                        help="类别数量")

    # 模型选项
    parser.add_argument("--model", type=str, default='deeplabv3plus_mobilenet',
                        help='模型名称')
    parser.add_argument("--output_stride", type=int, default=16, choices=[8, 16])

    # 训练选项
    parser.add_argument("--total_epochs", type=int, default=200,
                        help="总轮次数 (默认: 100)")
    parser.add_argument("--lr", type=float, default=0.01,
                        help="学习率 (默认: 0.01)")
    parser.add_argument("--batch_size", type=int, default=4,
                        help='批次大小 (默认: 4)')
    parser.add_argument("--val_batch_size", type=int, default=4,
                        help='验证批次大小 (默认: 4)')
    parser.add_argument("--crop_size", type=int, default=513)
    parser.add_argument("--ckpt", default=None, type=str,
                        help="从检查点恢复模型")
    parser.add_argument("--gpu_id", type=str, default='0',
                        help="GPU ID")
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--random_seed", type=int, default=1)
    parser.add_argument("--print_interval", type=int, default=10)
    parser.add_argument("--val_interval", type=int, default=5)
    parser.add_argument("--train_csv", type=str, default='train_metrics.csv',
                        help="保存训练指标的 CSV 文件路径")
    parser.add_argument("--val_csv", type=str, default='val_metrics.csv',
                        help="保存验证指标的 CSV 文件路径")
    parser.add_argument("--train_plot", type=str, default='train_loss_plot.png',
                        help="保存训练损失图表的 PNG 文件路径")
    parser.add_argument("--val_plot", type=str, default='val_metrics_plot.png',
                        help="保存验证指标图表的 PNG 文件路径")

    return parser


def get_dataset(opts):
    train_transform = et.ExtCompose([
        et.ExtRandomScale((0.5, 2.0)),
        et.ExtRandomCrop(size=(opts.crop_size, opts.crop_size), pad_if_needed=True),
        et.ExtRandomHorizontalFlip(),
        et.ExtToTensor(),
        et.ExtNormalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    val_transform = et.ExtCompose([
        et.ExtResize((opts.crop_size, opts.crop_size)),
        et.ExtToTensor(),
        et.ExtNormalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    train_dst = CRACKSegmentation(root=opts.data_root, image_set='train', transform=train_transform)
    val_dst = CRACKSegmentation(root=opts.data_root, image_set='val', transform=val_transform)
    return train_dst, val_dst

def print_model_parameters(model):
    """打印模型的总参数量和可训练参数量"""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"模型参数总量: {total_params:,}")
    print(f"可训练参数量: {trainable_params:,}")

def main():
    opts = get_argparser().parse_args()
    opts.num_classes = 2

    # 设置 v3PLUStrainResult 目录
    train_result_dir = os.path.join(os.path.dirname(opts.data_root), 'trainResult')
    os.makedirs(train_result_dir, exist_ok=True)

    # 设置设备
    os.environ['CUDA_VISIBLE_DEVICES'] = opts.gpu_id
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用的设备: {device}")

    # 设置随机种子
    torch.manual_seed(opts.random_seed)
    np.random.seed(opts.random_seed)
    random.seed(opts.random_seed)

    # 设置数据加载器
    train_dst, val_dst = get_dataset(opts)
    train_loader = data.DataLoader(
        train_dst, batch_size=opts.batch_size, shuffle=True, num_workers=2, drop_last=True)
    val_loader = data.DataLoader(
        val_dst, batch_size=opts.val_batch_size, shuffle=False, num_workers=2)
    print(f"数据集: CRACK500, 训练集大小: {len(train_dst)}, 验证集大小: {len(val_dst)}")

    # 设置模型
    model = network.modeling.__dict__[opts.model](num_classes=opts.num_classes, output_stride=opts.output_stride)
    model = nn.DataParallel(model)
    model.to(device)
    # 打印模型参数量
    print_model_parameters(model)
    # 设置评估指标
    metrics = StreamSegMetrics(opts.num_classes)

    # 设置优化器和学习率调度器
    optimizer = torch.optim.SGD(params=[
        {'params': model.module.backbone.parameters(), 'lr': 0.1 * opts.lr},
        {'params': model.module.classifier.parameters(), 'lr': opts.lr},
    ], lr=opts.lr, momentum=0.9, weight_decay=opts.weight_decay)
    scheduler = utils.PolyLR(optimizer, max_iters=opts.total_epochs * len(train_loader), power=0.9)

    # 设置损失函数
    criterion = nn.CrossEntropyLoss(ignore_index=255, reduction='mean')

    # 存储训练和验证指标
    train_metrics = {'epoch': [], 'loss': []}
    val_metrics = {'epoch': [], 'overall_acc': [], 'mean_acc': [], 'mean_iou': []}

    def save_ckpt(path):
        torch.save({
            "cur_epochs": cur_epochs,
            "model_state": model.module.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict(),
            "best_score": best_score,
        }, path)
        print(f"模型保存为: {path}")

    # 如果有检查点则加载
    best_score = 0.0
    cur_epochs = 0
    if opts.ckpt and os.path.isfile(opts.ckpt):
        checkpoint = torch.load(opts.ckpt, map_location=device)
        model.module.load_state_dict(checkpoint["model_state"])
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        scheduler.load_state_dict(checkpoint["scheduler_state"])
        cur_epochs = checkpoint["cur_epochs"]
        best_score = checkpoint["best_score"]
        print(f"从 {opts.ckpt} 恢复训练状态")
    else:
        print("开始全新训练")

    # 训练循环
    while cur_epochs < opts.total_epochs:
        model.train()
        cur_epochs += 1
        epoch_loss = 0
        num_batches = 0

        for i, (images, labels) in enumerate(train_loader):
            images = images.to(device, dtype=torch.float32)
            labels = labels.to(device, dtype=torch.long)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            np_loss = loss.detach().cpu().numpy()
            epoch_loss += np_loss
            num_batches += 1

            if (i + 1) % opts.print_interval == 0:
                interval_loss = epoch_loss / num_batches
                print(f"轮次 {cur_epochs}/{opts.total_epochs}, 批次 {i+1}/{len(train_loader)}, 平均损失={interval_loss:.4f}")

        # 记录每轮训练的平均损失
        avg_loss = epoch_loss / num_batches
        train_metrics['epoch'].append(cur_epochs)
        train_metrics['loss'].append(avg_loss)
        print(f"轮次 {cur_epochs}/{opts.total_epochs} 完成，平均损失: {avg_loss:.4f}")

        # 验证
        if cur_epochs % opts.val_interval == 0:
            latest_ckpt_path = os.path.join(train_result_dir, f'latest_{opts.model}_crack500_os{opts.output_stride}.pth')
            save_ckpt(latest_ckpt_path)
            print("验证中...")
            model.eval()
            metrics.reset()
            with torch.no_grad():
                for images, labels in tqdm(val_loader):
                    images = images.to(device, dtype=torch.float32)
                    labels = labels.to(device, dtype=torch.long)
                    outputs = model(images)
                    preds = outputs.detach().max(dim=1)[1].cpu().numpy()
                    targets = labels.cpu().numpy()
                    metrics.update(targets, preds)
            val_score = metrics.get_results()
            print(metrics.to_str(val_score))

            # 记录验证指标
            val_metrics['epoch'].append(cur_epochs)
            val_metrics['overall_acc'].append(val_score['Overall Acc'])
            val_metrics['mean_acc'].append(val_score['Mean Acc'])
            val_metrics['mean_iou'].append(val_score['Mean IoU'])

            if val_score['Mean IoU'] > best_score:
                best_score = val_score['Mean IoU']
                best_ckpt_path = os.path.join(train_result_dir, f'best_{opts.model}_crack500_os{opts.output_stride}.pth')
                save_ckpt(best_ckpt_path)

            model.train()

        scheduler.step()

    # 保存训练指标到 CSV
    train_csv_path = os.path.join(train_result_dir, opts.train_csv)
    with open(train_csv_path, 'w', newline='') as csvfile:
        fieldnames = ['Epoch', 'Average Loss']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for i in range(len(train_metrics['epoch'])):
            writer.writerow({
                'Epoch': train_metrics['epoch'][i],
                'Average Loss': train_metrics['loss'][i]
            })
    print(f"训练指标已保存至: {train_csv_path}")

    # 保存验证指标到 CSV
    val_csv_path = os.path.join(train_result_dir, opts.val_csv)
    with open(val_csv_path, 'w', newline='') as csvfile:
        fieldnames = ['Epoch', 'Overall Accuracy', 'Mean Accuracy', 'Mean IoU']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for i in range(len(val_metrics['epoch'])):
            writer.writerow({
                'Epoch': val_metrics['epoch'][i],
                'Overall Accuracy': val_metrics['overall_acc'][i],
                'Mean Accuracy': val_metrics['mean_acc'][i],
                'Mean IoU': val_metrics['mean_iou'][i]
            })
    print(f"验证指标已保存至: {val_csv_path}")

    # 绘制训练损失图表
    train_plot_path = os.path.join(train_result_dir, opts.train_plot)
    plt.figure(figsize=(10, 6))
    plt.plot(train_metrics['epoch'], train_metrics['loss'], label='Average Loss', marker='o')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss per Epoch')
    plt.legend()
    plt.grid(True)
    plt.savefig(train_plot_path)
    plt.close()
    print(f"训练损失图表已保存至: {train_plot_path}")

    # 绘制验证指标图表
    val_plot_path = os.path.join(train_result_dir, opts.val_plot)
    plt.figure(figsize=(10, 6))
    plt.plot(val_metrics['epoch'], val_metrics['overall_acc'], label='Overall Accuracy', marker='o')
    plt.plot(val_metrics['epoch'], val_metrics['mean_acc'], label='Mean Accuracy', marker='s')
    plt.plot(val_metrics['epoch'], val_metrics['mean_iou'], label='Mean IoU', marker='^')
    plt.xlabel('Epoch')
    plt.ylabel('Metric Value')
    plt.title('Validation Metrics per Epoch')
    plt.legend()
    plt.grid(True)
    plt.savefig(val_plot_path)
    plt.close()
    print(f"验证指标图表已保存至: {val_plot_path}")


if __name__ == '__main__':
    main()