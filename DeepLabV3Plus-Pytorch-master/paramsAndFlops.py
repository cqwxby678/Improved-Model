import os
import argparse
import torch
import torch.nn as nn
from thop import profile
from thop import clever_format

# 假设你的network模块在相同目录下
import network


def get_argparser():
    parser = argparse.ArgumentParser(description='计算模型FLOPs和参数量')

    # 模型选项
    parser.add_argument("--model", type=str, default='deeplabv3plus_mobilenet',
                        help='模型名称')
    parser.add_argument("--output_stride", type=int, default=16, choices=[8, 16])
    parser.add_argument("--num_classes", type=int, default=2,
                        help="类别数量")
    parser.add_argument("--input_size", type=int, nargs=2, default=[513, 513],
                        help="输入尺寸 [高度, 宽度]")
    parser.add_argument("--ckpt", type=str, default=None,
                        help="模型检查点路径 (可选)")
    parser.add_argument("--gpu_id", type=str, default='0',
                        help="GPU ID")

    return parser


def calculate_flops(model, input_size):
    """计算模型的FLOPs和参数数量"""
    model.eval()
    device = next(model.parameters()).device

    # 创建随机输入张量 (batch_size=1, channels=3)
    dummy_input = torch.randn(1, 3, input_size[0], input_size[1]).to(device)

    # 计算FLOPs和参数量
    macs, params = profile(model, inputs=(dummy_input,), verbose=False)
    macs, params = clever_format([macs, params], "%.3f")

    return macs, params


def print_model_parameters(model):
    """打印模型的总参数量和可训练参数量"""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"模型参数总量: {total_params:,}")
    print(f"可训练参数量: {trainable_params:,}")
    print(f"不可训练参数量: {total_params - trainable_params:,}")


def main():
    opts = get_argparser().parse_args()

    # 设置设备
    os.environ['CUDA_VISIBLE_DEVICES'] = opts.gpu_id
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用的设备: {device}")

    # 创建模型
    print(f"创建模型: {opts.model}, 输出步长: {opts.output_stride}, 类别数: {opts.num_classes}")
    model = network.modeling.__dict__[opts.model](
        num_classes=opts.num_classes,
        output_stride=opts.output_stride
    )

    # 如果提供了检查点，则加载权重
    if opts.ckpt and os.path.isfile(opts.ckpt):
        print(f"从检查点加载权重: {opts.ckpt}")
        checkpoint = torch.load(opts.ckpt, map_location=device)

        # 处理可能存在的DataParallel包装
        if 'model_state' in checkpoint:
            state_dict = checkpoint['model_state']
        else:
            state_dict = checkpoint

        # 移除可能的"module."前缀（如果是DataParallel保存的）
        new_state_dict = {}
        for k, v in state_dict.items():
            name = k.replace("module.", "")  # 移除"module."前缀
            new_state_dict[name] = v

        model.load_state_dict(new_state_dict)

    model.to(device)

    # 打印模型参数信息
    print("\n===== 模型参数信息 =====")
    print_model_parameters(model)

    # 计算并打印FLOPs
    print("\n===== 计算FLOPs =====")
    print(f"输入尺寸: {opts.input_size[0]}x{opts.input_size[1]}")
    macs, params = calculate_flops(model, opts.input_size)

    print("\n===== 计算结果 =====")
    print(f"模型: {opts.model}")
    print(f"输入尺寸: {opts.input_size[0]}x{opts.input_size[1]}")
    print(f"FLOPs: {macs}")
    print(f"参数量: {params}")

    # 保存结果到文件
    result_file = f"{opts.model}_flops_results.txt"
    with open(result_file, 'w') as f:
        f.write(f"模型: {opts.model}\n")
        f.write(f"输入尺寸: {opts.input_size[0]}x{opts.input_size[1]}\n")
        f.write(f"FLOPs: {macs}\n")
        f.write(f"参数量: {params}\n")
        f.write(f"总参数: {sum(p.numel() for p in model.parameters()):,}\n")
        f.write(f"可训练参数: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}\n")

    print(f"\n结果已保存到: {result_file}")


if __name__ == '__main__':
    main()