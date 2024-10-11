import argparse
import time
from models.yolo import DetectionModel
import torch
import torch_pruning as tp
#####################
# 剪枝方法基于BN剪枝
####################
parser = argparse.ArgumentParser()
# 需要剪枝的文件
parser.add_argument('--source', type=str, required=True)
# 剪枝后存储路径
parser.add_argument('--target', type=str, default=None, required=False)
# 图像大小，用于构建样例数据
parser.add_argument('--image_size', type=int, default=640)
# 裁剪率
parser.add_argument('--pruning_ratio', type=float, default=0.5)
# 分类数量
parser.add_argument('--num_classes', type=int, default=1)
# 正则化
parser.add_argument('--norm', type=int, default=1)
args = parser.parse_args()
# 打印选择参数
print("#########################################")
for item, value in args.__dict__.items():
    print(item, value)
print("#########################################")
source_path = args.source
if args.target is None:
    target_path = args.source.replace('.pth', '_pruned.pth')
else:
    target_path = args.target


weights = torch.load(args.source, weights_only=False, map_location="cpu")


model = weights['model'].float().cpu() # type: DetectionModel
# 不打开梯度无法进行剪枝
for param in model.parameters():
    param.requires_grad = True

print(model)
# 采用的范数类型
imp = tp.importance.GroupNormImportance(args.norm)
# 忽略的层不会被剪枝
ignored_layers = []

for k, m in model.named_modules():
    if isinstance(m, torch.nn.Conv2d):
        # (5 * 3) + (num_classes * 3)
        if m.out_channels == ((5 + args.num_classes) * 3):
            ignored_layers.append(m)

print(model)
base_macs, base_nparms = tp.utils.count_ops_and_params(model, torch.randn(64, 3, args.image_size, args.image_size))
print("===================================================")

# 范数一般采用L1
# 基于BN的系数来决定权重的重要性，但是在训练时候需要使用L1正则化参数拉开不同通道下BN系数的差异
pruner = tp.pruner.BNScalePruner(
    model,
    torch.randn(64, 3, args.image_size, args.image_size),
    importance=imp,
    pruning_ratio=args.pruning_ratio,
    global_pruning=True,
    isomorphic=True,
    ignored_layers=ignored_layers
)
pruner.step()

# 打印网络结构
print(model)
print("===================================================")
macs, nparms = tp.utils.count_ops_and_params(model, torch.randn(64, 3, args.image_size, args.image_size))
print(f"MACs:{base_macs / 1e9}G -> {macs / 1e9}G, Parms: {base_nparms / 1e6}M -> {nparms / 1e6}M")
weights['model'] = model
torch.save(weights, 'best_prune.pt')
