import argparse
import torch
import datetime
import os
import yaml
import json

from main_model import CSDI_Forecasting
from dataset_forecasting import get_dataloader
from utils import train, evaluate

# ===== argparse 配置 =====
parser = argparse.ArgumentParser(description="CSDI Forecasting on Speed + Weather")
parser.add_argument("--config", type=str, default="base_forecasting.yaml")
parser.add_argument("--device", type=str, default="cuda:0")
parser.add_argument("--mask_mode", type=str, default="middle", choices=["front", "back", "middle"])
parser.add_argument("--nsample", type=int, default=100)
parser.add_argument("--modelfolder", type=str, default="")  # 若为""则训练，否则加载模型
args = parser.parse_args()

# ===== 读取配置文件并覆盖 mask 策略 =====
with open("config/" + args.config, "r") as f:
    config = yaml.safe_load(f)
config["model"]["mask_mode"] = args.mask_mode

# ===== 创建保存路径并记录 config =====
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
foldername = f"./save/speed_{args.mask_mode}_{timestamp}/"
os.makedirs(foldername, exist_ok=True)
with open(foldername + "config.json", "w") as f:
    json.dump(config, f, indent=4)

# ===== 加载数据（使用新的 get_dataloader）=====
train_loader, valid_loader, test_loader, scaler, mean_scaler = get_dataloader(
    data_path="./data/speed_data",
    device=args.device,
    batch_size=config["train"]["batch_size"],
    mask_strategy=args.mask_mode
)

# ===== 初始化模型（速度 + 天气 = 25 个特征）=====
target_dim = 25
model = CSDI_Forecasting(config, args.device, target_dim).to(args.device)

# ===== 模型训练 / 加载 =====
if args.modelfolder == "":
    train(model, config["train"], train_loader, valid_loader, foldername=foldername)
else:
    model_path = f"./save/{args.modelfolder}/model.pth"
    model.load_state_dict(torch.load(model_path))

# ===== 评估并保存结果 =====
evaluate(
    model=model,
    test_loader=test_loader,
    nsample=args.nsample,
    scaler=scaler,
    mean_scaler=mean_scaler,
    foldername=foldername,
)

# ✅ 打印每个数据集的样本数，确认切分是否正确
print(f"🧪 Train: {len(train_loader.dataset)} samples")
print(f"🧪 Valid: {len(valid_loader.dataset)} samples")
print(f"🧪 Test : {len(test_loader.dataset)} samples")