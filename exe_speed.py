# import argparse
# import torch
# import datetime
# import json
# import yaml
# import os
# import sys

# from main_model import CSDI_Speed   # 确保 main_model.py 中定义了新的 CSDI_Speed(config, device)
# from dataset_speed import get_dataloader  # 或者你存成 dataset_speed.py
# from utils import train, evaluate

# #保存训练 验证和测试数据
# def save_dataset_info(data_loader, file_path, info_type="filenames"):
#     with open(file_path, "w") as f:
#         for i, data in enumerate(data_loader.dataset):
#             if info_type == "filenames":
#                 # Assuming each data item has a 'filename' attribute or similar
#                 filename = data.get('filename', f"Sample {i}")
#                 f.write(f"{filename}\n")
#             elif info_type == "data":
#                 # Assuming each data item is a tuple or dict with the original data
#                 f.write(f"{data}\n")


# if __name__ == "__main__":
#     parser = argparse.ArgumentParser(description="CSDI Speed+Weather")
#     parser.add_argument("--config", type=str, default="base.yaml",
#                         help="YAML config under config/")    
#     parser.add_argument('--device', default='cuda:0', help='Device for training')
#     parser.add_argument("--seed", type=int, default=1)
#     parser.add_argument("--testmissingratio", type=float, default=0.1)
#     parser.add_argument("--nfold", type=int, default=0,
#                         help="for 5-fold test (valid: 0–4)")
#     parser.add_argument("--unconditional", action="store_true")
#     parser.add_argument("--modelfolder", type=str, default="",
#                         help="如果要加载已有模型，填这里的子目录名")
#     parser.add_argument("--nsample", type=int, default=100)
#     parser.add_argument("--data_folder", type=str, default="./data/speed_data",
#                         help="CSV 数据所在文件夹")

#     args = parser.parse_args()
#     print("Args:", args)

#     # 读取 YAML 配置
#     cfg_path = os.path.join("config", args.config)
#     with open(cfg_path, "r") as f:
#         config = yaml.safe_load(f)

#     # 更新 config
#     config["model"]["is_unconditional"] = bool(args.unconditional)
#     config["model"]["test_missing_ratio"] = args.testmissingratio

#     print("Config:")
#     print(json.dumps(config, indent=4))

#     # 创建结果目录
#     now = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
#     foldername = f"./save/speed_fold{args.nfold}_seed{args.seed}_{now}/"
#     os.makedirs(foldername, exist_ok=True)
#     with open(os.path.join(foldername, "config.json"), "w") as f:
#         json.dump(config, f, indent=4)
#     print("Model folder:", foldername)

#     # 准备 DataLoader
#     train_loader, valid_loader, test_loader = get_dataloader(
#         folder=args.data_folder,
#         seed=args.seed,
#         nfold=args.nfold,
#         batch_size=config["train"]["batch_size"],
#         missing_ratio=config["model"]["test_missing_ratio"],
#     )

#     print(type(train_loader))
#     print(111111111111111111111111111)
#     print(train_loader)
#     #a1,b1,c1,d,e,f,g,h,aa,bb = train_loader
#     #a2,b2,c2 = valid_loader
#     #a3,b3,c3 = test_loader

#     #print(a1.keys())
#     #print(a1['observed_data'][0])
#     #print(222222222222222222222222222222222222222222222)
#     #print(b1['observed_data'][0])

#     #print(c1['observed_data'][0])
# #    print(a2)
# #    print(a3)



    
#     print(111111111111111111111111111)
    
#     print(f"→ Dataset sizes: train={len(train_loader.dataset)}, "
#           f"val={len(valid_loader.dataset)}, test={len(test_loader.dataset)}")

#     save_dataset_info(train_loader, "train_dataset_info.txt", info_type="filenames")
#     save_dataset_info(valid_loader, "valid_dataset_info.txt", info_type="filenames")
#     save_dataset_info(test_loader, "test_dataset_info.txt", info_type="filenames")
#     print("Dataset saved to dataset_sizes.txt")

#     #print(f"Missing Ratio: {missing_ratio}")

#     # 初始化模型（不再传 target_dim）
#     model = CSDI_Speed(config, args.device).to(args.device)
#     print("Learning Rate:", config["train"]["lr"])

#     # 训练或加载
#     if args.modelfolder == "":
#         train(
#             model,
#             config,
#             train_loader,
#             valid_loader=valid_loader,
#             foldername=foldername,
#         )
#     else:
#         ckpt = torch.load(os.path.join("./save", args.modelfolder, "model.pth"))
#         model.load_state_dict(ckpt)
#         print(f"Loaded checkpoint from save/{args.modelfolder}/model.pth")

#     # 获取 Speed 的归一化参数（假设 dataset 定义了 get_scalers）
#     mean_vec, std_vec     = train_loader.dataset.get_scalers()
#     mean_speed, std_speed = mean_vec[0], std_vec[0]
#     print("Scaler for speed:", mean_speed, std_speed)

#     # 调用 evaluate 时，使用新的参数名
#     evaluate(
#         model,
#         test_loader,
#         nsample=args.nsample,
#         scaler=std_speed,       # 对应你代码里的 scaler_vec
#         mean_scaler=mean_speed, # 对应你代码里的 mean_scaler_vec
#         foldername=foldername,
#     )

import argparse
import torch
import datetime
import json
import yaml
import os
import sys

from main_model import CSDI_Speed
from dataset_speed import get_dataloader
from utils import train, evaluate

def save_dataset_info(data_loader, file_path, info_type="filenames"):
    with open(file_path, "w") as f:
        for i, data in enumerate(data_loader.dataset):
            if info_type == "filenames":
                filename = data.get('filename', f"Sample {i}")
                f.write(f"{filename}\n")
            elif info_type == "data":
                f.write(f"{data}\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CSDI Speed+Weather")
    parser.add_argument("--config", type=str, default="base.yaml", help="YAML config path or '-' to read from stdin")
    parser.add_argument("--device", default="cuda:0", help="Device for training")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--testmissingratio", type=float, default=0.1)
    parser.add_argument("--nfold", type=int, default=0)
    parser.add_argument("--unconditional", action="store_true")
    parser.add_argument("--modelfolder", type=str, default="")
    parser.add_argument("--nsample", type=int, default=100)
    parser.add_argument("--data_folder", type=str, default="./data/speed_data")

    args = parser.parse_args()

    # ✅ 读取 config：支持 stdin 或 YAML 文件
    if args.config == "-":
        config = json.loads(sys.stdin.read())
    else:
        cfg_path = os.path.join("config", args.config)
        with open(cfg_path, "r") as f:
            config = yaml.safe_load(f)

    config["model"]["is_unconditional"] = bool(args.unconditional)
    config["model"]["test_missing_ratio"] = args.testmissingratio

    # ✅ 结果保存路径
    foldername = config.get("save_folder")
    if not foldername:
        now = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        foldername = f"./save/speed_fold{args.nfold}_seed{args.seed}_{now}/"
    os.makedirs(foldername, exist_ok=True)

    with open(os.path.join(foldername, "config.json"), "w") as f:
        json.dump(config, f, indent=4)

    print(f"\n📂 Model folder: {foldername}")
    print("📋 Training config:")
    print(json.dumps(config, indent=2))

    # ✅ 数据加载
    train_loader, valid_loader, test_loader = get_dataloader(
        folder=args.data_folder,
        seed=args.seed,
        nfold=args.nfold,
        batch_size=config["train"]["batch_size"],
        missing_ratio=config["model"]["test_missing_ratio"],
    )

    print(f"\n📊 Dataset sizes: train={len(train_loader.dataset)}, "
          f"val={len(valid_loader.dataset)}, test={len(test_loader.dataset)}")

    # ✅ 保存数据划分信息
    save_dataset_info(train_loader, os.path.join(foldername, "train_dataset.txt"))
    save_dataset_info(valid_loader, os.path.join(foldername, "valid_dataset.txt"))
    save_dataset_info(test_loader, os.path.join(foldername, "test_dataset.txt"))

    # ✅ 初始化模型并训练
    model = CSDI_Speed(config, args.device).to(args.device)
    if args.modelfolder == "":
        train(model, config, train_loader, valid_loader, foldername=foldername)
    else:
        ckpt = torch.load(os.path.join("./save", args.modelfolder, "model.pth"))
        model.load_state_dict(ckpt)
        print(f"✅ Loaded checkpoint from save/{args.modelfolder}/model.pth")

    # ✅ 获取 speed 的 scaler 参数
    mean_vec, std_vec = train_loader.dataset.get_scalers()
    mean_speed, std_speed = mean_vec[0], std_vec[0]

    # ✅ 评估模型
    evaluate(
        model,
        test_loader,
        nsample=args.nsample,
        scaler=std_speed,
        mean_scaler=mean_speed,
        foldername=foldername,
    )
