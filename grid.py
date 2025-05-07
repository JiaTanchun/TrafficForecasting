import itertools
import json
import subprocess
import datetime
import time

# 固定 test_missing_ratio
test_missing_ratios = [1.0]

param_grid = {
    "train.epochs": [600],
    "train.lr": [ 0.001, 0.0005],  # 保留高低中三个
    "diffusion.num_steps": [100],         # 更倾向更多步骤
    "diffusion.schedule": ["quad"],       # 优先非线性
    "diffusion.channels": [128],          # 强建模能力
    "diffusion.layers": [4, 6],           # 可扩展结构
    "seed": [ 2]
}


# 基础配置
base_config = {
    "train": {
        "epochs": 600,
        "batch_size": 16,
        "lr": 0.001,
        "itr_per_epoch": 1e8
    },
    "diffusion": {
        "layers": 4,
        "channels": 64,
        "nheads": 8,
        "diffusion_embedding_dim": 128,
        "beta_start": 0.0001,
        "beta_end": 0.5,
        "num_steps": 50,
        "schedule": "quad",
        "is_linear": False
    },
    "model": {
        "is_unconditional": False,
        "timeemb": 128,
        "featureemb": 16,
        "target_strategy": "random",
        "data_dim": 25,
        "test_missing_ratio": 1.0
    }
}

def set_nested(config, key_path, value):
    keys = key_path.split('.')
    for k in keys[:-1]:
        config = config.setdefault(k, {})
    config[keys[-1]] = value

param_keys = list(param_grid.keys())
param_values = list(param_grid.values())
combinations = list(itertools.product(*param_values))

print(f"🔁 Total combinations: {len(combinations)}")

for i, values in enumerate(combinations):
    config = json.loads(json.dumps(base_config))
    args = {
        "seed": None,
        "nfold": 0,
        "testmissingratio": test_missing_ratios[0],
        "unconditional": False
    }

    for k, v in zip(param_keys, values):
        if k == "seed":
            args["seed"] = v
        else:
            set_nested(config, k, v)

    config["model"]["test_missing_ratio"] = test_missing_ratios[0]
    config["model"]["is_unconditional"] = args["unconditional"]

    now = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    foldername = f"./save/grid_fold{args['nfold']}_seed{args['seed']}_{now}/"
    config["save_folder"] = foldername

    print(f"\n🚀 [{i+1}/{len(combinations)}] Launching...")
    subprocess.run([
        "python", "exe_speed.py",
        "--config", "-",
        "--seed", str(args["seed"]),
        "--nfold", str(args["nfold"]),
        "--testmissingratio", str(args["testmissingratio"]),
    ], input=json.dumps(config), text=True)

    time.sleep(1)  # 避免文件夹时间戳重复
