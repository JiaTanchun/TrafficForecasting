from datetime import datetime, timedelta
import numpy as np
import os
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

class SpeedForecastingDataset(Dataset):
    def __init__(self, data_dir, mode="train", mask_strategy="front"):
        self.seq_len = 8 * 144  # 8 days
        self.mask_len = 1 * 144
        self.pred_len = self.mask_len
        self.mask_strategy = mask_strategy

        if mode in ["train", "valid"]:
            all_files = [f for f in os.listdir(data_dir) if f.endswith(".csv")]
            all_files = [f for f in all_files if "2022-09-28" <= f[:10] <= "2022-10-27"]
        else:  # mode == "test"
            all_files = [f for f in os.listdir(data_dir) if f.endswith(".csv")]
            all_files = [f for f in all_files if "2023-03-02" <= f[:10] <= "2023-03-30"]

        self.data = []
        self.timestamps = []

        for f in all_files:
            df = pd.read_csv(os.path.join(data_dir, f))
            if df.shape != (144, 26):
                continue
            self.data.append(df.iloc[:, 1:].values.astype(np.float32))
            self.timestamps.append(f[:10])

        self.data = np.stack(self.data, axis=0)  # (D, 144, 25)
        self.data = self.data.reshape(-1, 25)

        self.mean = np.nanmean(self.data, axis=0)
        self.std = np.nanstd(self.data, axis=0)
        self.std[self.std == 0] = 1e-6

        self.data = (self.data - self.mean) / self.std
        self.data[np.isnan(self.data)] = 0
        self.data = self.data.reshape(-1, 144, 25)

        self.total_days = len(self.data)
        self.samples = []
        self.sample_dates = []
        self.masked_days = []  # 保存每个样本被 mask 的日期

        for i in range(self.total_days - 8 + 1):
            self.samples.append(self.data[i:i + 8])
            date_range = tuple(self.timestamps[i:i + 8])
            self.sample_dates.append(date_range)

            if self.mask_strategy == "front":
                self.masked_days.append(date_range[0])
            elif self.mask_strategy == "back":
                self.masked_days.append(date_range[7])
            elif self.mask_strategy == "middle":
                self.masked_days.append(date_range[4])  # 中间那一天，确保存在于 date_range 中

        total = len(self.samples)
        if mode == "train":
            # 使用前 70% 作为训练集
            self.samples = self.samples[:int(0.7 * total)]
            self.sample_dates = self.sample_dates[:int(0.7 * total)]
            self.masked_days = self.masked_days[:int(0.7 * total)]
        elif mode == "valid":
            # 使用剩下 30% 作为验证集
            self.samples = self.samples[int(0.7 * total):]
            self.sample_dates = self.sample_dates[int(0.7 * total):]
            self.masked_days = self.masked_days[int(0.7 * total):]
        # 🧪 注意：test 模式下的数据是 3 月份的，已经在文件名筛选中处理，无需切分

        
        print(f"🧩 Initializing dataset | mode={mode} | mask_strategy={mask_strategy} | total_samples={len(self.samples)}")

        # ✅ Debug masked_day 是否合规
        print("\n🔍 [Sanity Check: masked_day]")
        for i in range(min(10, len(self.sample_dates))):
            date_range = self.sample_dates[i]
            masked = self.masked_days[i]
            print(f"Sample {i:2d} | Dates: {date_range} | Masked: {masked}")

            # 对于 front 和 back 策略，确保 masked_day 是第一或最后一天
            if self.mask_strategy in ["front", "back"]:
                assert masked == date_range[0] or masked == date_range[-1], \
                    f"❌ masked_day '{masked}' 不在 date_range 中: {date_range}"

            # 对于 middle 策略，确保 masked_day 就是 date_range 中的某个值
            if self.mask_strategy == "middle":
                assert masked in date_range, \
                    f"❌ middle 策略生成了不在范围内的 masked_day: {masked}, date_range: {date_range}"

        # ✅ 检查数据集是否为空
        assert len(self.samples) > 0, "❌ No valid samples found in dataset!"
        assert len(self.samples) == len(self.sample_dates) == len(self.masked_days), \
            f"❌ 样本数量、日期和 masked 不一致！samples={len(self.samples)}, dates={len(self.sample_dates)}, masked={len(self.masked_days)}"


    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        assert len(self.sample_dates[idx]) == 8, f"❌ sample_dates[{idx}] 长度不是8，而是 {len(self.sample_dates[idx])}"
        assert idx < len(self.masked_days), f"❌ idx={idx} 超出 masked_days 范围（{len(self.masked_days)}）"
        seq = self.samples[idx].reshape(-1, 25)
        observed_data = torch.tensor(seq, dtype=torch.float)
        observed_mask = (~torch.isnan(observed_data)).float()
        gt_mask = observed_mask.clone()

        if self.mask_strategy == "front":
            gt_mask[:self.mask_len, 0] = 0
        elif self.mask_strategy == "back":
            gt_mask[-self.mask_len:, 0] = 0
        elif self.mask_strategy == "middle":
            mid_start = (self.seq_len - self.mask_len) // 2
            gt_mask[mid_start:mid_start + self.mask_len, 0] = 0
        if idx < 5:
            masked_speed_count = (gt_mask[:, 0] == 0).sum().item()
            print(f"🧪 idx={idx} gt_mask 0-count on speed: {masked_speed_count}")

        timepoints = torch.arange(self.seq_len).float()

        return {
            "observed_data": observed_data,
            "observed_mask": observed_mask,
            "gt_mask": gt_mask,
            "timepoints": timepoints,
            "dates": self.sample_dates[idx],
            "masked_day": self.masked_days[idx]
        }
def get_dataloader(data_path, device, batch_size=8, mask_strategy="back"):
    # 加载完整训练集，提取 mean 和 std
    full_train = SpeedForecastingDataset(data_path, mode="train", mask_strategy=mask_strategy)
    mean = torch.tensor(full_train.mean).to(device).float()
    std = torch.tensor(full_train.std).to(device).float()

    # 打印前几个样本的信息
    for i in range(min(10, len(full_train))):
        print(f"📌 [Train] Sample {i:2d} | Date: {full_train.sample_dates[i]} | Masked: {full_train.masked_days[i]}")

    # ✅ 提取公用 collate_fn
    def collate_fn(batch):
        return {
            "observed_data": torch.stack([b["observed_data"] for b in batch]),
            "observed_mask": torch.stack([b["observed_mask"] for b in batch]),
            "gt_mask": torch.stack([b["gt_mask"] for b in batch]),
            "timepoints": torch.stack([b["timepoints"] for b in batch]),
            "dates": [b["dates"] for b in batch],
            "masked_day": [b["masked_day"] for b in batch],
        }

    # ✅ 所有 loader 都使用同样的 collate_fn
    train_loader = DataLoader(full_train, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)

    valid_loader = DataLoader(
        SpeedForecastingDataset(data_path, mode="valid", mask_strategy=mask_strategy),
        batch_size=batch_size, shuffle=False, collate_fn=collate_fn
    )

    test_loader = DataLoader(
        SpeedForecastingDataset(data_path, mode="test", mask_strategy=mask_strategy),
        batch_size=batch_size, shuffle=False, collate_fn=collate_fn
    )

    # 🧪 打印首个 batch 的结构确认
    batch = next(iter(train_loader))
    print("🧪 First batch dates:")
    for i, date_tuple in enumerate(batch["dates"]):
        print(f"Sample {i}: {date_tuple} (len={len(date_tuple)})")
    print(f"🧪 Train: {len(train_loader.dataset)} samples")
    print(f"🧪 Valid: {len(valid_loader.dataset)} samples")
    print(f"🧪 Test : {len(test_loader.dataset)} samples")

    return train_loader, valid_loader, test_loader, std, mean
