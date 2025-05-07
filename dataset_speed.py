import os
import pickle
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader

def get_file_list(folder="./data/speed_data"):
    """
    Return sorted list of csv file paths in the folder.
    """
    return sorted([
        os.path.join(folder, fn)
        for fn in os.listdir(folder)
        if fn.lower().endswith('.csv')
    ])

def print_dataset_samples(dataset, num_samples=5):
    print(f"Dataset contains {len(dataset)} samples.")
    for idx in range(min(num_samples, len(dataset))):
        sample = dataset[idx]
        print(f"\nSample {idx + 1}:")
        for key, value in sample.items():
            print(f"  {key}: {value.shape if isinstance(value, np.ndarray) else value}")

#-----------以上是debug增加的函数

def parse_csv(filepath, missing_ratio=0.0):
    """
    Parse a single CSV file with columns: Datetime, Speed, <weather1>, <weather2>, ...
    Returns:
      values      (144, D)   - raw values with NaN where missing
      observed_mask (144, D) - 1 if observed (all weather cols), 1/0 in speed col
      gt_mask     (144, 1)   - mask of true-speed observed locations
    """
    df = pd.read_csv(filepath)
    values = df.iloc[:, 1:].values.astype(np.float32)  # shape (144, D)
    # separate speed and weather
    speed = values[:, 0:1]
    # build mask: all ones, then zero out random speed entries
    observed_mask = np.ones_like(values, dtype=np.float32)
    speed_mask = observed_mask[:, 0].copy()
    obs_idx = np.where(~np.isnan(speed[:, 0]))[0]
    np.random.shuffle(obs_idx)
    miss_cnt = int(len(obs_idx) * missing_ratio)
    if miss_cnt > 0:
        speed_mask[obs_idx[:miss_cnt]] = 0.0
    gt_mask = speed_mask.reshape(-1, 1)
    # fill NaN in values for training
    values = np.nan_to_num(values)
    return values, observed_mask, gt_mask


class SpeedDataset(Dataset):
    def __init__(self,
                 file_list,
                 missing_ratio=0.1,
                 seed=0,
                 cache_path=None,
                 external_mean=None,
                 external_std=None):
        self.file_list = file_list
        self.missing_ratio = missing_ratio
        np.random.seed(seed)

        print(missing_ratio)
        print(file_list)

        # determine cache path
        # if cache_path is None:
        #     folder = os.path.dirname(file_list[0])
        #     cache_name = os.path.basename(folder) + f"_mr{missing_ratio:.2f}.pkl"
        #     cache_path = os.path.join(folder, cache_name)

        # if os.path.exists(cache_path):
        #     # load and handle old vs new cache
        #     with open(cache_path, 'rb') as f:
        #         data = pickle.load(f)
        #     if len(data) == 3:
        #         # old format
        #         self.values, self.observed_masks, self.gt_masks = data
        #         # recompute mean/std
        #         N, T, D = self.values.shape
        #         flat_vals = self.values.reshape(-1, D)
        #         flat_masks = self.observed_masks.reshape(-1, D)
        #         mean = np.zeros(D, np.float32)
        #         std  = np.ones(D, np.float32)
        #         for d in range(D):
        #             vals_d = flat_vals[flat_masks[:, d] == 1, d]
        #             if vals_d.size > 0:
        #                 mean[d] = vals_d.mean()
        #                 std[d]  = vals_d.std() if vals_d.std() > 0 else 1.0
        #         self.mean_vec = mean
        #         self.std_vec  = std
        #         # rewrite new-format cache
        #         with open(cache_path, 'wb') as f2:
        #             pickle.dump((self.values,
        #                          self.observed_masks,
        #                          self.gt_masks,
        #                          self.mean_vec,
        #                          self.std_vec),
        #                         f2)
        #     else:
        #         # new format
        #         (self.values,
        #          self.observed_masks,
        #          self.gt_masks,
        #          self.mean_vec,
        #          self.std_vec) = data
        # else:
        #     # first-time load: parse all CSVs, compute mean/std, save cache

        all_vals, all_masks, all_gt = [], [], []
        for path in self.file_list:
            try:
                vals, om, gt = parse_csv(path, missing_ratio)
                if vals.shape != (144,vals.shape[1]):
                    print(f"⛔ 文件 {path} 的 shape 异常：{vals.shape}，已跳过")
                    continue
                all_vals.append(vals)
                all_masks.append(om)
                all_gt.append(gt)
            except Exception as e:
                print(f"⛔ 无法处理文件 {path}，原因: {e}")
        self.values = np.stack(all_vals, axis=0)
        self.observed_masks = np.stack(all_masks, axis=0)
        self.gt_masks = np.stack(all_gt, axis=0)

        # compute mean/std per feature
        if external_mean is None or external_std is None:
            N, T, D = self.values.shape
            flat_vals = self.values.reshape(-1, D)
            flat_masks = self.observed_masks.reshape(-1, D)
            mean = np.zeros(D, np.float32)
            std  = np.ones(D, np.float32)
            print("✅ 使用 external_mean/std 归一化当前数据集")
            for d in range(D):
                data_d = flat_vals[flat_masks[:, d] == 1, d]
                if data_d.size > 0:
                    mean[d] = data_d.mean()
                    std[d]  = data_d.std() if data_d.std() > 0 else 1.0
            self.mean_vec = mean   #每个特征的平均值
            self.std_vec  = std    #每个特征的标准差

        else:
            self.mean_vec = np.array(external_mean, dtype=np.float32)
            self.std_vec = np.array(external_std, dtype=np.float32)

        # normalize values
        self.values = ((self.values - self.mean_vec) / self.std_vec) * self.observed_masks

        # save new-format cache
        with open(cache_path, 'wb') as f:
            pickle.dump((self.values,
                         self.observed_masks,
                         self.gt_masks,
                         self.mean_vec,
                         self.std_vec),
                        f)

    def __len__(self):
        return len(self.values)

    def __getitem__(self, idx):
        return {
            'observed_data':  self.values[idx],       # (144, D)
            'observed_mask':  self.observed_masks[idx],
            'gt_mask':        self.gt_masks[idx],     # (144,1)
            'timepoints':     np.arange(self.values.shape[1])
        }

    def get_scalers(self):
        """
        Returns two arrays of shape (D,):
          mean_vec[d], std_vec[d] for each feature d.
        To get Speed scalers, use mean_vec[0], std_vec[0].
        """
        return self.mean_vec, self.std_vec

    def get_scalers(self):
        """Returns mean and std arrays of shape (D,)."""
        return self.mean_vec, self.std_vec

# def get_dataloader(folder="./data/speed_data",
#                    missing_ratio=0.1,
#                    seed=0,
#                    batch_size=16,
#                    split=(0.7, 0.15, 0.15)):
#     files = get_file_list(folder)
#     np.random.seed(seed)
#     np.random.shuffle(files)

#     N = len(files)
#     n_train = int(N * split[0])
#     n_val   = int(N * split[1])

#     train_files = files[:n_train]
#     val_files   = files[n_train:n_train + n_val]
#     test_files  = files[n_train + n_val:]

#     print(f"\n>>> Split with seed={seed}, split={split}:")
#     print("  TRAIN files ({}):".format(len(train_files)))
#     for p in train_files: print("    ", os.path.basename(p))
#     print("  VALID files ({}):".format(len(val_files)))
#     for p in val_files:   print("    ", os.path.basename(p))
#     print("  TEST  files ({}):".format(len(test_files)))
#     for p in test_files:  print("    ", os.path.basename(p))
#     print()

#     train_ds = SpeedDataset(train_files, missing_ratio, seed)
#     val_ds   = SpeedDataset(val_files,   missing_ratio, seed)
#     test_ds  = SpeedDataset(test_files,  missing_ratio, seed)

#     train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
#     val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False)
#     test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False)
#     return train_loader, val_loader, test_loader
def get_dataloader(folder="./data/speed_data",
                   seed=0,
                   nfold=None,
                   batch_size=16,
                   missing_ratio=0.1,
                   split=(0.6,0.3,0.1)):
    files = get_file_list(folder)
    np.random.seed(seed)
    np.random.shuffle(files)
    N = len(files)

    if nfold is not None:
        # 五折切分
        fold_size = N // 5
        start = nfold * fold_size
        end   = start + fold_size
        test_files  = files[start:end]
        remain      = files[:start] + files[end:]
        np.random.seed(seed)
        np.random.shuffle(remain)
        n_train = int(split[0] * len(remain))
        train_files = remain[:n_train]
        valid_files = remain[n_train:]
    else:
        # 一次性 70/15/15
        n_train = int(split[0] * N)
        n_val   = int(split[1] * N)
        train_files = files[:n_train]
        valid_files = files[n_train:n_train+n_val]
        test_files  = files[n_train+n_val:]

    # （可选）打印一下切分结果，方便 debug
    print(f"\n>>> seed={seed}  nfold={nfold}  split={split}")
    print(f" TRAIN {len(train_files)}:", [os.path.basename(p) for p in train_files])
    print(f" VALID {len(valid_files)}:", [os.path.basename(p) for p in valid_files])
    print(f" TEST  {len(test_files)}:",  [os.path.basename(p) for p in test_files])
    print()



    print(f"\n>>> seed={seed}  nfold={nfold}  split={split}")
    print(f" TRAIN {len(train_files)} files:")
    for file in train_files:
        print(f"   {os.path.basename(file)}")
    print(f" VALID {len(valid_files)} files:")
    for file in valid_files:
        print(f"   {os.path.basename(file)}")
    print(f" TEST  {len(test_files)} files:")
    for file in test_files:
        print(f"   {os.path.basename(file)}")
    print()

    #上述打印验证原始文件被成功划分

    # 下面不变：构造 Dataset & DataLoader
    print(missing_ratio)
    train_ds = SpeedDataset(train_files, missing_ratio, seed, cache_path="disable")
    mean_vec, std_vec = train_ds.get_scalers()
    valid_ds = SpeedDataset(valid_files, missing_ratio, seed, cache_path="disable", external_mean=mean_vec,external_std=std_vec)
    test_ds  = SpeedDataset(test_files, missing_ratio, seed, cache_path="disable", external_mean=mean_vec,external_std=std_vec)


    print(1000000000000000000000000000000000000001)
    print(train_ds.values.shape)
    print(valid_ds.values.shape)
    print(test_ds.values.shape)
    print(1000000000000000000000000000000000000001)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=False)
    valid_loader = DataLoader(valid_ds,   batch_size=batch_size, shuffle=False)
    test_loader  = DataLoader(test_ds,    batch_size=1, shuffle=False)
    return train_loader, valid_loader, test_loader
