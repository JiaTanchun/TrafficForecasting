import numpy as np
import torch
from torch.optim import Adam
from tqdm import tqdm
import pickle
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
import os

def train(
    model,
    config,
    train_loader,
    valid_loader=None,
    valid_epoch_interval=20,
    foldername="",
):
    """
    训练函数，附带中文打印：
      - 每个 epoch 的开始和结束损失
      - 前两批样本的日期和 mask 信息
      - 学习率调度情况
    Args:
      model         : 待训练的 CSDI 模型
      config        : 配置字典，需包含 config["train"]["lr"], config["train"]["epochs"], config["train"]["itr_per_epoch"]
      train_loader  : 训练集 DataLoader，batch 中含 'observed_data','observed_mask','gt_mask','timepoints'，可选 'dates'
      valid_loader  : 验证集 DataLoader，和 train_loader 同结构
      valid_epoch_interval : 每隔多少 epoch 做一次验证
      foldername    : 若非空，则在此目录下保存最佳模型和 loss 曲线
    """
    # 优化器 & 学习率调度器
    optimizer = Adam(model.parameters(), lr=config["train"]["lr"], weight_decay=1e-6)
    if foldername:
        os.makedirs(foldername, exist_ok=True)
        output_path = os.path.join(foldername, "model.pth")

    p1 = int(0.75 * config["train"]["epochs"])
    p2 = int(0.9 * config["train"]["epochs"])
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[p1, p2], gamma=0.1)

    best_valid_loss = float('inf')
    train_losses, valid_losses = [], []

    for epoch_no in range(config["train"]["epochs"]):
        model.train()
        epoch_loss = 0.0
        print(f"\n📅 Epoch {epoch_no+1}/{config['train']['epochs']} 开始训练，当前学习率：{optimizer.param_groups[0]['lr']:.6f}")
        with tqdm(train_loader, desc=f"Epoch {epoch_no}") as it:
            for batch_no, batch in enumerate(it, start=1):
                # 前两批打印日期和 mask 数量
                if batch_no <= 2 and "dates" in batch:
                    for b in range(len(batch["dates"])):
                        mcnt = int((batch["gt_mask"][b,:,0]==0).sum().item())
                        print(f"  [Train Batch{batch_no} Sample{b}] Date={batch['dates'][b]}, mask_count={mcnt}")

                optimizer.zero_grad()
                loss = model(batch)    # forward 返回 loss
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()
                it.set_postfix(avg_loss=epoch_loss / batch_no)
                if batch_no >= config["train"]["itr_per_epoch"]:
                    break

        avg_train_loss = epoch_loss / batch_no
        train_losses.append(avg_train_loss)
        print(f"✅ Epoch {epoch_no+1} 训练完毕，平均 Loss={avg_train_loss:.4f}")
        lr_scheduler.step()

        # 验证
        if valid_loader and (epoch_no+1) % valid_epoch_interval == 0:
            model.eval()
            valid_sum = 0.0
            print(f"🔍 开始第 {epoch_no+1} 个 epoch 的验证")
            with torch.no_grad(), tqdm(valid_loader, desc=f"Valid {epoch_no}") as itv:
                for vb, vbatch in enumerate(itv, start=1):
                    vloss = model(vbatch, is_train=0)
                    valid_sum += vloss.item()
                    itv.set_postfix(valid_loss=valid_sum / vb)
            avg_valid_loss = valid_sum / vb
            valid_losses.append(avg_valid_loss)
            print(f"✅ 验证完毕，平均 Valid Loss={avg_valid_loss:.4f}")
            # 保存最佳
            if avg_valid_loss < best_valid_loss:
                best_valid_loss = avg_valid_loss
                if foldername:
                    torch.save(model.state_dict(), output_path)
                    print(f"💾 New best model saved at epoch {epoch_no+1}, loss={avg_valid_loss:.4f}")

        # 绘制并保存 loss 曲线
        plt.figure()
        plt.plot(range(1, len(train_losses)+1), train_losses, label="Train")
        if valid_losses:
            x_valid = list(range(valid_epoch_interval, valid_epoch_interval*len(valid_losses)+1, valid_epoch_interval))
            plt.plot(x_valid, valid_losses, label="Valid")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        plt.grid(True)
        if foldername:
            curve_path = os.path.join(foldername, "loss_curve.png")
            plt.savefig(curve_path)
            print(f"📈 Loss curve saved to {curve_path}")
        plt.close()
    


def quantile_loss(target, forecast, q: float, eval_points) -> float:
    return 2 * torch.sum(
        torch.abs((forecast - target) * eval_points * ((target <= forecast) * 1.0 - q))
    )


def calc_denominator(target, eval_points):
    return torch.sum(torch.abs(target * eval_points))
def r_squared(y_true, y_pred, eval_points):
    y_true = y_true * eval_points
    y_pred = y_pred * eval_points
    mean_y = torch.sum(y_true) / torch.sum(eval_points)
    ss_total = torch.sum((y_true - mean_y) ** 2)
    ss_res = torch.sum((y_true - y_pred) ** 2)
    return 1 - ss_res / ss_total

def calc_quantile_CRPS(target, forecast, eval_points, mean_scaler, scaler):

    target = target * scaler + mean_scaler
    forecast = forecast * scaler + mean_scaler

    quantiles = np.arange(0.05, 1.0, 0.05)
    denom = calc_denominator(target, eval_points)
    CRPS = 0
    for i in range(len(quantiles)):
        q_pred = []
        for j in range(len(forecast)):
            q_pred.append(torch.quantile(forecast[j : j + 1], quantiles[i], dim=1))
        q_pred = torch.cat(q_pred, 0)
        q_loss = quantile_loss(target, q_pred, quantiles[i], eval_points)
        CRPS += q_loss / denom
    return CRPS.item() / len(quantiles)

def calc_quantile_CRPS_sum(target, forecast, eval_points, mean_scaler, scaler):

    eval_points = eval_points.mean(-1)
    target = target * scaler + mean_scaler
    target = target.sum(-1)
    forecast = forecast * scaler + mean_scaler

    quantiles = np.arange(0.05, 1.0, 0.05)
    denom = calc_denominator(target, eval_points)
    CRPS = 0
    for i in range(len(quantiles)):
        q_pred = torch.quantile(forecast.sum(-1),quantiles[i],dim=1)
        q_loss = quantile_loss(target, q_pred, quantiles[i], eval_points)
        CRPS += q_loss / denom
    return CRPS.item() / len(quantiles)
import os, pickle, numpy as np
import torch
from tqdm import tqdm
from sklearn.metrics import r2_score
def evaluate(
    model,
    test_loader,
    nsample=100,
    scaler=1.0,
    mean_scaler=0.0,
    foldername="",
):
    """
    评估函数，输出 RMSE/MAE/R²/MAPE，并可保存结果与中间样本。
    Args:
      model        : 已训练好的 CSDI_Speed 模型
      test_loader  : 测试集 DataLoader，batch 中需含 'observed_data','observed_mask','gt_mask','timepoints'，可选 'raw_data','date'
      nsample      : 每条序列生成样本数
      scaler       : 反归一化用标准差（scalar）
      mean_scaler  : 反归一化用均值（scalar）
      foldername   : 非空时保存 metrics.txt + generated_outputs_nsample{nsample}.pk + result_nsample{nsample}.pk
    """
    device = model.device
    print(f"\n🧪 Starting evaluation on {len(test_loader.dataset)} samples")
    print(f"   → 使用 scaler={scaler:.6f}, mean={mean_scaler:.6f} 反归一化\n")

    all_preds, all_trues = [], []
    all_generated_samples = []
    all_target = []
    all_evalpoint = []
    all_observed_time = []

    mse_sum = mae_sum = total_pts = 0

    model.eval()
    with torch.no_grad():
        for batch_i, batch in enumerate(tqdm(test_loader, desc="Evaluating"), start=1):
            batch_gpu = {k: v.to(device) for k, v in batch.items() if isinstance(v, torch.Tensor)}

            gm = batch_gpu["gt_mask"].float()
            eval_mask = (gm == 0).float()
            eval_mask[:, :, 1:] = 0.0  # 只评估 speed 通道
            mask_np = eval_mask.cpu().numpy().squeeze(-1).astype(bool)

            pts = int(mask_np.sum())
            if pts == 0:
                print(f"⚠️  Batch {batch_i} 没有 eval 点，跳过")
                continue

            samples, full_seq, _, _, _ = model.evaluate(batch, nsample)
            samples = samples.permute(0, 1, 3, 2)  # (B, S, T, D)
            samples_np = samples.cpu().numpy()
            full_np = full_seq.cpu().numpy()

            # 用中位数预测（归一化空间）
            median_pred = np.median(samples_np, axis=1)[:, :, 0]  # (B, T)
            true_norm = full_np[:, :, 0]

            # 累计误差
            delta2 = (median_pred - true_norm) ** 2 * mask_np
            delta1 = np.abs(median_pred - true_norm) * mask_np
            mse_sum += delta2.sum() * (scaler ** 2)
            mae_sum += delta1.sum() * scaler
            total_pts += mask_np.sum()

            all_preds.append(median_pred[mask_np] * scaler + mean_scaler)
            all_trues.append(true_norm[mask_np] * scaler + mean_scaler)

            all_generated_samples.append(samples_np)
            all_target.append(full_np)
            all_evalpoint.append(mask_np)
            all_observed_time.append(batch["timepoints"].numpy())

    if total_pts == 0:
        print("⚠️  无任何 evaluation 点，退出")
        return

    # 聚合指标
    preds = np.concatenate(all_preds)
    trues = np.concatenate(all_trues)
    rmse = np.sqrt(mse_sum / total_pts)
    mae = mae_sum / total_pts
    r2 = r2_score(trues, preds)
    mape = np.mean(np.abs((preds - trues) / (trues + 1e-5))) * 100

    print("\n✅ Evaluation finished.")
    print(f"   RMSE = {rmse:.4f}")
    print(f"   MAE  = {mae:.4f}")
    print(f"   R²   = {r2:.4f}")
    print(f"   MAPE = {mape:.2f}%")

    if foldername:
        os.makedirs(foldername, exist_ok=True)

        with open(os.path.join(foldername, f"result_nsample{nsample}.pk"), "wb") as f:
            pickle.dump([rmse, mae, r2, mape], f)

        # ⬇️ 拼接所有样本数据再保存
        all_generated_samples = np.concatenate(all_generated_samples, axis=0)  # (N, nsample, T, D)
        all_target = np.concatenate(all_target, axis=0)                        # (N, T, D)
        all_evalpoint = np.concatenate(all_evalpoint, axis=0)                 # (N, T)
        all_observed_time = np.concatenate(all_observed_time, axis=0)         # (N, T)

        with open(os.path.join(foldername, f"generated_outputs_nsample{nsample}.pk"), "wb") as f:
            pickle.dump([
                all_generated_samples,
                all_target,
                all_evalpoint,
                all_observed_time,
                scaler,
                mean_scaler
            ], f)

        with open(os.path.join(foldername, "metrics.txt"), "w") as f:
            f.write(f"RMSE: {rmse:.4f}\n")
            f.write(f"MAE : {mae:.4f}\n")
            f.write(f"R2  : {r2:.4f}\n")
            f.write(f"MAPE: {mape:.2f}%\n")

        print(f"\n📂 所有指标和结果文件已保存到：{foldername}/\n")


# def evaluate(
#     model,
#     test_loader,
#     nsample=100,
#     scaler=1.0,
#     mean_scaler=0.0,
#     foldername="",
# ):
#     """
#     评估函数，输出 RMSE/MAE/R²/MAPE，并可保存结果与中间样本。
#     Args:
#       model        : 已训练好的 CSDI_Speed 模型
#       test_loader  : 测试集 DataLoader，batch 中需含 'observed_data','observed_mask','gt_mask','timepoints'，可选 'dates'
#       nsample      : 每条序列生成样本数
#       scaler       : 反归一化用标准差（scalar）
#       mean_scaler  : 反归一化用均值（scalar）
#       foldername   : 非空时保存 metrics.txt + generated_outputs_nsample{nsample}.pk + result_nsample{nsample}.pk
#     """
#     device = model.device
#     print(f"\n🧪 Starting evaluation on {len(test_loader.dataset)} samples")
#     print(f"   → 使用 scaler={scaler:.6f}, mean={mean_scaler:.6f} 反归一化\n")

#     all_preds, all_trues = [], []
#     mse_sum = mae_sum = total_pts = 0

#     model.eval()
#     with torch.no_grad():
#         for batch_i, batch in enumerate(tqdm(test_loader, desc="Evaluating"), start=1):
#             # 推到 GPU
#             batch_gpu = {k: v.to(device) for k,v in batch.items() if isinstance(v, torch.Tensor)}
#             gm = batch_gpu["gt_mask"].float()     # (B, T, 1)
#             # 构造 eval mask 只评 speed
#             eval_mask = (gm == 0).float()
#             eval_mask[:,:,1:] = 0.0
#             pts = int(eval_mask[:,:,0].sum().item())
#             if pts == 0:
#                 print(f"⚠️  Batch {batch_i} 没有 eval 点，跳过")
#                 continue

#             # 调用 model.evaluate
#             samples, full_seq, _, _, _ = model.evaluate(batch, nsample)
#             # samples: (B,nsample,D,T) → (B,nsample,T,D)
#             samples = samples.permute(0,1,3,2)
#             # 转 NumPy
#             samples_np = samples.cpu().numpy()             # (B, S, T, D)
#             full_np    = full_seq.cpu().numpy()            # (B, T, D)
#             mask_np    = eval_mask.cpu().numpy().squeeze(-1).astype(bool)  # (B, T)
#             time_np    = batch_gpu["timepoints"].cpu().numpy()            # (B, T)

#             B, S, T, D = samples_np.shape

#             # 计算中位数，在归一化空间
#             med       = np.median(samples_np, axis=1)      # (B, T, D)
#             pred_norm = med[:,:,0]                         # (B, T)
#             true_norm = full_np[:,:,0]                     # (B, T)

#             if foldername:
#                 os.makedirs(foldername, exist_ok=True)
#                 with open(os.path.join(foldername, f"pred_true_values_batch{batch_i}.txt"), "w") as f:
#                     f.write("Predicted Values (normalized):\n")
#                     np.savetxt(f, pred_norm[mask_np], fmt='%.6f')
#                     f.write("\nTrue Values (normalized):\n")
#                     np.savetxt(f, true_norm[mask_np], fmt='%.6f')

#             # 累计 MSE/MAE（先在归一化空间算，再乘回 scale）
#             delta2 = (pred_norm - true_norm)**2 * mask_np
#             delta1 = np.abs(pred_norm - true_norm) * mask_np
#             mse_sum   += delta2.sum() * (scaler**2)
#             mae_sum   += delta1.sum() * scaler
#             total_pts += mask_np.sum()

#             # 收集所有点，用于全局 R²/MAPE
#             all_preds.append(pred_norm[mask_np] * scaler + mean_scaler)
#             all_trues.append(true_norm[mask_np] * scaler + mean_scaler)

#         if total_pts == 0:
#             print("⚠️  无任何 evaluation 点，退出")
#             return

#     # 全局指标
#     preds = np.concatenate(all_preds)
#     trues = np.concatenate(all_trues)
#     rmse  = np.sqrt(mse_sum / total_pts)
#     mae   = mae_sum / total_pts
#     r2    = r2_score(trues, preds)
#     mape  = np.mean(np.abs((preds - trues) / (trues + 1e-5))) * 100

#     print("\n✅ Evaluation finished.")
#     print(f"   RMSE = {rmse:.4f}")
#     print(f"   MAE  = {mae:.4f}")
#     print(f"   R²   = {r2:.4f}")
#     print(f"   MAPE = {mape:.2f}%\n")

#     if foldername:
#         os.makedirs(foldername, exist_ok=True)
#         # 保存简单结果
#         with open(os.path.join(foldername, f"result_nsample{nsample}.pk"), "wb") as f:
#             pickle.dump([rmse, mae, r2, mape], f)

#         # 保存生成样本 —— 注意这里存入 6 项：samples/full_seq/mask/time/scaler/mean_scaler
#         with open(os.path.join(foldername, f"generated_outputs_nsample{nsample}.pk"), "wb") as f:
#             pickle.dump([
#                 samples_np,   # (B, nsample, T, D)
#                 full_np,      # (B, T, D)
#                 mask_np,      # (B, T)
#                 time_np,      # (B, T)
#                 scaler,
#                 mean_scaler
#             ], f)

#         # 保存 metrics.txt
#         with open(os.path.join(foldername, "metrics.txt"), "w") as f:
#             f.write(f"RMSE: {rmse:.4f}\n")
#             f.write(f"MAE : {mae:.4f}\n")
#             f.write(f"R2  : {r2:.4f}\n")
#             f.write(f"MAPE: {mape:.2f}%\n")
#         print(f"📂 Saved metrics & generated .pk to {foldername}")
# def evaluate(
#     model,
#     test_loader,
#     nsample=100,
#     scaler=1.0,
#     mean_scaler=0.0,
#     foldername="",
# ):
#     """
#     评估函数，输出 RMSE/MAE/R²/MAPE，并可保存结果与中间样本。
#     Args:
#       model        : 已训练好的 CSDI_Speed 模型
#       test_loader  : 测试集 DataLoader，batch 中需含 'observed_data','observed_mask','gt_mask','timepoints'，可选 'dates'
#       nsample      : 每条序列生成样本数
#       scaler       : 反归一化用标准差（scalar）
#       mean_scaler  : 反归一化用均值（scalar）
#       foldername   : 非空时保存 metrics.txt + generated_outputs_nsample{nsample}.pk + result_nsample{nsample}.pk
#     """
#     device = model.device
#     print(f"\n🧪 Starting evaluation on {len(test_loader.dataset)} samples")
#     print(f"   → 使用 scaler={scaler:.6f}, mean={mean_scaler:.6f} 反归一化\n")

#     all_preds, all_trues = [], []
#     mse_sum = mae_sum = total_pts = 0

#     model.eval()
#     with torch.no_grad():
#         for batch_i, batch in enumerate(tqdm(test_loader, desc="Evaluating"), start=1):
#             # 推到 GPU
#             batch_gpu = {k: v.to(device) for k,v in batch.items() if isinstance(v, torch.Tensor)}
#             gm = batch_gpu["gt_mask"].float()     # (B, T, 1)
#             # 构造 eval mask 只评 speed
#             eval_mask = (gm == 0).float()
#             eval_mask[:,:,1:] = 0.0
#             pts = int(eval_mask[:,:,0].sum().item())
#             if pts == 0:
#                 print(f"⚠️  Batch{batch_i} 没有 eval 点，跳过")
#                 continue

#             # 调用 model.evaluate
#             samples, full_seq, _, _, _ = model.evaluate(batch, nsample)
#             # samples: (B,nsample,D,T) → (B,nsample,T,D)
#             samples = samples.permute(0,1,3,2)
#             # 转 NumPy
#             samples_np = samples.cpu().numpy()
#             full_np    = full_seq.cpu().numpy()
#             mask_np    = eval_mask.cpu().numpy().squeeze(-1).astype(bool)

#             B, S, T, D = samples_np.shape

#             # 计算中位数，归一化空间
#             med = np.median(samples_np, axis=1)  # (B,T,D)
#             pred_norm = med[:,:,0]               # (B,T)
#             true_norm = full_np[:,:,0]           # (B,T)

#             # 累计 MSE/MAE（在归一化空间先算，再乘回 scale）
#             delta2 = (pred_norm - true_norm)**2 * mask_np
#             delta1 = np.abs(pred_norm - true_norm) * mask_np
#             mse_sum += delta2.sum() * (scaler**2)
#             mae_sum += delta1.sum() * scaler
#             total_pts += mask_np.sum()

#             # 收集所有点用于全局 R²/MAPE
#             all_preds.append(pred_norm[mask_np] * scaler + mean_scaler)
#             all_trues.append(true_norm[mask_np] * scaler + mean_scaler)

#         if total_pts == 0:
#             print("⚠️  无任何 evaluation 点，退出")
#             return

#     # 全局指标
#     preds = np.concatenate(all_preds)
#     trues = np.concatenate(all_trues)
#     rmse = np.sqrt(mse_sum / total_pts)
#     mae  = mae_sum / total_pts
#     r2   = r2_score(trues, preds)
#     mape = np.mean(np.abs((preds - trues) / (trues + 1e-5))) * 100

#     print("\n✅ Evaluation finished.")
#     print(f"   RMSE = {rmse:.4f}")
#     print(f"   MAE  = {mae:.4f}")
#     print(f"   R²   = {r2:.4f}")
#     print(f"   MAPE = {mape:.2f}%\n")

#     # 结果保存
#     if foldername:
#         os.makedirs(foldername, exist_ok=True)
#         # 保存简单结果
#         res_path = os.path.join(foldername, f"result_nsample{nsample}.pk")
#         with open(res_path, "wb") as f:
#             pickle.dump([rmse, mae, r2, mape], f)
#         # 保存生成样本
#         gen_path = os.path.join(foldername, f"generated_outputs_nsample{nsample}.pk")
#         with open(gen_path, "wb") as f:
#             pickle.dump([samples_np, full_np, mask_np, scaler, mean_scaler], f)
#         # 保存 metrics.txt
#         with open(os.path.join(foldername, "metrics.txt"), "w") as f:
#             f.write(f"RMSE: {rmse:.4f}\n")
#             f.write(f"MAE : {mae:.4f}\n")
#             f.write(f"R2  : {r2:.4f}\n")
#             f.write(f"MAPE: {mape:.2f}%\n")
#         print(f"📂 Saved metrics to {foldername}")