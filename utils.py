import numpy as np
import torch
from torch.optim import Adam
from tqdm import tqdm
import pickle


def train(
    model,
    config,
    train_loader,
    valid_loader=None,
    valid_epoch_interval=20,
    foldername="",
):
    optimizer = Adam(model.parameters(), lr=config["lr"], weight_decay=1e-6)
    if foldername != "":
        output_path = foldername + "/model.pth"

    p1 = int(0.75 * config["epochs"])
    p2 = int(0.9 * config["epochs"])
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=[p1, p2], gamma=0.1
    )

    best_valid_loss = 1e10
    for epoch_no in range(config["epochs"]):
        avg_loss = 0
        model.train()
        with tqdm(train_loader, mininterval=5.0, maxinterval=50.0) as it:
            for batch_no, train_batch in enumerate(it, start=1):
                print("👀 当前 batch_size =", len(train_batch["dates"]))
                if batch_no <= 2:
                    for b in range(len(train_batch["dates"])):
                        print(f"📌 [Train] Sample {b} | Date: {train_batch['dates'][b]} | Masked: {train_batch['masked_day'][b]}")

                optimizer.zero_grad()
                loss = model(train_batch)
                loss.backward()
                avg_loss += loss.item()
                optimizer.step()
                it.set_postfix(
                    ordered_dict={
                        "avg_epoch_loss": avg_loss / batch_no,
                        "epoch": epoch_no,
                    },
                    refresh=False,
                )
                if batch_no >= config["itr_per_epoch"]:
                    break

            lr_scheduler.step()
        if valid_loader is not None and (epoch_no + 1) % valid_epoch_interval == 0:
            model.eval()
            avg_loss_valid = 0
            with torch.no_grad():
                with tqdm(valid_loader, mininterval=5.0, maxinterval=50.0) as it:
                    for batch_no, valid_batch in enumerate(it, start=1):
                        loss = model(valid_batch, is_train=0)
                        avg_loss_valid += loss.item()
                        it.set_postfix(
                            ordered_dict={
                                "valid_avg_epoch_loss": avg_loss_valid / batch_no,
                                "epoch": epoch_no,
                            },
                            refresh=False,
                        )
            if best_valid_loss > avg_loss_valid:
                best_valid_loss = avg_loss_valid
                print(
                    "\n best loss is updated to ",
                    avg_loss_valid / batch_no,
                    "at",
                    epoch_no,
                )

    if foldername != "":
        torch.save(model.state_dict(), output_path)


def quantile_loss(target, forecast, q: float, eval_points) -> float:
    return 2 * torch.sum(
        torch.abs((forecast - target) * eval_points * ((target <= forecast) * 1.0 - q))
    )


def calc_denominator(target, eval_points):
    return torch.sum(torch.abs(target * eval_points))


def calc_quantile_CRPS(target, forecast, eval_points, mean_scaler, scaler):
    target = target * scaler[0] + mean_scaler[0]
    forecast = forecast * scaler[0] + mean_scaler[0]

    quantiles = np.arange(0.05, 1.0, 0.05)
    denom = calc_denominator(target, eval_points)
    CRPS = 0
    for q in quantiles:
        q_pred = torch.quantile(forecast, q, dim=1)
        q_loss = quantile_loss(target, q_pred, q, eval_points)
        CRPS += q_loss / denom
    return CRPS.item() / len(quantiles)


def calc_quantile_CRPS_sum(target, forecast, eval_points, mean_scaler, scaler):
    eval_points = eval_points.mean(-1)
    target = target * scaler[0] + mean_scaler[0]
    target = target.sum(-1)
    forecast = forecast * scaler[0] + mean_scaler[0]

    quantiles = np.arange(0.05, 1.0, 0.05)
    denom = calc_denominator(target, eval_points)
    CRPS = 0
    for i in range(len(quantiles)):
        q_pred = torch.quantile(forecast.sum(-1), quantiles[i], dim=1)
        q_loss = quantile_loss(target, q_pred, quantiles[i], eval_points)
        CRPS += q_loss / denom
    return CRPS.item() / len(quantiles)

def evaluate(model, test_loader, nsample=100, scaler=1, mean_scaler=0, foldername=""):
    print(f"🧪 Starting evaluation on {len(test_loader.dataset)} test samples")

    with torch.no_grad():
        model.eval()
        mse_total = 0
        mae_total = 0
        evalpoints_total = 0

        all_target = []
        all_evalpoint = []
        all_observed_time = []
        all_generated_samples = []

        for batch_no, batch in enumerate(tqdm(test_loader), start=1):
            observed_data = batch["observed_data"].to(model.device)
            observed_mask = batch["observed_mask"].to(model.device)
            gt_mask = batch["gt_mask"].to(model.device)
            timepoints = batch["timepoints"].to(model.device)

            if batch_no == 1:
                print(f"🧪 🔍 [Sanity] Checking batch on batch_no={batch_no}")
                print(f"🔢 batch['dates'] length      = {len(batch['dates'])}")
                print(f"🔢 batch['masked_day'] length = {len(batch['masked_day'])}")

            if batch_no <= 2:
                for b in range(len(batch["dates"])):
                    print(f"📌 [Evaluate] Sample {b} | Date: {batch['dates'][b]} | Masked: {batch['masked_day'][b]}")
                    masked_count = (gt_mask[b, :, 0] == 0).sum().item()
                    print(f"📎 Sample {b}: masked speed timesteps = {masked_count}")

            # 🔧 保留你自己的 gt_mask 转换而来的 eval_points
            eval_points = (gt_mask == 0).float()
            eval_points[:, :, 1:] = 0.0  # 只评估 speed

            batch_eval_total = eval_points[:, :, 0].sum().item()
            print(f"🎯 Eval batch #{batch_no} | eval_points sum (speed): {batch_eval_total}")

            if batch_eval_total == 0:
                print(f"⚠️  Skipping batch {batch_no}: no speed values to evaluate.")
                continue

            batch_data = {
                "observed_data": observed_data,
                "observed_mask": observed_mask,
                "timepoints": timepoints,
            }

            samples, data, _, mask, time = model.evaluate(batch_data, n_samples=nsample)
            print("🧪 Raw samples shape:", samples.shape)

            # samples = samples.permute(0, 1, 3, 2)  # (B, nsample, T, K)
            samples_median = samples.median(dim=1)  # (B, T, K)

            speed_pred   = samples_median.values[:, :, 0]  # (B, T)
            speed_target = data[:, :, 0]                   # (B, T)
            speed_mask   = eval_points[:, :, 0]            # (B, T)

            scale = scaler[0]
            print("🧮 pred:", speed_pred.shape)
            print("🧮 target:", speed_target.shape)
            print("🧮 mask:", speed_mask.shape)

            mae_current = torch.abs(speed_pred - speed_target) * speed_mask * scale
            mse_current = ((speed_pred - speed_target) ** 2) * speed_mask * (scale ** 2)

            mae_total += mae_current.sum().item()
            mse_total += mse_current.sum().item()
            evalpoints_total += speed_mask.sum().item()

            all_target.append(data)
            all_evalpoint.append(eval_points)
            all_observed_time.append(time)
            all_generated_samples.append(samples)

        if evalpoints_total == 0:
            print("⚠️ No valid evaluation points found. Nothing to save.")
            return

        all_target = torch.cat(all_target, dim=0)
        all_evalpoint = torch.cat(all_evalpoint, dim=0)
        all_observed_time = torch.cat(all_observed_time, dim=0)
        all_generated_samples = torch.cat(all_generated_samples, dim=0)

        with open(foldername + f"/generated_outputs_nsample{nsample}.pk", "wb") as f:
            pickle.dump(
                [
                    all_generated_samples,
                    all_target,
                    all_evalpoint,
                    all_observed_time,
                    scaler,
                    mean_scaler,
                ],
                f,
            )

        CRPS = calc_quantile_CRPS(all_target, all_generated_samples, all_evalpoint, mean_scaler, scaler)
        CRPS_sum = calc_quantile_CRPS_sum(all_target, all_generated_samples, all_evalpoint, mean_scaler, scaler)

        with open(foldername + f"/result_nsample{nsample}.pk", "wb") as f:
            pickle.dump([
                np.sqrt(mse_total / evalpoints_total),
                mae_total / evalpoints_total,
                CRPS,
            ], f)

        print("✅ Evaluate finished.")
        print("RMSE:", np.sqrt(mse_total / evalpoints_total))
        print("MAE:", mae_total / evalpoints_total)
        print("CRPS:", CRPS)
        print("CRPS_sum:", CRPS_sum)
