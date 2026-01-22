import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import warnings

warnings.filterwarnings("ignore")

# ==========================================
#
# ==========================================
CONFIG = {
    "DATA_ROOT": r"C:\Users\Administrator\Desktop\dataset",
    "CODE_DIR": r"C:\Users\Administrator\Desktop\mycode",
    "FAST_CSV": "dataset_index_fast_v174.csv",
    "PACKED_NPY": "dataset_packed_221_v174.npy",

    "DEVICE": "cuda",
    "BATCH_SIZE": 64
}


# ==========================================
#
# ==========================================
def generate_non_uniform_grid():
    p1 = np.arange(-50, -16 + 1e-5, 1);
    p2 = np.arange(-15, 15 + 1e-5, 0.2);
    p3 = np.arange(16, 50 + 1e-5, 1)
    return np.concatenate([p1, p2, p3]).astype(np.float32)


class FourierFeatureEmbedding(nn.Module):
    def __init__(self, input_dim=2, mapping_size=128, scale=10.0):
        super().__init__()
        self.register_buffer('B', torch.randn(input_dim, mapping_size) * scale)

    def forward(self, x):
        x_proj = (2.0 * np.pi * x) @ self.B
        return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)


class CoordBlock(nn.Module):
    def __init__(self, in_c, out_c, stride=1, dilation=1):
        super().__init__()
        padding = dilation if stride == 1 else 1
        self.conv = nn.Sequential(
            nn.Conv2d(in_c + 2, out_c, 3, stride, padding=padding, dilation=dilation),
            nn.GroupNorm(8, out_c), nn.GELU(),
            nn.Conv2d(out_c, out_c, 3, 1, 1), nn.GroupNorm(8, out_c)
        )
        self.short = nn.Sequential(nn.Conv2d(in_c, out_c, 1, stride, 0),
                                   nn.GroupNorm(8, out_c)) if stride != 1 or in_c != out_c else nn.Sequential()

    def forward(self, x, grid):
        if grid.shape[-1] != x.shape[-1]:
            grid_resized = F.interpolate(grid, size=x.shape[-2:], mode='bilinear', align_corners=False)
        else:
            grid_resized = grid
        return F.gelu(self.conv(torch.cat([x, grid_resized], dim=1)) + self.short(x))


class MultiScaleStem(nn.Module):
    def __init__(self):
        super().__init__()
        self.b1 = CoordBlock(1, 16, dilation=1);
        self.b2 = CoordBlock(1, 16, dilation=2);
        self.b3 = CoordBlock(1, 16, dilation=4)
        self.fusion = nn.Sequential(nn.Conv2d(48, 32, 1), nn.GroupNorm(4, 32), nn.GELU())

    def forward(self, x, grid):
        return self.fusion(torch.cat([self.b1(x, grid), self.b2(x, grid), self.b3(x, grid)], dim=1))


class PhysModel(nn.Module):
    def __init__(self, out_dim):
        super().__init__()
        grid_vec = (generate_non_uniform_grid() - (-50.0)) / 100.0
        self.register_buffer('grid_tensor',
                             torch.from_numpy(np.stack(np.meshgrid(grid_vec, grid_vec))).unsqueeze(0).float())
        self.fourier = FourierFeatureEmbedding(2, 128, 10.0)
        self.pos_adapter = nn.Sequential(nn.Linear(256, 256), nn.GELU(), nn.Linear(256, 256))
        self.stem = MultiScaleStem()
        self.layer1 = CoordBlock(32, 64, stride=1);
        self.layer2 = CoordBlock(64, 128, stride=2);
        self.layer3 = CoordBlock(128, 256, stride=2)
        enc = nn.TransformerEncoderLayer(d_model=256, nhead=4, dim_feedforward=1024, dropout=0.1, activation='gelu',
                                         batch_first=True)
        self.trans = nn.TransformerEncoder(enc, num_layers=3)
        self.detail_branch = nn.Sequential(nn.Conv2d(64, 32, 3, 1, 1), nn.GELU(),
                                           nn.Conv2d(32, 16, 3, 1, 4, dilation=4), nn.GELU(),
                                           nn.Conv2d(16, 8, 3, 1, 8, dilation=8), nn.GELU())
        self.detail_downsample = nn.Sequential(nn.Conv2d(8, 8, 3, 2, 1), nn.GELU(), nn.Conv2d(8, 4, 3, 2, 1), nn.GELU(),
                                               nn.Conv2d(4, 1, 3, 2, 1), nn.GELU())
        self.head = nn.Sequential(nn.LayerNorm(1296), nn.Linear(1296, 1024), nn.GELU(), nn.Dropout(0.1),
                                  nn.Linear(1024, out_dim))

    def forward(self, x):
        B = x.shape[0];
        grid = self.grid_tensor.expand(B, -1, -1, -1)
        f1 = self.layer1(self.stem(x, grid), grid);
        f2 = self.layer2(f1, grid);
        f3 = self.layer3(f2, grid)
        if f3.shape[-1] != 56: f3 = F.interpolate(f3, size=(56, 56), mode='bilinear', align_corners=False)
        grid_56 = F.interpolate(grid, size=(56, 56), mode='bilinear', align_corners=False)
        pe = self.pos_adapter(self.fourier(grid_56.flatten(2).transpose(1, 2)))
        t_out = self.trans(f3.flatten(2).transpose(1, 2) + pe)
        concat = torch.cat(
            [self.detail_downsample(self.detail_branch(f1)).flatten(1), F.adaptive_max_pool2d(f3, 1).flatten(1),
             t_out.mean(1)], dim=1)
        return self.head(concat)


# ==========================================
#
# ==========================================
class PhysDataset(Dataset):
    def __init__(self, df, data_path, task_cfg, gmax=311.0, gmin=-311.0):
        self.df = df.reset_index(drop=True)
        self.data = np.load(data_path, mmap_mode='r')
        self.task_cfg = task_cfg;
        self.gmax, self.gmin = gmax, gmin

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img = self.data[int(self.df.iloc[idx]['npy_index'])].astype(np.float32)
        img = np.clip((img - self.gmin) / (self.gmax - self.gmin), 0, 1).reshape(1, 221, 221)
        val = float(self.df.iloc[idx][self.task_cfg['col']])
        if self.task_cfg['type'] == 'cyclic':
            rad = np.deg2rad(val);
            target = torch.tensor([np.sin(rad), np.cos(rad)], dtype=torch.float32)
        else:
            target = torch.tensor(val * self.task_cfg['scale'], dtype=torch.float32)
        return torch.from_numpy(img), target


def run_inference(model, loader, device, t_cfg):
    model.eval();
    preds, trues = [], []
    with torch.no_grad():
        for x, y in tqdm(loader, leave=False):
            p = model(x.to(device)).cpu()
            if t_cfg['type'] == 'cyclic':
                preds.extend((np.rad2deg(np.arctan2(p[:, 0], p[:, 1])) + 360) % 360)
                trues.extend((np.rad2deg(np.arctan2(y[:, 0], y[:, 1])) + 360) % 360)
            else:
                preds.extend(p.squeeze().numpy() / t_cfg['scale'])
                trues.extend(y.numpy() / t_cfg['scale'])
    return np.array(trues), np.array(preds)


# ==========================================
# 3. 主分析逻辑
# ==========================================
def main():
    print(f" let go...: {CONFIG['DEVICE']}")


    df_all = pd.read_csv(os.path.join(CONFIG['DATA_ROOT'], CONFIG['FAST_CSV']))
    df_train_full, df_test = train_test_split(df_all, test_size=0.1, random_state=42)
    df_train, df_val = train_test_split(df_train_full, test_size=0.1111, random_state=42)

    sets = {"Train": df_train, "Val": df_val, "Test": df_test}
    tasks = {
        "b": {"col": "b", "scale": 1.0, "type": "linear"},
        "zeta": {"col": "zeta_raw", "scale": 100.0, "type": "linear"},
        "phi": {"col": "c", "scale": 1.0, "type": "cyclic"}
    }

    for t_name, t_cfg in tasks.items():
        best_pth = os.path.join(CONFIG['CODE_DIR'], f"Best_{t_name}.pth")

        if not os.path.exists(best_pth):
            print(f" The weight file for task {t_name} does not exist; skip this step.")
            continue

        print(f"\n go: {t_name}")
        model = PhysModel(2 if t_cfg['type'] == 'cyclic' else 1).to(CONFIG['DEVICE'])
        model.load_state_dict(torch.load(best_pth, map_location=CONFIG['DEVICE'])['model_state_dict'])


        task_summary = []


        plt.figure(figsize=(20, 15))

        for i, (set_name, df_subset) in enumerate(sets.items()):

            if set_name == "Train" and len(df_subset) > 5000:
                df_subset = df_subset.sample(5000, random_state=42)

            ds = PhysDataset(df_subset, os.path.join(CONFIG['DATA_ROOT'], CONFIG['PACKED_NPY']), t_cfg)
            loader = DataLoader(ds, batch_size=CONFIG['BATCH_SIZE'], num_workers=0)

            y_true, y_pred = run_inference(model, loader, CONFIG['DEVICE'], t_cfg)


            if t_cfg['type'] == 'cyclic':
                errors = (y_pred - y_true + 180) % 360 - 180
            else:
                errors = y_pred - y_true
            mae = np.mean(np.abs(errors))
            rmse = np.sqrt(np.mean(errors ** 2))

            task_summary.append({"Set": set_name, "MAE": mae, "RMSE": rmse})


            plt.subplot(3, 3, i + 1)
            plt.scatter(y_true, y_pred, alpha=0.3, s=5, c='blue' if set_name == "Train" else 'green')
            rg = [y_true.min(), y_true.max()]
            plt.plot(rg, rg, 'r--', lw=2)
            plt.title(f"{set_name} Scatter (MAE: {mae:.4f})")


            plt.subplot(3, 3, i + 4)
            plt.scatter(y_true, errors, alpha=0.3, s=5, c='orange')
            plt.axhline(0, color='red', linestyle='--')
            plt.title(f"{set_name} Residuals")


            plt.subplot(3, 3, i + 7)
            sns.histplot(errors, kde=True, color='purple')
            plt.title(f"{set_name} Error Dist")


            pd.DataFrame({'True': y_true, 'Pred': y_pred}).to_csv(
                os.path.join(CONFIG['CODE_DIR'], f"Detail_{t_name}_{set_name}.csv"), index=False
            )


        pd.DataFrame(task_summary).to_csv(os.path.join(CONFIG['CODE_DIR'], f"Metrics_{t_name}_Summary.csv"),
                                          index=False)


        plt.tight_layout()
        plt.savefig(os.path.join(CONFIG['CODE_DIR'], f"Full_Analysis_{t_name}.png"), dpi=200)
        plt.close()

        print(f"  {t_name} Done。")
        del model;
        torch.cuda.empty_cache()

    print("\n All Done。")


if __name__ == "__main__":
    main()