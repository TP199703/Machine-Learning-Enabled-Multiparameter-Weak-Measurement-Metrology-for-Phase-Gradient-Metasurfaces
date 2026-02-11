import os
import sys
import time
import random
import math
import shutil
import ctypes
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.multiprocessing as mp
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from multiprocessing import shared_memory
from sklearn.model_selection import train_test_split
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm


matplotlib.use('Agg')

# ==========================================

# ==========================================
CONFIG = {
    "DEBUG": False,
# ==========================================
#  CSV and npy file of the dataset
# ==========================================
    "DATA_ROOT": r"C:\Users\Administrator\Desktop\dataset\hige_super_step_15_npy",
    "OUTPUT_DIR": r"C:\Users\Administrator\Desktop\mycode",

    "ORIGINAL_CSV": "dataset_index.csv",
    "FAST_CSV": "dataset_index_fast_v174.csv",
    "PACKED_NPY": "dataset_packed_221_v174.npy",

    "RESUME": True,
    "COMPILE": False,
    "ACCUM_STEPS": 1,
    "EPOCHS": 280,


    "VAL_INTERVAL": 1,

    "LR": 1e-3,
    "SEED": 42
}

CONFIG["TASKS"] = {
    "phi": {"col": "c", "scale": 1.0, "type": "cyclic"},
    "zeta": {"col": "zeta_raw", "scale": 100.0, "type": "linear"},
    "b": {"col": "b", "scale": 1.0, "type": "linear"}
}

os.makedirs(CONFIG['OUTPUT_DIR'], exist_ok=True)


# ==========================================
#Generate non-uniform coordinate space
# ==========================================
def generate_non_uniform_grid():
    p1 = np.arange(-50, -16 + 1e-5, 1)
    p2 = np.arange(-15, 15 + 1e-5, 0.2)
    p3 = np.arange(16, 50 + 1e-5, 1)
    return np.concatenate([p1, p2, p3]).astype(np.float32)


CONFIG["IMG_SIZE"] = len(generate_non_uniform_grid())


def setup_seed(seed):
    torch.manual_seed(seed);
    torch.cuda.manual_seed_all(seed);
    np.random.seed(seed);
    random.seed(seed)
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2 ** 32;
    np.random.seed(worker_seed);
    random.seed(worker_seed)


def check_disk_space(path):
    try:
        total, used, free = shutil.disk_usage(path); tqdm.write(f"Disk Free: {free / (1024 ** 3):.2f} GB")
    except:
        pass


def check_and_convert_data():
    if CONFIG['DEBUG']: return
    npy_path = os.path.join(CONFIG['DATA_ROOT'], CONFIG['PACKED_NPY'])
    csv_path = os.path.join(CONFIG['DATA_ROOT'], CONFIG['FAST_CSV'])
    if os.path.exists(npy_path) and os.path.exists(csv_path):
        print(f"Data Found: {os.path.basename(npy_path)} (Skipping Pack)")
        return
    print("Data file not found!");
    sys.exit(1)


def auto_scale_batch_size(gpu_id):
    try:
        total_mem = torch.cuda.get_device_properties(gpu_id).total_memory / (1024 ** 3)
        usable_gb = max(1.0, total_mem - 4.0)
        #  系数 2.9: 占用 38G显存
        raw_bs = int(usable_gb * 2.9)
        aligned_bs = max(16, (raw_bs // 8) * 8)
        tqdm.write(f"🖥️ [GPU {gpu_id}] VRAM: {total_mem:.1f}G | BatchSize: {aligned_bs}")
        return aligned_bs
    except:
        return 48


def prepare_data(config, num_gpus):
    if config['DEBUG']: return None, {'mode': 'fake'}, 1.0, 0.0, None, 4
    csv_path = os.path.join(config['DATA_ROOT'], config['FAST_CSV'])
    npy_path = os.path.join(config['DATA_ROOT'], config['PACKED_NPY'])
    df = pd.read_csv(csv_path)
    temp_mmap = np.load(npy_path, mmap_mode='r')
    sample = temp_mmap[np.random.choice(len(temp_mmap), min(5000, len(temp_mmap)))].astype(np.float32)
    gmax, gmin = float(np.percentile(sample, 99.5)), float(np.percentile(sample, 0.5))
    print(f"if you have any question please contact me 1056575808@qq.com Norm Stats: Gmax={gmax:.4f}, Gmin={gmin:.4f}")

    shm_name = 'v190_shm_data'
    try:
        shared_memory.SharedMemory(name=shm_name).unlink()
    except:
        pass
    print(f"Loading RAM...")
    shm = shared_memory.SharedMemory(create=True, size=temp_mmap.nbytes, name=shm_name)
    shared_arr = np.ndarray(temp_mmap.shape, dtype=temp_mmap.dtype, buffer=shm.buf)
    chunk = 100000;
    total = temp_mmap.shape[0]
    for i in tqdm(range(0, total, chunk), desc="RAM Load"):
        end = min(i + chunk, total);
        shared_arr[i:end] = temp_mmap[i:end]
    data_info = {'mode': 'ram', 'name': shm_name, 'shape': temp_mmap.shape, 'dtype': np.float16}
    workers = 4
    return df, data_info, gmax, gmin, shm, workers


class PhysDataset(Dataset):
    def __init__(self, df, data_info, task_config, global_max, global_min):
        self.df = df.reset_index(drop=True);
        self.data_info = data_info
        self.task_cfg = task_config;
        self.g_max, self.g_min = global_max, global_min
        self.data_source = None;
        self.shm = None

    def _init_data(self):
        if self.data_source is not None: return
        if self.data_info['mode'] == 'ram':
            self.shm = shared_memory.SharedMemory(name=self.data_info['name'])
            self.data_source = np.ndarray(self.data_info['shape'], dtype=self.data_info['dtype'], buffer=self.shm.buf)
        else:
            self.data_source = np.load(self.data_info['path'], mmap_mode='r')

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        if self.data_source is None: self._init_data()
        img = self.data_source[int(self.df.iloc[idx]['npy_index'])].astype(np.float32)
        denom = self.g_max - self.g_min
        img = np.clip((img - self.g_min) / (denom if denom > 1e-6 else 1.0), 0.0, 1.0)
        img = img.reshape(221, 221)[np.newaxis, :, :]
        val = float(self.df.iloc[idx][self.task_cfg['col']])
        if self.task_cfg['type'] == 'cyclic':
            rad = np.deg2rad(val);
            target = torch.tensor([np.sin(rad), np.cos(rad)], dtype=torch.float32)
        else:
            target = torch.tensor(val * self.task_cfg['scale'], dtype=torch.float32)
        return torch.from_numpy(img).float(), target

    def close(self):
        if self.shm: self.shm.close()


# ==========================================
# Main Model
# ==========================================
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
        x_in = torch.cat([x, grid_resized], dim=1)
        return F.gelu(self.conv(x_in) + self.short(x))


class MultiScaleStem(nn.Module):
    def __init__(self):
        super().__init__()
        self.b1 = CoordBlock(1, 16, dilation=1)
        self.b2 = CoordBlock(1, 16, dilation=2)
        self.b3 = CoordBlock(1, 16, dilation=4)
        self.fusion = nn.Sequential(nn.Conv2d(48, 32, 1), nn.GroupNorm(4, 32), nn.GELU())

    def forward(self, x, grid):
        o1 = self.b1(x, grid);
        o2 = self.b2(x, grid);
        o3 = self.b3(x, grid)
        return self.fusion(torch.cat([o1, o2, o3], dim=1))


class PhysModel(nn.Module):
    def __init__(self, out_dim, dropout=0.1):
        super().__init__()
        grid_vec = (generate_non_uniform_grid() - (-50.0)) / 100.0
        lp_grid, qwp_grid = np.meshgrid(grid_vec, grid_vec)
        self.register_buffer('grid_tensor', torch.from_numpy(np.stack([lp_grid, qwp_grid])).unsqueeze(0).float())
        self.trans_dim = 256
        self.fourier = FourierFeatureEmbedding(input_dim=2, mapping_size=128, scale=10.0)
        self.pos_adapter = nn.Sequential(nn.Linear(256, self.trans_dim), nn.GELU(),
                                         nn.Linear(self.trans_dim, self.trans_dim))

        self.stem = MultiScaleStem()
        self.layer1 = CoordBlock(32, 64, stride=1)
        self.layer2 = CoordBlock(64, 128, stride=2)
        self.layer3 = CoordBlock(128, 256, stride=2)

        enc = nn.TransformerEncoderLayer(d_model=self.trans_dim, nhead=4, dim_feedforward=1024, dropout=0.1,
                                         activation='gelu', batch_first=True)
        self.trans = nn.TransformerEncoder(enc, num_layers=3)

        self.detail_branch = nn.Sequential(nn.Conv2d(64, 32, 3, 1, 1), nn.GELU(),
                                           nn.Conv2d(32, 16, 3, 1, 4, dilation=4), nn.GELU(),
                                           nn.Conv2d(16, 8, 3, 1, 8, dilation=8), nn.GELU())
        self.detail_downsample = nn.Sequential(nn.Conv2d(8, 8, 3, 2, 1), nn.GELU(), nn.Conv2d(8, 4, 3, 2, 1), nn.GELU(),
                                               nn.Conv2d(4, 1, 3, 2, 1), nn.GELU())
        self.head = nn.Sequential(nn.LayerNorm(1296), nn.Linear(1296, 1024), nn.GELU(), nn.Dropout(dropout),
                                  nn.Linear(1024, out_dim))

    def forward(self, x):
        B = x.shape[0];
        grid = self.grid_tensor.expand(B, -1, -1, -1)
        s = self.stem(x, grid)
        f1 = self.layer1(s, grid);
        f2 = self.layer2(f1, grid);
        f3 = self.layer3(f2, grid)
        if f3.shape[-1] != 56: f3 = F.interpolate(f3, size=(56, 56), mode='bilinear', align_corners=False)

        grid_56 = F.interpolate(grid, size=(56, 56), mode='bilinear', align_corners=False)
        coords = grid_56.flatten(2).transpose(1, 2)
        pe = self.pos_adapter(self.fourier(coords))
        src = f3.flatten(2).transpose(1, 2) + pe
        try:
            with torch.backends.cuda.sdp_kernel(enable_flash=True, enable_math=True, enable_mem_efficient=True):
                t_out = self.trans(src)
        except:
            t_out = self.trans(src)

        concat = torch.cat(
            [self.detail_downsample(self.detail_branch(f1)).flatten(1), F.adaptive_max_pool2d(f3, 1).flatten(1),
             t_out.mean(1)], dim=1)
        return self.head(concat)


# ==========================================
# 5.
# ==========================================
class PhiGeometricLoss(nn.Module):
    def __init__(self, w=0.1): super().__init__(); self.mse = nn.MSELoss(); self.w = w

    def forward(self, p, t): return self.mse(p, t) + self.w * torch.mean((torch.norm(p, p=2, dim=1) - 1) ** 2)


class ScaledHuberLoss(nn.Module):
    def __init__(self, s=1.0, d=1.0): super().__init__(); self.s = s; self.h = nn.HuberLoss(delta=d)

    def forward(self, p, t): return self.s * self.h(p, t)


#
def save_log(h, t, d):
    try:
        dt = pd.DataFrame({'Epoch': h['ep'], 'Train_Loss': h['train_loss']})
        #
        if CONFIG['VAL_INTERVAL'] == 1:
            dt['Val_MAE'] = h['val_mae']
        else:
            #
            val_df = pd.DataFrame({'Epoch': h['val_epoch'], 'Val_MAE': h['val_mae']})
            dt = pd.merge(dt, val_df, on='Epoch', how='left')

        csv_path = os.path.join(d, f"History_{t}.csv")
        dt.to_csv(csv_path, index=False)
    except Exception as e:
        print(f" Save CSV Error: {e}")


def run_train(gpu, q, cfg_pkt, res_dict, num_workers):
    df, data_info, gmax, gmin = cfg_pkt
    setup_seed(CONFIG['SEED']);
    dev = torch.device(f"cuda:{gpu}")
    batch_size = auto_scale_batch_size(gpu)
    amp_dtype = torch.bfloat16
    g = torch.Generator();
    g.manual_seed(CONFIG['SEED'])

    while not q.empty():
        try:
            t_name = q.get(timeout=2)
        except:
            break
        tqdm.write(f"\n [GPU {gpu}] Start Task: {t_name} | BatchSize: {batch_size}")

        path_ckpt = os.path.join(CONFIG['OUTPUT_DIR'], f"Ckpt_{t_name}.pth")
        path_best = os.path.join(CONFIG['OUTPUT_DIR'], f"Best_{t_name}.pth")
        t_cfg = CONFIG['TASKS'][t_name]

        tr_df, te_df = train_test_split(df, test_size=0.1, random_state=42)
        tr_df, va_df = train_test_split(tr_df, test_size=0.1111, random_state=42)
        ds_tr = PhysDataset(tr_df, data_info, t_cfg, gmax, gmin)
        ds_va = PhysDataset(va_df, data_info, t_cfg, gmax, gmin)
        ld_tr = DataLoader(ds_tr, batch_size, True, num_workers=num_workers, pin_memory=True, persistent_workers=True,
                           prefetch_factor=2, worker_init_fn=seed_worker, generator=g)
        ld_va = DataLoader(ds_va, batch_size, False, num_workers=max(1, num_workers // 2), pin_memory=True,
                           persistent_workers=True, prefetch_factor=2, worker_init_fn=seed_worker, generator=g)

        model = PhysModel(2 if t_cfg['type'] == 'cyclic' else 1).to(dev)
        if t_name == 'phi':
            crit = PhiGeometricLoss(0.1).to(dev)
        elif t_name == 'zeta':
            crit = ScaledHuberLoss(20.0).to(dev)
        else:
            crit = ScaledHuberLoss(25.0).to(dev)

        opt = optim.AdamW(model.parameters(), lr=CONFIG['LR'], weight_decay=1e-4)
        sch = optim.lr_scheduler.OneCycleLR(opt, max_lr=CONFIG['LR'], total_steps=len(ld_tr) * CONFIG['EPOCHS'],
                                            pct_start=0.05, div_factor=25, final_div_factor=1000)


        st_ep, best_mae = 0, float('inf')

        hist = {'ep': [], 'train_loss': [], 'val_mae': [], 'val_epoch': []}

        if CONFIG['RESUME'] and os.path.exists(path_ckpt):
            try:
                c = torch.load(path_ckpt, map_location=dev, weights_only=False)
                model.load_state_dict(c['model']);
                opt.load_state_dict(c['opt']);
                sch.load_state_dict(c['sch'])
                st_ep = c['ep'];
                best_mae = c['best'];
                hist = c['hist']
                tqdm.write(f" [GPU {gpu}] Resumed from Epoch {st_ep}")
            except:
                pass

        for ep in range(st_ep, CONFIG['EPOCHS']):
            model.train()
            l_sum, n = 0.0, 0


            pbar = tqdm(ld_tr, desc=f"[{t_name} Ep {ep + 1}/{CONFIG['EPOCHS']}]", leave=False, position=gpu,
                        mininterval=0.5)

            for i, (x, y) in enumerate(pbar):
                x, y = x.to(dev), y.to(dev)
                if t_cfg['type'] != 'cyclic': y = y.unsqueeze(1)


                if t_name == 'b' and not CONFIG['RESUME']: x = x + torch.randn_like(x) * 0.005

                if i % CONFIG['ACCUM_STEPS'] == 0: opt.zero_grad()

                with torch.amp.autocast('cuda', dtype=amp_dtype):
                    out = model(x);
                    loss = crit(out, y) / CONFIG['ACCUM_STEPS']

                if torch.isnan(loss) or torch.isinf(loss): continue
                loss.backward()

                if (i + 1) % CONFIG['ACCUM_STEPS'] == 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    opt.step();
                    sch.step()

                l_cur = loss.item() * CONFIG['ACCUM_STEPS']
                l_sum += l_cur;
                n += 1


                pbar.set_postfix({'AvgL': f"{l_sum / n:.5f}"})

            avg_l = l_sum / n
            hist['ep'].append(ep + 1)
            hist['train_loss'].append(avg_l)

            # 验证环节
            if (ep + 1) % CONFIG['VAL_INTERVAL'] == 0:
                model.eval();
                err, cn = 0, 0
                with torch.no_grad(), torch.amp.autocast('cuda', dtype=amp_dtype):
                    for x, y in ld_va:
                        p = model(x.to(dev));
                        y = y.to(dev)
                        if t_cfg['type'] == 'cyclic':
                            d = torch.abs(torch.rad2deg(torch.atan2(p[:, 0], p[:, 1])) - torch.rad2deg(
                                torch.atan2(y[:, 0], y[:, 1])))
                            err += torch.sum(torch.min(d, 360 - d)).item()
                        else:
                            err += torch.sum(torch.abs(p.squeeze() - y) / t_cfg['scale']).item()
                        cn += len(x)

                mae = err / cn
                hist['val_mae'].append(mae)
                hist['val_epoch'].append(ep + 1)

                is_best = False
                if mae < best_mae:
                    best_mae = mae
                    is_best = True
                    torch.save({'model_state_dict': model.state_dict(), 'best_mae': best_mae}, path_best)


                tqdm.write(
                    f" Ep {ep + 1:03d} | Tr_Loss: {avg_l:.5f} | Val_MAE: {mae:.6f} | Best: {best_mae:.6f} {'🔥' if is_best else ''}")


                torch.save({'ep': ep + 1, 'model': model.state_dict(), 'opt': opt.state_dict(), 'sch': sch.state_dict(),
                            'best': best_mae, 'hist': hist}, path_ckpt)


            save_log(hist, t_name, CONFIG['OUTPUT_DIR'])

        ds_tr.close();
        ds_va.close()
        res_dict[t_name] = best_mae


if __name__ == '__main__':
    mp.set_start_method('spawn', force=True)
    print(f"  | Optimized Logging & CSV | Multi-Scale Physics Stem")
    setup_seed(CONFIG['SEED'])
    gpus = list(range(torch.cuda.device_count()))
    check_and_convert_data()
    df, d_info, gmax, gmin, shm, nw = prepare_data(CONFIG, len(gpus))

    q = mp.Queue();
    [q.put(t) for t in ['zeta']]
    mgr = mp.Manager();
    res = mgr.dict();
    procs = []

    for g in gpus:
        p = mp.Process(target=run_train, args=(g, q, (df, d_info, gmax, gmin), res, nw))
        p.start();
        procs.append(p);
        time.sleep(5)

    for p in procs: p.join()
    if shm: shm.close(); shm.unlink()
    print(" All Tasks Done")