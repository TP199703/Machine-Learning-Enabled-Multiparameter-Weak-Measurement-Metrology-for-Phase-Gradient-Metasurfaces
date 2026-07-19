
















import gc
import json
import os
import random
import warnings
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

warnings.filterwarnings("ignore")





ROOT = Path(r"E:\2026\最终结果\机器学习相关")

DATA_ROOT = Path(__file__).resolve().parent

FAST_CSV = DATA_ROOT / "dataset_index_fast_v174.csv"
ORIGINAL_CSV = DATA_ROOT / "dataset_index.csv"
PACKED_NPY = DATA_ROOT / "dataset_packed_221_v174.npy"

TASK_DIRS = {
    "phi": ROOT / "phi" / "mycode",
    "b": ROOT / "b" / "mycode",
    "zeta": ROOT / "zeta" / "mycode",
}

TASKS = {
    "phi": {
        "col": "c",
        "scale": 1.0,
        "type": "cyclic",
        "out_dim": 2,
    },
    "b": {
        "col": "b",
        "scale": 1.0,
        "type": "linear",
        "out_dim": 1,
    },
    "zeta": {
        "col": "zeta_raw",
        "scale": 100.0,
        "type": "linear",
        "out_dim": 1,
    },
}

SEED = 42

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)



BATCH_SIZE = 4


NUM_WORKERS = 0

RUN_TIME = datetime.now().strftime("%Y%m%d-%H%M%S")
OUTPUT_DIR = ROOT / f"saved_model_full_train_inference_{RUN_TIME}"





def setup_seed(seed):
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    np.random.seed(seed)
    random.seed(seed)





def generate_non_uniform_grid():
    p1 = np.arange(-50, -16 + 1e-5, 1)
    p2 = np.arange(-15, 15 + 1e-5, 0.2)
    p3 = np.arange(16, 50 + 1e-5, 1)

    return np.concatenate(
        [p1, p2, p3]
    ).astype(np.float32)


class FourierFeatureEmbedding(nn.Module):
    def __init__(
        self,
        input_dim=2,
        mapping_size=128,
        scale=10.0,
    ):
        super().__init__()

        self.register_buffer(
            "B",
            torch.randn(input_dim, mapping_size) * scale,
        )

    def forward(self, x):
        x_proj = (2.0 * np.pi * x) @ self.B

        return torch.cat(
            [
                torch.sin(x_proj),
                torch.cos(x_proj),
            ],
            dim=-1,
        )


class CoordBlock(nn.Module):
    def __init__(
        self,
        in_c,
        out_c,
        stride=1,
        dilation=1,
    ):
        super().__init__()

        padding = dilation if stride == 1 else 1

        self.conv = nn.Sequential(
            nn.Conv2d(
                in_c + 2,
                out_c,
                3,
                stride,
                padding=padding,
                dilation=dilation,
            ),
            nn.GroupNorm(8, out_c),
            nn.GELU(),

            nn.Conv2d(
                out_c,
                out_c,
                3,
                1,
                1,
            ),
            nn.GroupNorm(8, out_c),
        )

        if stride != 1 or in_c != out_c:
            self.short = nn.Sequential(
                nn.Conv2d(
                    in_c,
                    out_c,
                    1,
                    stride,
                    0,
                ),
                nn.GroupNorm(8, out_c),
            )
        else:
            self.short = nn.Sequential()

    def forward(self, x, grid):
        if grid.shape[-2:] != x.shape[-2:]:
            grid_resized = F.interpolate(
                grid,
                size=x.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
        else:
            grid_resized = grid

        x_in = torch.cat(
            [x, grid_resized],
            dim=1,
        )

        return F.gelu(
            self.conv(x_in) + self.short(x)
        )


class MultiScaleStem(nn.Module):
    def __init__(self):
        super().__init__()

        self.b1 = CoordBlock(
            1,
            16,
            dilation=1,
        )

        self.b2 = CoordBlock(
            1,
            16,
            dilation=2,
        )

        self.b3 = CoordBlock(
            1,
            16,
            dilation=4,
        )

        self.fusion = nn.Sequential(
            nn.Conv2d(
                48,
                32,
                1,
            ),
            nn.GroupNorm(4, 32),
            nn.GELU(),
        )

    def forward(self, x, grid):
        o1 = self.b1(x, grid)
        o2 = self.b2(x, grid)
        o3 = self.b3(x, grid)

        return self.fusion(
            torch.cat(
                [o1, o2, o3],
                dim=1,
            )
        )


class PhysModel(nn.Module):
    def __init__(
        self,
        out_dim,
        dropout=0.1,
    ):
        super().__init__()

        grid_vec = (
            generate_non_uniform_grid() - (-50.0)
        ) / 100.0

        lp_grid, qwp_grid = np.meshgrid(
            grid_vec,
            grid_vec,
        )

        self.register_buffer(
            "grid_tensor",
            torch.from_numpy(
                np.stack(
                    [lp_grid, qwp_grid]
                )
            ).unsqueeze(0).float(),
        )

        self.trans_dim = 256

        self.fourier = FourierFeatureEmbedding(
            input_dim=2,
            mapping_size=128,
            scale=10.0,
        )

        self.pos_adapter = nn.Sequential(
            nn.Linear(
                256,
                self.trans_dim,
            ),
            nn.GELU(),
            nn.Linear(
                self.trans_dim,
                self.trans_dim,
            ),
        )

        self.stem = MultiScaleStem()

        self.layer1 = CoordBlock(
            32,
            64,
            stride=1,
        )

        self.layer2 = CoordBlock(
            64,
            128,
            stride=2,
        )

        self.layer3 = CoordBlock(
            128,
            256,
            stride=2,
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.trans_dim,
            nhead=4,
            dim_feedforward=1024,
            dropout=0.1,
            activation="gelu",
            batch_first=True,
        )

        self.trans = nn.TransformerEncoder(
            encoder_layer,
            num_layers=3,
        )

        self.detail_branch = nn.Sequential(
            nn.Conv2d(
                64,
                32,
                3,
                1,
                1,
            ),
            nn.GELU(),

            nn.Conv2d(
                32,
                16,
                3,
                1,
                4,
                dilation=4,
            ),
            nn.GELU(),

            nn.Conv2d(
                16,
                8,
                3,
                1,
                8,
                dilation=8,
            ),
            nn.GELU(),
        )

        self.detail_downsample = nn.Sequential(
            nn.Conv2d(
                8,
                8,
                3,
                2,
                1,
            ),
            nn.GELU(),

            nn.Conv2d(
                8,
                4,
                3,
                2,
                1,
            ),
            nn.GELU(),

            nn.Conv2d(
                4,
                1,
                3,
                2,
                1,
            ),
            nn.GELU(),
        )

        self.head = nn.Sequential(
            nn.LayerNorm(1296),

            nn.Linear(
                1296,
                1024,
            ),
            nn.GELU(),
            nn.Dropout(dropout),

            nn.Linear(
                1024,
                out_dim,
            ),
        )

    def forward(self, x):
        batch_size = x.shape[0]

        grid = self.grid_tensor.expand(
            batch_size,
            -1,
            -1,
            -1,
        )

        stem = self.stem(
            x,
            grid,
        )

        f1 = self.layer1(
            stem,
            grid,
        )

        f2 = self.layer2(
            f1,
            grid,
        )

        f3 = self.layer3(
            f2,
            grid,
        )

        if f3.shape[-2:] != (56, 56):
            f3 = F.interpolate(
                f3,
                size=(56, 56),
                mode="bilinear",
                align_corners=False,
            )

        grid_56 = F.interpolate(
            grid,
            size=(56, 56),
            mode="bilinear",
            align_corners=False,
        )

        coords = (
            grid_56
            .flatten(2)
            .transpose(1, 2)
        )

        positional_encoding = self.pos_adapter(
            self.fourier(coords)
        )

        transformer_input = (
            f3
            .flatten(2)
            .transpose(1, 2)
            + positional_encoding
        )

        transformer_output = self.trans(
            transformer_input
        )

        fused = torch.cat(
            [
                self.detail_downsample(
                    self.detail_branch(f1)
                ).flatten(1),

                F.adaptive_max_pool2d(
                    f3,
                    1,
                ).flatten(1),

                transformer_output.mean(1),
            ],
            dim=1,
        )

        return self.head(fused)





class PhysDataset(Dataset):
    def __init__(
        self,
        dataframe,
        data_path,
        task_config,
        global_min,
        global_max,
    ):
        self.df = dataframe.reset_index(drop=True)

        self.data = np.load(
            data_path,
            mmap_mode="r",
        )

        self.task_cfg = task_config

        self.gmin = float(global_min)
        self.gmax = float(global_max)

        if self.gmax <= self.gmin:
            raise ValueError(
                f"错误的归一化范围："
                f"gmin={self.gmin}, "
                f"gmax={self.gmax}"
            )

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        row = self.df.iloc[index]

        npy_index = int(
            row["npy_index"]
        )

        image = self.data[
            npy_index
        ].astype(np.float32)

        image = np.clip(
            (
                image - self.gmin
            )
            / (
                self.gmax - self.gmin
            ),
            0.0,
            1.0,
        )

        image = image.reshape(
            1,
            221,
            221,
        )

        value = float(
            row[
                self.task_cfg["col"]
            ]
        )

        if self.task_cfg["type"] == "cyclic":
            radians = np.deg2rad(value)

            target = torch.tensor(
                [
                    np.sin(radians),
                    np.cos(radians),
                ],
                dtype=torch.float32,
            )
        else:
            target = torch.tensor(
                value
                * self.task_cfg["scale"],
                dtype=torch.float32,
            )

        return (
            torch.from_numpy(image).float(),
            target,
        )





def check_required_files():
    required_files = [
        FAST_CSV,
        PACKED_NPY,
    ]

    for task_name, task_dir in TASK_DIRS.items():
        required_files.append(
            task_dir
            / f"Best_{task_name}.pth"
        )

    missing_files = [
        file
        for file in required_files
        if not file.is_file()
    ]

    if missing_files:
        text = "\n".join(
            str(file)
            for file in missing_files
        )

        raise FileNotFoundError(
            "以下文件不存在：\n"
            + text
        )


def safe_torch_load(
    file_path,
    map_location,
):
    try:
        return torch.load(
            file_path,
            map_location=map_location,
            weights_only=False,
        )
    except TypeError:
        return torch.load(
            file_path,
            map_location=map_location,
        )


def get_model_state(checkpoint):
    if not isinstance(
        checkpoint,
        dict,
    ):
        raise TypeError(
            "PTH 文件不是字典格式。"
        )

    if "model_state_dict" in checkpoint:
        state_dict = checkpoint[
            "model_state_dict"
        ]

    elif "model" in checkpoint:
        state_dict = checkpoint[
            "model"
        ]

    elif "state_dict" in checkpoint:
        state_dict = checkpoint[
            "state_dict"
        ]

    elif checkpoint and all(
        torch.is_tensor(value)
        for value in checkpoint.values()
    ):
        state_dict = checkpoint

    else:
        raise KeyError(
            "PTH 中没有找到 "
            "model_state_dict、model "
            "或 state_dict。"
        )

    if state_dict and all(
        key.startswith("module.")
        for key in state_dict
    ):
        state_dict = {
            key[len("module."):]: value
            for key, value
            in state_dict.items()
        }

    if state_dict and all(
        key.startswith("_orig_mod.")
        for key in state_dict
    ):
        state_dict = {
            key[len("_orig_mod."):]: value
            for key, value
            in state_dict.items()
        }

    return state_dict


def load_best_model(
    task_name,
    task_config,
):
    best_path = (
        TASK_DIRS[task_name]
        / f"Best_{task_name}.pth"
    )

    checkpoint = safe_torch_load(
        best_path,
        map_location="cpu",
    )

    state_dict = get_model_state(
        checkpoint
    )

    model = PhysModel(
        task_config["out_dim"]
    )

    model.load_state_dict(
        state_dict,
        strict=True,
    )

    model = model.to(DEVICE)
    model.eval()

    return model, checkpoint, best_path





def reproduce_training_normalization():















    np.random.seed(SEED)

    packed_data = np.load(
        PACKED_NPY,
        mmap_mode="r",
    )

    sample_count = min(
        5000,
        len(packed_data),
    )

    sampled_indices = np.random.choice(
        len(packed_data),
        sample_count,
    )

    sampled_data = packed_data[
        sampled_indices
    ].astype(np.float32)

    gmax = float(
        np.percentile(
            sampled_data,
            99.5,
        )
    )

    gmin = float(
        np.percentile(
            sampled_data,
            0.5,
        )
    )

    del sampled_data
    del packed_data

    gc.collect()

    return (
        gmin,
        gmax,
        sampled_indices,
    )





def make_original_splits(
    dataframe,
):
    dataframe = dataframe.copy()

    
    dataframe["source_row"] = np.arange(
        len(dataframe),
        dtype=np.int64,
    )

    train_full, test = train_test_split(
        dataframe,
        test_size=0.1,
        random_state=42,
    )

    train, val = train_test_split(
        train_full,
        test_size=0.1111,
        random_state=42,
    )

    
    return {
        "Train": train.copy(),
        "Val": val.copy(),
        "Test": test.copy(),
    }





def run_inference(
    model,
    dataframe,
    task_config,
    global_min,
    global_max,
    description,
):
    dataset = PhysDataset(
        dataframe=dataframe,
        data_path=PACKED_NPY,
        task_config=task_config,
        global_min=global_min,
        global_max=global_max,
    )

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=(
            DEVICE.type == "cuda"
        ),
    )

    predicted_values = []

    with torch.inference_mode():
        for images, _ in tqdm(
            loader,
            desc=description,
            leave=False,
        ):
            images = images.to(
                DEVICE,
                non_blocking=(
                    DEVICE.type == "cuda"
                ),
            )

            raw_prediction = model(
                images
            ).detach().float().cpu()

            if (
                task_config["type"]
                == "cyclic"
            ):
                prediction = (
                    np.rad2deg(
                        np.arctan2(
                            raw_prediction[
                                :,
                                0,
                            ].numpy(),
                            raw_prediction[
                                :,
                                1,
                            ].numpy(),
                        )
                    )
                    + 360.0
                ) % 360.0

            else:
                prediction = (
                    raw_prediction
                    .reshape(-1)
                    .numpy()
                    / task_config["scale"]
                )

            predicted_values.append(
                prediction.astype(np.float64)
            )

    if not predicted_values:
        return np.empty(
            0,
            dtype=np.float64,
        )

    return np.concatenate(
        predicted_values
    )





def wrap180(values):
    return (
        values + 180.0
    ) % 360.0 - 180.0


def circular_correlation(
    true_degree,
    pred_degree,
):
    true_rad = np.deg2rad(
        true_degree
    )

    pred_rad = np.deg2rad(
        pred_degree
    )

    true_mean = np.angle(
        np.mean(
            np.exp(
                1j * true_rad
            )
        )
    )

    pred_mean = np.angle(
        np.mean(
            np.exp(
                1j * pred_rad
            )
        )
    )

    true_sine = np.sin(
        true_rad - true_mean
    )

    pred_sine = np.sin(
        pred_rad - pred_mean
    )

    denominator = np.sqrt(
        np.sum(
            true_sine ** 2
        )
        * np.sum(
            pred_sine ** 2
        )
    )

    if denominator <= 0:
        return np.nan

    return float(
        np.sum(
            true_sine
            * pred_sine
        )
        / denominator
    )


def calculate_metrics(
    true_values,
    predicted_values,
    errors,
    is_cyclic,
):
    absolute_errors = np.abs(
        errors
    )

    squared_errors = errors ** 2

    metrics = {
        "N": int(
            len(true_values)
        ),

        "MAE": float(
            np.mean(
                absolute_errors
            )
        ),

        "RMSE": float(
            np.sqrt(
                np.mean(
                    squared_errors
                )
            )
        ),

        "MeanBias": float(
            np.mean(
                errors
            )
        ),

        "MedianAbsoluteError": float(
            np.median(
                absolute_errors
            )
        ),

        "P95AbsoluteError": float(
            np.percentile(
                absolute_errors,
                95,
            )
        ),

        "P99AbsoluteError": float(
            np.percentile(
                absolute_errors,
                99,
            )
        ),

        "MaximumAbsoluteError": float(
            np.max(
                absolute_errors
            )
        ),

        "R_squared": np.nan,
        "Pearson_r": np.nan,
        "Circular_r": np.nan,
    }

    if is_cyclic:
        metrics["Circular_r"] = (
            circular_correlation(
                true_values,
                predicted_values,
            )
        )

    else:
        denominator = np.sum(
            (
                true_values
                - np.mean(true_values)
            ) ** 2
        )

        if denominator > 0:
            metrics["R_squared"] = float(
                1.0
                - np.sum(
                    squared_errors
                )
                / denominator
            )

        if len(true_values) > 1:
            metrics["Pearson_r"] = float(
                np.corrcoef(
                    true_values,
                    predicted_values,
                )[0, 1]
            )

    return metrics





def make_result_table(
    task_name,
    set_name,
    dataframe,
    predicted_values,
    task_config,
):
    true_values = dataframe[
        task_config["col"]
    ].to_numpy(
        dtype=np.float64
    )

    if (
        len(true_values)
        != len(predicted_values)
    ):
        raise RuntimeError(
            f"{task_name}/{set_name} "
            "真实值和预测值数量不一致。"
        )

    if (
        task_config["type"]
        == "cyclic"
    ):
        errors = wrap180(
            predicted_values
            - true_values
        )

        pred_wrapped_180 = wrap180(
            predicted_values
        )

    else:
        errors = (
            predicted_values
            - true_values
        )

        pred_wrapped_180 = np.full(
            len(predicted_values),
            np.nan,
        )

    result_table = pd.DataFrame(
        {
            "Task": task_name,
            "Set": set_name,

            "source_row": dataframe[
                "source_row"
            ].to_numpy(),

            "filename": dataframe[
                "filename"
            ].astype(str).to_numpy(),

            "npy_index": dataframe[
                "npy_index"
            ].to_numpy(
                dtype=np.int64
            ),

            "True": true_values,
            "Pred": predicted_values,

            "PredWrapped180": (
                pred_wrapped_180
            ),

            "Residual": errors,

            "AbsoluteError": np.abs(
                errors
            ),

            "SquaredError": errors ** 2,
        }
    )

    metrics = calculate_metrics(
        true_values=true_values,
        predicted_values=predicted_values,
        errors=errors,
        is_cyclic=(
            task_config["type"]
            == "cyclic"
        ),
    )

    metrics["Task"] = task_name
    metrics["Set"] = set_name

    return (
        result_table,
        metrics,
    )





def draw_task_figure(
    task_name,
    task_results,
    task_metrics,
):
    figure = plt.figure(
        figsize=(20, 15)
    )

    set_order = [
        "Train",
        "Val",
        "Test",
    ]

    for index, set_name in enumerate(
        set_order
    ):
        result = task_results[
            set_name
        ]

        metrics = task_metrics[
            set_name
        ]

        true_values = result[
            "True"
        ].to_numpy()

        predicted_values = result[
            "Pred"
        ].to_numpy()

        errors = result[
            "Residual"
        ].to_numpy()

        
        axis = plt.subplot(
            3,
            3,
            index + 1,
        )

        axis.scatter(
            true_values,
            predicted_values,
            alpha=0.20,
            s=4,
            c=(
                "blue"
                if set_name == "Train"
                else "green"
            ),
        )

        range_min = min(
            true_values.min(),
            predicted_values.min(),
        )

        range_max = max(
            true_values.max(),
            predicted_values.max(),
        )

        axis.plot(
            [
                range_min,
                range_max,
            ],
            [
                range_min,
                range_max,
            ],
            "r--",
            linewidth=2,
        )

        axis.set_title(
            f"{set_name} Scatter "
            f"(MAE={metrics['MAE']:.6g})"
        )

        axis.set_xlabel(
            "True"
        )

        axis.set_ylabel(
            "Predicted"
        )

        
        axis = plt.subplot(
            3,
            3,
            index + 4,
        )

        axis.scatter(
            true_values,
            errors,
            alpha=0.20,
            s=4,
            c="orange",
        )

        axis.axhline(
            0,
            color="red",
            linestyle="--",
        )

        axis.set_title(
            f"{set_name} Residuals"
        )

        axis.set_xlabel(
            "True"
        )

        axis.set_ylabel(
            "Predicted - True"
        )

        
        axis = plt.subplot(
            3,
            3,
            index + 7,
        )

        sns.histplot(
            errors,
            kde=True,
            color="purple",
            ax=axis,
        )

        axis.set_title(
            f"{set_name} Error Distribution"
        )

        axis.set_xlabel(
            "Residual"
        )

    figure.suptitle(
        f"Full analysis: {task_name}",
        fontsize=18,
    )

    figure.tight_layout(
        rect=[
            0,
            0,
            1,
            0.97,
        ]
    )

    figure.savefig(
        OUTPUT_DIR
        / f"Full_Analysis_{task_name}.png",
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(
        figure
    )





def main():
    setup_seed(SEED)

    check_required_files()

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=False,
    )

    print("=" * 80)
    print("已保存模型的完整 Train / Val / Test 推理")
    print(f"设备：{DEVICE}")
    print(f"数据：{PACKED_NPY}")
    print(f"输出：{OUTPUT_DIR}")
    print("=" * 80)

    dataframe_all = pd.read_csv(
        FAST_CSV
    )

    required_columns = {
        "filename",
        "npy_index",
        "b",
        "c",
        "zeta_raw",
    }

    missing_columns = (
        required_columns
        - set(
            dataframe_all.columns
        )
    )

    if missing_columns:
        raise KeyError(
            "FAST CSV 缺少列："
            + str(
                sorted(
                    missing_columns
                )
            )
        )

    packed_data = np.load(
        PACKED_NPY,
        mmap_mode="r",
    )

    if (
        len(packed_data)
        != len(dataframe_all)
    ):
        raise ValueError(
            "NPY 与 CSV 样本数不一致："
            f"{len(packed_data)} "
            f"vs {len(dataframe_all)}"
        )

    expected_indices = np.arange(
        len(dataframe_all),
        dtype=np.int64,
    )

    actual_indices = dataframe_all[
        "npy_index"
    ].to_numpy(
        dtype=np.int64
    )

    if not np.array_equal(
        expected_indices,
        actual_indices,
    ):
        raise ValueError(
            "npy_index 不是连续的 "
            "0 到 N-1，停止运行，"
            "避免图像与标签错位。"
        )

    del packed_data

    
    gmin, gmax, sampled_indices = (
        reproduce_training_normalization()
    )

    print(
        "训练归一化参数："
        f"gmin={gmin:.9g}, "
        f"gmax={gmax:.9g}"
    )

    np.save(
        OUTPUT_DIR
        / "normalization_sample_indices_seed42.npy",
        sampled_indices,
    )

    normalization_info = {
        "seed": SEED,

        "sample_count": int(
            len(sampled_indices)
        ),

        "gmin": gmin,
        "gmax": gmax,

        "method": (
            "seed=42; "
            "np.random.choice(N,5000); "
            "gmin=P0.5; "
            "gmax=P99.5"
        ),
    }

    with open(
        OUTPUT_DIR
        / "normalization_info.json",
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            normalization_info,
            file,
            ensure_ascii=False,
            indent=2,
        )

    
    split_dataframes = make_original_splits(
        dataframe_all
    )

    split_summary = []

    for set_name, split_dataframe in (
        split_dataframes.items()
    ):
        split_dataframe.to_csv(
            OUTPUT_DIR
            / f"split_{set_name}.csv",
            index=False,
        )

        split_summary.append(
            {
                "Set": set_name,
                "Rows": len(
                    split_dataframe
                ),
            }
        )

    pd.DataFrame(
        split_summary
    ).to_csv(
        OUTPUT_DIR
        / "split_summary.csv",
        index=False,
    )

    print(
        "数据划分："
        f"Train={len(split_dataframes['Train'])}, "
        f"Val={len(split_dataframes['Val'])}, "
        f"Test={len(split_dataframes['Test'])}"
    )

    all_metrics = []
    checkpoint_inventory = []

    for task_name in [
        "phi",
        "b",
        "zeta",
    ]:
        task_config = TASKS[
            task_name
        ]

        print(
            "\n"
            + "-" * 80
        )

        print(
            f"任务：{task_name}"
        )

        model, checkpoint, best_path = (
            load_best_model(
                task_name,
                task_config,
            )
        )

        checkpoint_inventory.append(
            {
                "Task": task_name,
                "BestPath": str(
                    best_path
                ),

                "CheckpointKeys": (
                    ";".join(
                        checkpoint.keys()
                    )
                    if isinstance(
                        checkpoint,
                        dict,
                    )
                    else str(
                        type(
                            checkpoint
                        )
                    )
                ),

                "StoredBestMAE": (
                    checkpoint.get(
                        "best_mae",
                        np.nan,
                    )
                    if isinstance(
                        checkpoint,
                        dict,
                    )
                    else np.nan
                ),
            }
        )

        task_results = {}
        task_metrics = {}

        for set_name in [
            "Train",
            "Val",
            "Test",
        ]:
            split_dataframe = (
                split_dataframes[
                    set_name
                ]
            )

            predicted_values = run_inference(
                model=model,

                dataframe=split_dataframe,

                task_config=task_config,

                global_min=gmin,
                global_max=gmax,

                description=(
                    f"{task_name}-{set_name}"
                ),
            )

            result_table, metrics = (
                make_result_table(
                    task_name=task_name,

                    set_name=set_name,

                    dataframe=split_dataframe,

                    predicted_values=(
                        predicted_values
                    ),

                    task_config=(
                        task_config
                    ),
                )
            )

            
            result_table.to_csv(
                OUTPUT_DIR
                / (
                    f"Detail_"
                    f"{task_name}_"
                    f"{set_name}_"
                    f"complete.csv"
                ),
                index=False,
            )

            
            result_table[
                [
                    "True",
                    "Pred",
                ]
            ].to_csv(
                OUTPUT_DIR
                / (
                    f"Detail_"
                    f"{task_name}_"
                    f"{set_name}.csv"
                ),
                index=False,
            )

            task_results[
                set_name
            ] = result_table

            task_metrics[
                set_name
            ] = metrics

            all_metrics.append(
                metrics
            )

            print(
                f"{set_name:5s} | "
                f"N={metrics['N']:6d} | "
                f"MAE={metrics['MAE']:.9g} | "
                f"RMSE={metrics['RMSE']:.9g} | "
                f"P99={metrics['P99AbsoluteError']:.9g}"
            )

        
        draw_task_figure(
            task_name,
            task_results,
            task_metrics,
        )

        del model
        del task_results

        gc.collect()

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    metrics_dataframe = pd.DataFrame(
        all_metrics
    )

    metrics_columns = [
        "Task",
        "Set",
        "N",
        "MAE",
        "RMSE",
        "MeanBias",
        "MedianAbsoluteError",
        "P95AbsoluteError",
        "P99AbsoluteError",
        "MaximumAbsoluteError",
        "R_squared",
        "Pearson_r",
        "Circular_r",
    ]

    metrics_dataframe = metrics_dataframe[
        metrics_columns
    ]

    metrics_dataframe.to_csv(
        OUTPUT_DIR
        / "all_metrics.csv",
        index=False,
    )

    pd.DataFrame(
        checkpoint_inventory
    ).to_csv(
        OUTPUT_DIR
        / "checkpoint_inventory.csv",
        index=False,
    )

    run_config = {
        "root": str(ROOT),
        "data_root": str(DATA_ROOT),
        "fast_csv": str(FAST_CSV),
        "original_csv": str(
            ORIGINAL_CSV
        ),
        "packed_npy": str(
            PACKED_NPY
        ),
        "task_dirs": {
            key: str(value)
            for key, value
            in TASK_DIRS.items()
        },
        "seed": SEED,
        "test_size": 0.1,
        "val_size_within_train_full": 0.1111,
        "train_sampling": (
            "disabled; full Train split used"
        ),
        "batch_size": BATCH_SIZE,
        "num_workers": NUM_WORKERS,
        "device": str(DEVICE),
        "output_dir": str(
            OUTPUT_DIR
        ),
    }

    with open(
        OUTPUT_DIR
        / "run_config.json",
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            run_config,
            file,
            ensure_ascii=False,
            indent=2,
        )

    print(
        "\n"
        + "=" * 80
    )

    print(
        "全部任务推理完成。"
    )

    print(
        f"输出目录：{OUTPUT_DIR}"
    )

    print(
        "Train 使用完整划分，"
        "不再抽样 5000 条。"
    )

    print(
        "=" * 80
    )


if __name__ == "__main__":
    main()
