import os
import argparse
import logging
import copy
from typing import List, Tuple

import numpy as np
import matplotlib.pyplot as plt
import librosa as lb
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

import utils.constants as constants
from online_alignment.cost import cosine
from online_alignment.constants import DEFAULT_DTW_STEPS, DEFAULT_DTW_WEIGHTS

from noa import alignNOA


logger = logging.getLogger(__name__)


def resolve_training_device(device: str) -> torch.device:
    """Map CLI/device string to torch.device; fall back safely when CUDA is unavailable."""
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device == "cuda":
        if not torch.cuda.is_available():
            logger.warning("CUDA requested but not available; training on CPU.")
            return torch.device("cpu")
        return torch.device("cuda")
    return torch.device("cpu")


def data_collect_noa(scenario_path: str, s_id: str, L: int, root: str = "noa_nn", force: bool = False, augment: bool = True, offset: int = 20) -> None:
    if augment:
        save_path_X = os.path.join(root, "X_aug", s_id, "X.npy")
        save_path_y = os.path.join(root, "y_aug", s_id, "y.npy")
        os.makedirs(os.path.join(root, "X_aug", s_id), exist_ok=True)
        os.makedirs(os.path.join(root, "y_aug", s_id), exist_ok=True)
    else:
        save_path_X = os.path.join(root, "X", s_id, "X.npy")
        save_path_y = os.path.join(root, "y", s_id, "y.npy")
        os.makedirs(os.path.join(root, "X", s_id), exist_ok=True)
        os.makedirs(os.path.join(root, "y", s_id), exist_ok=True)
    
    if (not force) and os.path.exists(save_path_X) and os.path.exists(save_path_y):
        return

    # 1. Load and Extract Features (Same as before)
    pairs_file = os.path.join(scenario_path, "pair.txt")
    with open(pairs_file) as f:
        line = f.readline().strip()
        if line: 
            items = line.split()
            if len(items) == 2:
                query_id, ref_id = items[0], items[1]
    
    query_feature_file = f"features/chroma_stft_norm2/{query_id}.npy"
    ref_feature_file = f"features/chroma_stft_norm2/{ref_id}.npy"
    Fq = np.load(query_feature_file)
    Fref = np.load(ref_feature_file)

    # 2. Get the NOA Path and Cumulative Cost Matrix D
    # Note: alignNOA returns time; we need the frame indices from your NOA implementation
    # Assuming path_indices is (2, N) and D is (Query_Len, Ref_Len)
    hop = constants.DEFAULT_HOP_LENGTH / constants.DEFAULT_SR
    # ref_start_time is usually 0 unless you are offsetting the reference
    path_sec, D_matrix = alignNOA(Fq, Fref, return_D=True, hop_sec=hop, ref_start_time=0)
    
    # 3. Get Ground Truth (Same as your snippet)
    wp = np.load(f"experiments/DTW/{s_id}/hyp.npy") / hop
    GT_dict = {entry[0]: entry[1] for entry in wp.T}

    X, y = [], []


    # 4. Generate NOA Horizontal Windows
    # Since path_sec is [2, N] in seconds, iterate through columns
    for col in range(path_sec.shape[1]):
        # Convert seconds back to frame indices
        q_frame = int(round(path_sec[0, col] / hop))
        r_frame = int(round(path_sec[1, col] / hop))

        r_frame_aug = []
        if not augment:
            offset = 0

        for i in range(-offset, offset + 1):
            if r_frame + i >= 0 and r_frame + i <= Fref.shape[1]:
                r_frame_aug.append(r_frame + i)
            
        for r_frame in r_frame_aug:
            if q_frame not in GT_dict:
                continue
                
            start_j = r_frame - L
            end_j = r_frame + L
            
            # Boundary handling with padding
            if start_j < 0 or end_j >= D_matrix.shape[1]:
                valid_start = max(0, start_j)
                valid_end = min(D_matrix.shape[1] - 1, end_j)
                raw_slice = D_matrix[q_frame, valid_start : valid_end + 1]
                x_vec = np.pad(raw_slice, (max(0, -start_j), max(0, end_j - (D_matrix.shape[1]-1))), constant_values=np.inf)
            else:
                x_vec = D_matrix[q_frame, start_j : end_j + 1]

            # --- CRUCIAL: Pre-process x_vec for NN ---
            mask = np.isinf(x_vec)
            if np.all(mask): continue
            
            # Replace inf and apply local min-max scaling
            x_vec[mask] = np.max(x_vec[~mask]) * 1.1
            v_min, v_max = x_vec.min(), x_vec.max()
            x_vec = (x_vec - v_min) / (v_max - v_min) if v_max > v_min else np.zeros_like(x_vec)

            # Labeling
            gt_ref = GT_dict[q_frame]
            if start_j <= gt_ref <= end_j:
                label = gt_ref - start_j
                X.append(x_vec)
                y.append(label)
    
    os.makedirs(os.path.join(root, "X", s_id), exist_ok=True)
    os.makedirs(os.path.join(root, "y", s_id), exist_ok=True)
    os.makedirs(os.path.join(root, "all_coords", s_id), exist_ok=True)

    
    np.save(save_path_X, X)
    np.save(save_path_y, y)

def preprocess_once(scenario_root: str, L: int, root: str = "noa_nn", force: bool = False, augment: bool = True, offset: int = 20) -> None:
    """Prepare per-scenario features once; skips scenarios that already have cached outputs."""
    scenario_ids = sorted([sid for sid in os.listdir(scenario_root) if os.path.isdir(os.path.join(scenario_root, sid))])
    for s_id in tqdm(scenario_ids, desc="Preprocessing scenarios"):
        scenario_path = os.path.join(scenario_root, s_id)
        data_collect_noa(scenario_path, s_id, L=L, root=root, force=force, augment=augment, offset=offset)


class SimpleDenseNet(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 256, output_dim: int = 1):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def load_dataset(root: str = "noa_nn", augment: bool = False):
    all_X = []
    all_y = []
    all_coords = []
    scenario_ids = []
    if augment:
        x_root = os.path.join(root, "X_aug")
        y_root = os.path.join(root, "y_aug")
        coords_root = os.path.join(root, "all_coords_aug")
    else:
        x_root = os.path.join(root, "X")
        y_root = os.path.join(root, "y")
        coords_root = os.path.join(root, "all_coords")

    for s_id in sorted(os.listdir(x_root)):
        x_path = os.path.join(x_root, s_id, "X.npy")
        y_path = os.path.join(y_root, s_id, "y.npy")
        coords_path = os.path.join(coords_root, s_id, "all_coords.npy")

        if not (os.path.exists(x_path) and os.path.exists(y_path)):
            logger.warning("Skipping %s: missing X or y", s_id)
            continue

        X_i = np.load(x_path, allow_pickle=True)
        y_i = np.load(y_path, allow_pickle=True)

        if len(X_i) != len(y_i):
            logger.warning("Skipping %s: len(X)=%d != len(y)=%d", s_id, len(X_i), len(y_i))
            continue

        all_X.append(X_i)
        all_y.append(y_i)

        if os.path.exists(coords_path):
            coords_i = np.load(coords_path, allow_pickle=True)
            all_coords.append(coords_i)

        scenario_ids.extend([s_id] * len(y_i))

    X = np.concatenate(all_X, axis=0).astype(np.float32)
    y = np.concatenate(all_y, axis=0).astype(np.float32)

    logger.info("X shape: %s", X.shape)
    logger.info("y shape: %s", y.shape)

    if len(all_coords) > 0:
        all_coords = np.concatenate(all_coords, axis=0)
        logger.info("all_coords shape: %s", all_coords.shape)

    scenario_ids = np.array(scenario_ids)
    logger.info("scenario_ids shape: %s", scenario_ids.shape)

    logger.info("X dtype: %s", X.dtype)
    logger.info("y dtype: %s", y.dtype)
    logger.info("X min/max: %s %s", X.min(), X.max())
    logger.info("y min/max: %s %s", y.min(), y.max())
    logger.info("unique label count: %d", len(np.unique(y)))

    return X, y, scenario_ids


def train_model(
    X: np.ndarray,
    y: np.ndarray,
    scenario_ids: np.ndarray,
    L: int,
    hidden_dim: int,
    batch_size: int,
    lr: float,
    num_epochs: int,
    patience: int = 10,
    train_ratio: float = 0.8,
    seed: int = 42,
    device: str = "auto",
):
    dev = resolve_training_device(device)
    logger.info("Training device: %s", dev)

    X_tensor = torch.tensor(X, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.float32)

    if not (0.0 < train_ratio < 1.0):
        raise ValueError("train_ratio must be between 0 and 1 (exclusive).")

    unique_scenarios = np.unique(scenario_ids)
    rng = np.random.default_rng(seed)
    shuffled_scenarios = unique_scenarios.copy()
    rng.shuffle(shuffled_scenarios)

    n_train_scenarios = max(1, int(round(len(shuffled_scenarios) * train_ratio)))
    if n_train_scenarios >= len(shuffled_scenarios):
        n_train_scenarios = max(1, len(shuffled_scenarios) - 1)

    train_scenarios_arr = shuffled_scenarios[:n_train_scenarios]
    test_scenarios_arr = shuffled_scenarios[n_train_scenarios:]

    train_scenarios = set(train_scenarios_arr)
    train_mask = np.array([sid in train_scenarios for sid in scenario_ids])
    test_mask = ~train_mask

    train_indices = np.where(train_mask)[0]
    test_indices = np.where(test_mask)[0]

    if len(train_indices) == 0 or len(test_indices) == 0:
        raise ValueError(
            "Scenario-level split produced an empty train/test split. "
            "Please add more scenarios or adjust train_ratio."
        )

    train_dataset = TensorDataset(X_tensor[train_indices], y_tensor[train_indices])
    test_dataset = TensorDataset(X_tensor[test_indices], y_tensor[test_indices])

    logger.info(
        "Scenario split -> train scenarios: %d, test scenarios: %d",
        len(set(scenario_ids[train_indices])),
        len(set(scenario_ids[test_indices])),
    )
    logger.info(
        "Sample split   -> train samples: %d, test samples: %d",
        len(train_indices),
        len(test_indices),
    )

    pin_memory = dev.type == "cuda"
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, pin_memory=pin_memory
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, pin_memory=pin_memory
    )

    input_dim = 2 * L + 1
    model = SimpleDenseNet(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=1).to(dev)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    train_losses = []
    test_losses = []
    test_maes = []
    best_test_loss = float("inf")
    best_state_dict = None
    epochs_no_improve = 0

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0

        for xb, yb in train_loader:
            xb = xb.to(dev, non_blocking=pin_memory)
            yb = yb.to(dev, non_blocking=pin_memory)
            optimizer.zero_grad()
            preds = model(xb).squeeze(1)
            loss = criterion(preds, yb)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * xb.size(0)

        train_loss /= len(train_loader.dataset)
        train_losses.append(train_loss)

        model.eval()
        test_loss = 0.0
        test_abs_error = 0.0

        with torch.no_grad():
            for xb, yb in test_loader:
                xb = xb.to(dev, non_blocking=pin_memory)
                yb = yb.to(dev, non_blocking=pin_memory)
                preds = model(xb).squeeze(1)
                loss = criterion(preds, yb)
                test_loss += loss.item() * xb.size(0)
                test_abs_error += torch.abs(preds - yb).sum().item()

        test_loss /= len(test_loader.dataset)
        test_mae = test_abs_error / len(test_loader.dataset)

        test_losses.append(test_loss)
        test_maes.append(test_mae)

        logger.info(
            "Epoch %d: train_loss=%.4f, test_loss=%.4f, test_mae=%.4f",
            epoch + 1,
            train_loss,
            test_loss,
            test_mae,
        )

        if test_loss < best_test_loss:
            best_test_loss = test_loss
            best_state_dict = copy.deepcopy(model.state_dict())
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= patience:
            logger.info(
                "Early stopping triggered at epoch %d (patience=%d). Best test_loss=%.4f",
                epoch + 1,
                patience,
                best_test_loss,
            )
            break

    if best_state_dict is not None:
        model.load_state_dict(best_state_dict)
        logger.info("Restored best model weights with test_loss=%.4f", best_test_loss)

    return model, X_tensor, y_tensor, train_losses, test_losses, test_maes, train_scenarios_arr, test_scenarios_arr


def save_split_txt(
    split_out_path: str,
    train_scenarios: np.ndarray,
    test_scenarios: np.ndarray,
    train_ratio: float,
    seed: int,
) -> None:
    split_dir = os.path.dirname(split_out_path)
    if split_dir:
        os.makedirs(split_dir, exist_ok=True)

    with open(split_out_path, "w", encoding="utf-8") as f:
        f.write("# Scenario-level split\n")
        f.write(f"train_ratio={train_ratio}\n")
        f.write(f"test_ratio={1.0 - train_ratio}\n")
        f.write(f"seed={seed}\n")
        f.write(f"train_scenarios_count={len(train_scenarios)}\n")
        f.write(f"test_scenarios_count={len(test_scenarios)}\n\n")

        f.write("[TRAIN_SCENARIOS]\n")
        for sid in train_scenarios:
            f.write(f"{sid}\n")

        f.write("\n[TEST_SCENARIOS]\n")
        for sid in test_scenarios:
            f.write(f"{sid}\n")

    logger.info("Saved train/test split to: %s", split_out_path)


def main():
    parser = argparse.ArgumentParser(description="DenseNN attempt script converted from notebook.")

    parser.add_argument("--scenario-root", type=str, default="scenarios")
    parser.add_argument("--noa-nn-root", type=str, default="noa_nn")

    parser.add_argument("--L", type=int, default=100)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument(
        "--patience",
        type=int,
        default=10,
        help="Early stopping patience on validation (test) loss.",
    )
    parser.add_argument("--model-out", type=str, default="noa_nn/model.pth")
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.6,
        help="Fraction of scenarios used for training (e.g., 0.6 => 60/40 train/test).",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cuda", "cpu"],
        help="Training device: auto (CUDA if available), cuda, or cpu.",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Logging verbosity level.",
    )
    parser.add_argument(
        "--log-file",
        type=str,
        default=None,
        help="Path to training log file. Defaults to <noa-nn-root>/logs/training.log.",
    )
    parser.add_argument(
        "--split-out",
        type=str,
        default="noa_nn/train_test_split.txt",
        help="Path to save the scenario train/test split as a text file.",
    )

    parser.add_argument(
        "--force-preprocess",
        action="store_true",
        help="Recompute preprocessing even if cached arrays exist.",
    )
    parser.add_argument(
        "--skip-preprocess",
        action="store_true",
        help="Skip preprocessing and use already prepared dense_nn/X and dense_nn/y.",
    )
    parser.add_argument(
        "--augment",
        action="store_true",
        help="Whether to apply random jitter augmentation to reference frame indices during preprocessing.",
    )
    parser.add_argument(
        "--augment-offset",
        type=int,
        default=100,
        help="Maximum frame offset for random jitter augmentation (e.g., 20 means +/- 20 frames).",
    )
    args = parser.parse_args()

    log_target = args.log_file or os.path.join(args.noa_nn_root, "logs", "training.log")
    if log_target.endswith(os.sep) or (os.path.exists(log_target) and os.path.isdir(log_target)):
        log_file = os.path.join(log_target, "training.log")
    else:
        log_file = log_target
    log_dir = os.path.dirname(log_file)
    if log_dir:
        os.makedirs(log_dir, exist_ok=True)

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(log_file, mode="a", encoding="utf-8"),
        ],
        force=True,
    )
    logger.info("Logging to file: %s", log_file)

    if not args.skip_preprocess:
        preprocess_once(
            scenario_root=args.scenario_root,
            L=args.L,
            root=args.noa_nn_root,
            force=args.force_preprocess,
            augment=args.augment,
            offset=args.augment_offset
        )

    X, y, scenario_ids = load_dataset(root=args.noa_nn_root, augment=args.augment)

    model, _, _, _, _, _, train_scenarios, test_scenarios = train_model(
        X=X,
        y=y,
        scenario_ids=scenario_ids,
        L=args.L,
        hidden_dim=args.hidden_dim,
        batch_size=args.batch_size,
        lr=args.lr,
        num_epochs=args.epochs,
        patience=args.patience,
        train_ratio=args.train_ratio,
        seed=args.seed,
        device=args.device,
    )

    save_split_txt(
        split_out_path=args.split_out,
        train_scenarios=train_scenarios,
        test_scenarios=test_scenarios,
        train_ratio=args.train_ratio,
        seed=args.seed,
    )

    model_out_dir = os.path.dirname(args.model_out)
    if model_out_dir:
        os.makedirs(model_out_dir, exist_ok=True)
    torch.save(model.state_dict(), args.model_out)
    logger.info("Saved model state_dict to: %s", args.model_out)


if __name__ == "__main__":
    main()