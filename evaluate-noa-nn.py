import argparse
import logging
from pathlib import Path
from datetime import datetime

from noa import alignNOA
import numpy as np
import torch
import torch.nn as nn
import librosa as lb
from tqdm import tqdm

import utils.constants as constants
# import noa_nn.data_collection as data_col
from online_alignment.cost import cosine


class SimpleDenseNet(nn.Module):
    def __init__(self, input_dim, hidden_dim=256, num_classes=None):
        super().__init__()
        if num_classes is None:
            num_classes = input_dim

        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x):
        return self.net(x)


def get_step_vector(index: int, L: int):
    if index < L - 1:
        return (0, 1)   # top edge
    elif index == L - 1:
        return (1, 1)   # diagonal
    else:
        return (1, 0)       # right edge


def get_step_class(index: int, L: int):
    if index < L - 1:
        return 0   # top / query-only
    elif index == L - 1:
        return 1   # diagonal
    else:
        return 2   # right / reference-only


def setup_logger(log_file: Path, verbose: bool = False):
    log_file.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("noa_nn_eval")
    logger.setLevel(logging.DEBUG if verbose else logging.INFO)
    logger.handlers.clear()

    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    fh = logging.FileHandler(log_file)
    fh.setFormatter(fmt)
    fh.setLevel(logging.DEBUG if verbose else logging.INFO)

    sh = logging.StreamHandler()
    sh.setFormatter(fmt)
    sh.setLevel(logging.INFO)

    logger.addHandler(fh)
    logger.addHandler(sh)
    return logger


def build_input_noa(D_matrix, q_idx, r_idx, L):
    """
    Extracts a horizontal window of size 2L+1 centered at r_idx for the current q_idx.
    """
    start_j = r_idx - L
    end_j = r_idx + L
    
    # Boundary handling with padding
    if start_j < 0 or end_j >= D_matrix.shape[1]:
        valid_start = max(0, start_j)
        valid_end = min(D_matrix.shape[1] - 1, end_j)
        raw_slice = D_matrix[q_idx, valid_start : valid_end + 1]
        
        # Pad with inf to maintain 2L+1 size
        x_vec = np.pad(raw_slice, 
                       (max(0, -start_j), max(0, end_j - (D_matrix.shape[1]-1))), 
                       constant_values=np.inf)
    else:
        x_vec = D_matrix[q_idx, start_j : end_j + 1]

    # Pre-process same as training
    mask = np.isinf(x_vec)
    if not np.all(mask):
        x_vec[mask] = np.max(x_vec[~mask]) * 1.1
        v_min, v_max = x_vec.min(), x_vec.max()
        if v_max > v_min:
            x_vec = (x_vec - v_min) / (v_max - v_min)
        else:
            x_vec = np.zeros_like(x_vec)
    else:
        x_vec = np.zeros_like(x_vec)
        
    return x_vec


@torch.no_grad()
def evaluate_scenario(model, scenario_id: str, scenarios_dir: Path, out_dir: Path, L: int, logger):
    query_wav = scenarios_dir / scenario_id / "query.wav"
    ref_wav = scenarios_dir / scenario_id / "ref.wav"

    if not query_wav.exists() or not ref_wav.exists():
        logger.warning(f"[{scenario_id}] Missing wav files. Skipping.")
        return

    y_q, _ = lb.load(str(query_wav), sr=constants.DEFAULT_SR)
    y_r, _ = lb.load(str(ref_wav), sr=constants.DEFAULT_SR)

    Fq = lb.feature.chroma_stft(
        y=y_q, center=False, norm=2, hop_length=constants.DEFAULT_HOP_LENGTH
    )
    Fref = lb.feature.chroma_stft(
        y=y_r, center=False, norm=2, hop_length=constants.DEFAULT_HOP_LENGTH
    )

    # 1. Run the base NOA alignment
    # Note: alignNOA must return D_matrix
    path_sec, D_matrix = alignNOA(Fq, Fref, return_D=True, hop_sec=(constants.DEFAULT_HOP_LENGTH / constants.DEFAULT_SR))
    
    # 2. Setup Ground Truth for scoring
    C_gt = cosine.cosine_dist_vec2vec(Fref, Fq)
    _, wp = lb.sequence.dtw(C=C_gt, backtrack=True)
    GT_dict = {q: r for r, q in wp[::-1]}

    corrected_path = []
    errors = []

    # 3. Iterate through NOA path and let NN correct it
    for col in range(path_sec.shape[1]):
        q_frame = int(round(path_sec[0, col] / (constants.DEFAULT_HOP_LENGTH / constants.DEFAULT_SR)))
        r_frame_noa = int(round(path_sec[1, col] / (constants.DEFAULT_HOP_LENGTH / constants.DEFAULT_SR)))

        # Get NN Input
        x_vec = build_input_noa(D_matrix, q_frame, r_frame_noa, L)
        x_tensor = torch.from_numpy(x_vec).float().unsqueeze(0)
        
        # NN Predicts relative offset within the window [0, 2L]
        logits = model(x_tensor)
        pred_offset = int(torch.argmax(logits, dim=1).item())
        
        # Convert relative offset back to absolute reference frame
        # start_j was (r_frame_noa - L)
        r_frame_corrected = (r_frame_noa - L) + pred_offset
        
        corrected_path.append((q_frame, r_frame_corrected))

        # 4. Scoring
        if q_frame in GT_dict:
            gt_r = GT_dict[q_frame]
            # Absolute error in frames
            errors.append(abs(r_frame_corrected - gt_r))

    # 1. Convert corrected_path (indices) to seconds
    hop_sec = constants.DEFAULT_HOP_LENGTH / constants.DEFAULT_SR
    sim_query = np.array([p[0] for p in corrected_path]) * hop_sec
    sim_ref = np.array([p[1] for p in corrected_path]) * hop_sec
    hyp = np.vstack([sim_query, sim_ref])

    # 2. Setup output directory
    scenario_out = out_dir / scenario_id
    scenario_out.mkdir(parents=True, exist_ok=True)
    # 1. Extract the raw GT path from the DTW result
    # wp is usually (2, N) where wp[0] is ref and wp[1] is query
    gt_ref_frames = wp[0][::-1]
    gt_query_frames = wp[1][::-1]

    # 2. Get your predicted query/ref frames
    pred_query_frames = np.array([p[0] for p in corrected_path])
    pred_ref_frames = np.array([p[1] for p in corrected_path])

    # 3. Interpolate GT: "What should the ref frame be at these specific pred_query_frames?"
    # np.interp(x_new, x_original, y_original)
    true_ref_frames_interp = np.interp(
        pred_query_frames, 
        gt_query_frames, 
        gt_ref_frames
    )

    # 4. Calculate Errors (Now the same size as your prediction)
    abs_errors = np.abs(pred_ref_frames - true_ref_frames_interp)
    mean_mae = np.mean(abs_errors)

    # 5. Tolerance Accuracy (e.g., within 3 frames)
    tolerance = 3
    acc_tol = (np.sum(abs_errors <= tolerance) / len(abs_errors)) * 100.0

    # 6. Save (Using the interpolated GT for side-by-side comparison)
    np.save(scenario_out / "hyp.npy", hyp)
    np.save(scenario_out / "pred_frames.npy", pred_ref_frames)
    np.save(scenario_out / "true_frames_interp.npy", true_ref_frames_interp)

    logger.info(
        f"[{scenario_id}] Path Size: {len(pred_ref_frames)} | "
        f"MAE: {mean_mae:.2f} frames | "
        f"Acc(tol={tolerance}): {acc_tol:.2f}%"
    )

def parse_test_scenarios(split_file: Path):
    if not split_file.exists():
        raise FileNotFoundError(f"Split file not found: {split_file}")

    scenarios = []
    in_test_block = False

    with split_file.open("r", encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()

            if not line or line.startswith("#"):
                continue

            if line.startswith("[") and line.endswith("]"):
                in_test_block = (line == "[TEST_SCENARIOS]")
                continue

            if in_test_block:
                # stop if another section starts unexpectedly (safety)
                if line.startswith("[") and line.endswith("]"):
                    break
                scenarios.append(line)

    if not scenarios:
        raise ValueError(f"No scenarios found under [TEST_SCENARIOS] in {split_file}")

    return scenarios

def main():
    parser = argparse.ArgumentParser("Evaluate NOA-NN model on scenarios and save hyp.npy.")
    parser.add_argument("--model-path", default="noa_nn/model2.pth")
    parser.add_argument("--scenarios-dir", default="scenarios")
    parser.add_argument("--out-dir", default="experiments/NOA_NN")
    parser.add_argument("--scenario", nargs="*", default=None, help="e.g. --scenario s1 s2")
    parser.add_argument("--split-file", default="noa_nn/train_test_split.txt")
    parser.add_argument("--L", type=int, default=100)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--log-file", default=None)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    scenarios_dir = Path(args.scenarios_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.log_file is None:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = out_dir / f"eval_{ts}.log"
    else:
        log_file = Path(args.log_file)

    logger = setup_logger(log_file, verbose=args.verbose)
    logger.info("Starting NOA-NN evaluation.")

    input_dim = 2 * args.L + 1
    model = SimpleDenseNet(
        input_dim=input_dim,
        hidden_dim=args.hidden_dim,
        num_classes=input_dim,
    )
    state = torch.load(args.model_path, map_location="cpu")
    model.load_state_dict(state)
    model.eval()

    if args.scenario:
        scenario_ids = args.scenario
        logger.info("Using scenarios from --scenario.")
    else:
        split_file = Path(args.split_file)
        scenario_ids = parse_test_scenarios(split_file)
        logger.info(f"Using [TEST_SCENARIOS] from split file: {split_file}")

    logger.info(f"Scenarios: {scenario_ids}")

    for sid in scenario_ids:
        evaluate_scenario(
            model=model,
            scenario_id=sid,
            scenarios_dir=scenarios_dir,
            out_dir=out_dir,
            L=args.L,
            logger=logger,
        )

    logger.info("Done.")


if __name__ == "__main__":
    main()

