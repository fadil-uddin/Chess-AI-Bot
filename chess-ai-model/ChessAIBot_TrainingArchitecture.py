"""
Chess AI Bot - Training Architecture


This script defines the training pipeline for the custom chess AI engine.

Features:
- Memory-mapped dataset loader with LRU caching for scalable training.
- Residual CNN architecture inspired by AlphaZero (Policy + Value heads).
- Training loop with checkpointing, evaluation, and TensorBoard logging.
- Support for GPU (CUDA / MPS) with automatic device detection.

Author: [Syed Fadil Uddin]
"""

import os, glob, re, gc, bisect, time
from pathlib import Path
from datetime import datetime
from collections import OrderedDict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

# ================================================================
# Configuration
# ================================================================
DATA_DIR      = "data/preparedData"  # directory containing processed npy triplets
SAVE_DIR      = "savedModels"        # model checkpoints & best model storage
RUNS_DIR      = "runs"               # tensorboard logs
EPOCHS        = 10                   # total training epochs
BATCH_SIZE    = 64
LR            = 1e-3
WEIGHT_DECAY  = 1e-4
LABEL_SMOOTH  = 0.1                   # label smoothing for policy head
VALUE_WEIGHT  = 0.25                  # weighting factor for value head loss
SEED          = 42                    # reproducibility

LAST_CKPT = Path(SAVE_DIR) / "last.pt"   # latest checkpoint filename

torch.manual_seed(SEED)
np.random.seed(SEED)
Path(SAVE_DIR).mkdir(exist_ok=True)


# ================================================================
# DATA DISCOVERY
# ================================================================
def discover_pairs(data_dir: str):
    """
    Discover (moves, positions, values) triplet files by shared index.
    Returns:
        list of tuples: [(moves.npy, positions.npy, values.npy), ...]
    """
    def idx_by_pattern(pattern):
        out = {}
        for p in glob.glob(os.path.join(data_dir, pattern)):
            m = re.search(r"(\d+)\.npy$", p)
            if m:
                out[int(m.group(1))] = p
        return out

    mov = idx_by_pattern("moves*.npy")
    pos = idx_by_pattern("positions*.npy")
    val = idx_by_pattern("values*.npy")
    common = sorted(set(mov) & set(pos) & set(val))
    return [(mov[i], pos[i], val[i]) for i in common]


# ================================================================
# DATASET
# ================================================================
class TripletDataset(Dataset):
    """
    Dataset for (positions, moves, values).
    Uses memory mapping for large npy files and LRU caching to reduce file handles.
    """

    def __init__(self, triplets, cache_size: int = 8):
        self.meta, self.cum = [], []
        self.cache_size  = max(1, int(cache_size))
        self._cache: "OrderedDict[int, tuple]" = OrderedDict()

        total = 0
        for mpath, ppath, vpath in triplets:
            try:
                n = min(
                    len(np.load(mpath, mmap_mode="r")),
                    len(np.load(ppath, mmap_mode="r")),
                    len(np.load(vpath, mmap_mode="r")),
                )
                if n <= 0:
                    raise ValueError("empty block")
                self.meta.append({"m": mpath, "x": ppath, "v": vpath, "n": n})
                total += n
                self.cum.append(total)
            except Exception as e:
                print(f"[skip] {mpath} / {ppath} / {vpath} -> {e}")

        if total == 0:
            raise RuntimeError("No valid triplets found.")
        print(f"Indexed {len(self.meta)} blocks, total samples: {total:,}")

    @staticmethod
    def _close(arr):
        """Close memory-mapped numpy arrays explicitly."""
        mm = getattr(arr, "_mmap", None)
        if mm is not None:
            mm.close()

    def _get_block(self, bid):
        """Load a block, with LRU eviction if cache exceeds size."""
        if bid in self._cache:
            self._cache.move_to_end(bid, last=True)
            return self._cache[bid]

        mpath = self.meta[bid]["m"]
        ppath = self.meta[bid]["x"]
        vpath = self.meta[bid]["v"]
        m = np.load(mpath, mmap_mode='r')
        x = np.load(ppath, mmap_mode='r')
        v = np.load(vpath, mmap_mode='r')
        self._cache[bid] = (m, x, v)

        # LRU eviction
        while len(self._cache) > self.cache_size:
            _, (m_old, x_old, v_old) = self._cache.popitem(last=False)
            self._close(m_old); self._close(x_old); self._close(v_old)

        return m, x, v

    def __len__(self): return self.cum[-1]

    def __getitem__(self, idx):
        bid = bisect.bisect_right(self.cum, idx)
        left = 0 if bid == 0 else self.cum[bid-1]
        local = idx - left

        m, x, v = self._get_block(bid)
        xb = torch.from_numpy(x[local].astype(np.float32, copy=True))
        yb = torch.tensor(int(m[local]), dtype=torch.long)
        vb = torch.tensor(float(v[local]), dtype=torch.float32)
        return xb, yb, vb

    def close(self):
        """Release cached memory maps."""
        while self._cache:
            _, (m, x, v) = self._cache.popitem()
            self._close(m); self._close(x); self._close(v)
        gc.collect()

# ================================================================
# MODEL
# ================================================================
class Residual(nn.Module):
    """Residual block for convolutional layers."""
    def __init__(self, c):
        super().__init__()
        self.c1 = nn.Conv2d(c, c, 3, 1, 1, bias=False)
        self.b1 = nn.BatchNorm2d(c)
        self.c2 = nn.Conv2d(c, c, 3, 1, 1, bias=False)
        self.b2 = nn.BatchNorm2d(c)

    def forward(self, x):
        y = F.relu(self.b1(self.c1(x)))
        y = self.b2(self.c2(y))
        return F.relu(x + y)


class PolicyValueNet(nn.Module):
    """
    Combined Policy + Value network.
    - Policy head: predicts next move (classification over action space).
    - Value head: predicts expected game outcome [-1, 1].
    """
    def __init__(self, planes=14, channels=96, nblocks=8, action_size=4672):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(planes, channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(channels), nn.ReLU(inplace=True)
        )
        self.body = nn.Sequential(*[Residual(channels) for _ in range(nblocks)])

        # Policy head
        self.p_head = nn.Sequential(
            nn.Conv2d(channels, 32, 1, bias=False),
            nn.BatchNorm2d(32), nn.ReLU(inplace=True)
        )
        self.p_fc = nn.Linear(32 * 8 * 8, action_size)

        # Value head
        self.v_head = nn.Sequential(
            nn.Conv2d(channels, 1, 1, bias=False),
            nn.GroupNorm(1, 1), nn.ReLU(inplace=True)
        )
        self.v_fc1 = nn.Linear(8 * 8, 128)
        self.v_fc2 = nn.Linear(128, 1)

    def forward(self, x):
        x = x.permute(0, 3, 1, 2).contiguous()  # (B, H, W, C) -> (B, C, H, W)
        h = self.body(self.stem(x))

        # Policy
        p = self.p_head(h).reshape(h.size(0), -1)
        p_out = self.p_fc(p)

        # Value
        v = self.v_head(h).reshape(h.size(0), -1)
        v_out = torch.tanh(self.v_fc2(F.relu(self.v_fc1(v), inplace=True))).squeeze(1)

        return p_out, v_out


# ============================================================
#                   Helper Functions
# ============================================================
def evaluate(model, loader, ce, mse, device):
    """Evaluate model on validation set."""
    model.eval()
    tot = lp_sum = lv_sum = t1 = t5 = mae = 0.0
    with torch.no_grad():
        for xb, yb, vb in loader:
            xb, yb, vb = xb.to(device), yb.to(device), vb.to(device)
            logits, vhat = model(xb)

            bs = xb.size(0)
            lp = ce(logits, yb)
            lv = mse(vhat, vb)

            tot += bs
            lp_sum += lp.item() * bs
            lv_sum += lv.item() * bs
            tk = logits.topk(5, 1).indices
            t1 += (tk[:, 0] == yb).sum().item()
            t5 += (tk == yb.unsqueeze(1)).any(1).sum().item()
            mae += torch.abs(vhat - vb).sum().item()

    return lp_sum / tot, lv_sum / tot, t1 / tot, t5 / tot, mae / tot


def save_checkpoint(ep, step, model, opt, sch, best_val):
    """Save training checkpoint."""
    ckpt = {
        'epoch': ep,
        'step': step,
        'model': model.state_dict(),
        'optim': opt.state_dict(),
        'sched': sch.state_dict(),
        'best_val': best_val
    }
    torch.save(ckpt, LAST_CKPT)


def load_checkpoint(model, opt, sch):
    """Load last checkpoint if available."""
    if LAST_CKPT.exists():
        ckpt = torch.load(LAST_CKPT, map_location='cpu')
        model.load_state_dict(ckpt['model'])
        opt.load_state_dict(ckpt['optim'])
        sch.load_state_dict(ckpt['sched'])
        return ckpt['epoch'] + 1, ckpt['step'] + 1, ckpt.get('best_val', float('inf'))
    return 1, 0, float('inf')

# ================================================================
# TRAIN LOOP
# ================================================================
def train():
    """Main training loop."""
    # device selection
    device = (
        torch.device('cuda') if torch.cuda.is_available() else
        torch.device('mps') if torch.backends.mps.is_available() else
        torch.device('cpu')
    )
    print("Using device:", device)

    # dataset
    triplets = discover_pairs(DATA_DIR)
    ds = TripletDataset(triplets, cache_size=8)
    N = len(ds)
    idx = np.arange(N); np.random.default_rng(SEED).shuffle(idx)
    split = int(0.9 * N)
    train_ds, val_ds = Subset(ds, idx[:split]), Subset(ds, idx[split:])
    pin, nw = (device.type == 'cuda'), (0 if device.type == 'mps' else 2)
    train_loader = DataLoader(train_ds, BATCH_SIZE, shuffle=True, pin_memory=pin, num_workers=nw)
    val_loader   = DataLoader(val_ds,   BATCH_SIZE, shuffle=False, pin_memory=pin, num_workers=nw)

    # model, optimizer, loss
    model = PolicyValueNet().to(device)
    opt   = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    sch   = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
    ce    = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTH)
    mse   = nn.MSELoss()

    # checkpoint resume
    start_ep, global_step, best_val = load_checkpoint(model, opt, sch)
    print(f"Resuming from epoch {start_ep}, step {global_step}, best_val {best_val:.4f}")

    # logging
    writer = SummaryWriter(f"{RUNS_DIR}/pv_{datetime.now():%Y%m%d_%H%M%S}")

    # training loop
    for ep in range(start_ep, EPOCHS + 1):
        model.train()
        run_p = run_v = seen = 0

        for xb, yb, vb in tqdm(train_loader, desc=f"Epoch {ep}/{EPOCHS}", leave=False):
            xb, yb, vb = xb.to(device), yb.to(device), vb.to(device)

            # forward + backward
            opt.zero_grad(set_to_none=True)
            logits, vhat = model(xb)
            lp, lv = ce(logits, yb), mse(vhat, vb)
            (lp + VALUE_WEIGHT * lv).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

            # logging
            bs = xb.size(0)
            run_p += lp.item() * bs
            run_v += lv.item() * bs
            seen += bs
            global_step += 1

            # periodic checkpoint
            if global_step % 10000 == 0:
                save_checkpoint(ep, global_step, model, opt, sch, best_val)

        sch.step()
        tr_p, tr_v = run_p / seen, run_v / seen
        va_p, va_v, t1, t5, mae = evaluate(model, val_loader, ce, mse, device)
        tot_val = va_p + VALUE_WEIGHT * va_v

        # tensorboard
        writer.add_scalars("loss/policy", {"train": tr_p, "val": va_p}, ep)
        writer.add_scalars("loss/value", {"train": tr_v, "val": va_v}, ep)
        writer.add_scalar ("val/total", tot_val, ep)

        print(f"EP{ep:02d} pol {tr_p:.4f}/{va_p:.4f} valL {tr_v:.4f}/{va_v:.4f} "
              f"top1 {t1:.3f} top5 {t5:.3f} vMAE {mae:.3f}")

        # save best
        if tot_val < best_val - 1e-4:
            best_val = tot_val
            best_path = Path(SAVE_DIR) / f"best_{best_val:.4f}.pt"
            torch.save(model.state_dict(), best_path)
            with open(Path(SAVE_DIR) / "bestModel.txt", "w") as f:
                f.write(f"{best_val}\n{best_path}")
            print(" >> new best model:", best_val)

        # save last checkpoint
        save_checkpoint(ep, global_step, model, opt, sch, best_val)

    writer.close()
    print("Best total-val loss:", best_val)


# ================================================================
# ENTRYPOINT
# ================================================================
if __name__ == "__main__":
    # quick startup benchmark
    t0 = time.time()
    triplets = discover_pairs(DATA_DIR)
    ds = TripletDataset(triplets, cache_size=8)
    print(f"[t+{time.time()-t0:5.1f}s] TripletDataset ready")
    loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    print(f"[t+{time.time()-t0:5.1f}s] DataLoader built")
    _ = next(iter(loader))
    print(f"[t+{time.time()-t0:5.1f}s] first batch pulled\n")

    # start training
    train()
