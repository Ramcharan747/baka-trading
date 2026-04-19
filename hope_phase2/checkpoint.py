"""
HuggingFace checkpointing for HOPE Phase 2.
Saves/loads model + optimizer + Titans states to HF Hub.
"""
from __future__ import annotations

import json
import os

import torch
from huggingface_hub import HfApi, hf_hub_download

HF_REPO_ID = "Baka7/hope-finance"


def save_checkpoint(model, optimizer, states, epoch, step, metrics,
                    hf_token=None, local_path="/content/checkpoints"):
    """
    Save full training state locally + upload to HuggingFace Hub.
    Call after every epoch.
    """
    os.makedirs(local_path, exist_ok=True)

    # Serialize Titans states (list of dicts with tensors/lists)
    states_serializable = []
    for layer_state in states:
        s = {}
        for k, v in layer_state.items():
            if isinstance(v, torch.Tensor):
                s[k] = v.cpu().tolist()
            elif isinstance(v, list):
                # List of per-batch memory matrices
                s[k] = [m.cpu().tolist() if isinstance(m, torch.Tensor) else m
                        for m in v]
            else:
                s[k] = v
        states_serializable.append(s)

    checkpoint = {
        'epoch': epoch,
        'step': step,
        'model_state': model.state_dict(),
        'optimizer_state': optimizer.state_dict(),
        'titans_states': states_serializable,
        'metrics': metrics,
    }

    ckpt_file = f"{local_path}/checkpoint_epoch{epoch:03d}.pt"
    torch.save(checkpoint, ckpt_file)

    # Always save a "latest" pointer for easy local resume
    latest = {"epoch": epoch, "step": step, "metrics": metrics}
    with open(f"{local_path}/latest.json", "w") as fp:
        json.dump(latest, fp)

    print(f"  Saved local checkpoint: {ckpt_file}")

    # Upload to HuggingFace Hub (token passed explicitly)
    try:
        api = HfApi()
        api.upload_file(
            path_or_fileobj=ckpt_file,
            path_in_repo=f"checkpoints/checkpoint_epoch{epoch:03d}.pt",
            repo_id=HF_REPO_ID,
            repo_type="model",
            token=hf_token,
        )
        api.upload_file(
            path_or_fileobj=f"{local_path}/latest.json",
            path_in_repo="checkpoints/latest.json",
            repo_id=HF_REPO_ID,
            repo_type="model",
            token=hf_token,
        )
        print(f"  ✅ Uploaded to HF: epoch {epoch}")
    except Exception as e:
        print(f"  ⚠️  HF upload failed: {e}")
        print(f"  Checkpoint saved locally only: {ckpt_file}")


def load_checkpoint(model, optimizer, device="cpu",
                    local_path="/content/checkpoints", epoch=None):
    """
    Load latest (or specific epoch) checkpoint from HuggingFace Hub.
    Returns: (model, optimizer, states, start_epoch, step)
    """
    os.makedirs(local_path, exist_ok=True)

    # Find latest epoch if not specified
    if epoch is None:
        latest_local = os.path.join(local_path, "latest.json")
        try:
            if os.path.exists(latest_local):
                with open(latest_local) as fp:
                    latest = json.load(fp)
                epoch = latest['epoch']
                print(f"  [Local] Found latest.json (epoch {epoch})")
            else:
                hf_hub_download(
                    repo_id=HF_REPO_ID, repo_type="model",
                    filename="checkpoints/latest.json",
                    local_dir=local_path,
                )
                hf_latest = os.path.join(local_path, "checkpoints", "latest.json")
                with open(hf_latest) as fp:
                    latest = json.load(fp)
                epoch = latest['epoch']
                print(f"  [HF] Found latest.json (epoch {epoch})")
        except Exception:
            print("  No checkpoint found, starting from scratch")
            return model, optimizer, None, 0, 0

    # Load checkpoint
    ckpt_filename = f"checkpoint_epoch{epoch:03d}.pt"
    local_file = os.path.join(local_path, ckpt_filename)
    
    if os.path.exists(local_file):
        print(f"  [Local] Loading {ckpt_filename}")
    else:
        try:
            local_file = hf_hub_download(
                repo_id=HF_REPO_ID, repo_type="model",
                filename=f"checkpoints/{ckpt_filename}",
                local_dir=local_path,
            )
            print(f"  [HF] Downloaded {ckpt_filename}")
        except Exception as e:
            print(f"  Failed to load {ckpt_filename}: {e}")
            return model, optimizer, None, 0, 0

    checkpoint = torch.load(local_file, map_location=device, weights_only=False)

    # Restore model and optimizer
    model.load_state_dict(checkpoint['model_state'])
    optimizer.load_state_dict(checkpoint['optimizer_state'])

    # Restore Titans states
    states = []
    for layer_state_raw in checkpoint['titans_states']:
        s = {}
        for k, v in layer_state_raw.items():
            if isinstance(v, list) and len(v) > 0:
                if isinstance(v[0], list):
                    # List of per-batch matrices
                    s[k] = [torch.tensor(m, device=device) for m in v]
                else:
                    s[k] = torch.tensor(v, device=device)
            else:
                s[k] = v
        states.append(s)

    print(f"  Resumed from epoch {checkpoint['epoch']}, step {checkpoint['step']}")
    print(f"  Previous metrics: {checkpoint['metrics']}")

    return model, optimizer, states, checkpoint['epoch'] + 1, checkpoint['step']
