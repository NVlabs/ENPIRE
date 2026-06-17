#!/usr/bin/env python3
"""Download and validate the facebook/sam3 HuggingFace model.

Usage:
    # Download to default HF cache (~/.cache/huggingface/hub/)
    uv run python scripts/download_sam3.py

    # Download to custom path (e.g. Lustre shared storage)
    HF_HOME=/mnt/amlfs-02/shared/wenli_vla_ft/pretrain/huggingface \
        uv run python scripts/download_sam3.py

    # Requires HF_TOKEN set for first download (gated model).
    # After cached, works offline without token.

Environment variables:
    HF_TOKEN          — HuggingFace token (required for first download)
    HF_HOME           — HuggingFace cache root (default: ~/.cache/huggingface)
    HUGGINGFACE_HUB_CACHE — Direct hub cache path (overrides HF_HOME/hub)
"""

import os
import sys
from pathlib import Path


def main():
    repo_id = "facebook/sam3"

    # Resolve cache location
    hf_home = os.environ.get("HF_HOME", str(Path.home() / ".cache" / "huggingface"))
    hub_cache = os.environ.get("HUGGINGFACE_HUB_CACHE", str(Path(hf_home) / "hub"))
    print(f"[sam3] HF cache: {hub_cache}")

    # Check if already cached
    repo_dir = Path(hub_cache) / "models--facebook--sam3"
    if repo_dir.exists():
        snapshots = list((repo_dir / "snapshots").iterdir()) if (repo_dir / "snapshots").exists() else []
        if snapshots:
            # Check if snapshot has actual model files (not just LICENSE/README)
            snapshot = snapshots[0]
            files = list(snapshot.iterdir())
            file_names = [f.name for f in files]
            if "preprocessor_config.json" in file_names and "config.json" in file_names:
                print(f"[sam3] Already cached at {snapshot}")
                print(f"[sam3] Files: {len(files)} ({', '.join(sorted(file_names)[:5])}...)")
                _validate(str(snapshot))
                return
            else:
                print(f"[sam3] Incomplete cache found ({file_names}), re-downloading...")
                import shutil
                shutil.rmtree(repo_dir)

    # Download
    token = os.environ.get("HF_TOKEN")
    if not token:
        print("[sam3] ERROR: HF_TOKEN not set. facebook/sam3 is a gated model.")
        print("[sam3] 1. Accept license at https://huggingface.co/facebook/sam3")
        print("[sam3] 2. Set HF_TOKEN=hf_xxx and rerun")
        sys.exit(1)

    print(f"[sam3] Downloading {repo_id} (gated model, token required)...")
    from huggingface_hub import snapshot_download

    path = snapshot_download(
        repo_id,
        token=token,
        cache_dir=hub_cache,
    )
    print(f"[sam3] Downloaded to {path}")
    _validate(path)


def _validate(snapshot_path: str):
    """Validate that the model loads correctly."""
    print("[sam3] Validating model load...")
    try:
        from transformers import Sam3Model, Sam3Processor

        proc = Sam3Processor.from_pretrained(snapshot_path)
        print(f"[sam3] Processor: OK ({type(proc).__name__})")

        # Only load model if GPU available (large model)
        import torch
        if torch.cuda.is_available():
            model = Sam3Model.from_pretrained(snapshot_path, torch_dtype=torch.float16)
            print(f"[sam3] Model: OK ({sum(p.numel() for p in model.parameters()) / 1e6:.0f}M params)")
            del model
            torch.cuda.empty_cache()
        else:
            print("[sam3] Model: skipped (no GPU, processor-only validation)")

        print("[sam3] Validation PASSED")
    except Exception as e:
        print(f"[sam3] Validation FAILED: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
