# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path


def test_sam3_resolves_current_processor_config_from_local_cache(
    tmp_path: Path, monkeypatch
) -> None:
    from enpire.env.forge.tools.vision import serve_sam3

    revision = "test-revision"
    repo = tmp_path / "models--facebook--sam3"
    snapshot = repo / "snapshots" / revision
    snapshot.mkdir(parents=True)
    (repo / "refs").mkdir()
    (repo / "refs" / "main").write_text(revision)
    (snapshot / "config.json").write_text("{}")
    (snapshot / "processor_config.json").write_text("{}")

    monkeypatch.setenv("HUGGINGFACE_HUB_CACHE", str(tmp_path))
    processor_ref, processor_kwargs, model_ref, model_kwargs = (
        serve_sam3._resolve_sam3_refs()
    )

    assert processor_ref == snapshot
    assert model_ref == snapshot
    assert processor_kwargs == {"local_files_only": True}
    assert model_kwargs == {"local_files_only": True}
