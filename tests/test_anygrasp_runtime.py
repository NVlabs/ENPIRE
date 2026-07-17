from __future__ import annotations

import zipfile
from pathlib import Path

import enpire.env.forge.cap.utils.anygrasp_runtime as runtime


_POINTER_TEXT = """version https://git-lfs.github.com/spec/v1
oid sha256:deadbeef
size 123
"""
_SUFFIX = ".cpython-311-x86_64-linux-gnu.so"


def _write_fake_elf(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"\x7fELFfake-binary")


def test_binary_path_uses_nearby_existing_elf_when_local_copy_is_lfs_pointer(
    monkeypatch,
    tmp_path: Path,
) -> None:
    repo_root = tmp_path / "repo"
    vendor_root = repo_root / "third_party" / "anygrasp_sdk"
    local_path = vendor_root / "grasp_detection" / "gsnet_versions" / f"gsnet{_SUFFIX}"
    local_path.parent.mkdir(parents=True, exist_ok=True)
    local_path.write_text(_POINTER_TEXT, encoding="utf-8")

    peer_root = tmp_path / "peer"
    peer_path = (
        peer_root
        / "third_party"
        / "anygrasp_sdk"
        / "grasp_detection"
        / "gsnet_versions"
        / f"gsnet{_SUFFIX}"
    )
    _write_fake_elf(peer_path)

    monkeypatch.setattr(runtime, "REPO_ROOT", repo_root)
    monkeypatch.setattr(runtime, "VENDOR_ROOT", vendor_root)
    monkeypatch.setattr(runtime, "_python_ext_suffix", lambda: _SUFFIX)
    monkeypatch.setattr(runtime, "_binary_search_roots", lambda: [tmp_path])
    monkeypatch.setattr(
        runtime,
        "_materialize_lfs_pointer",
        lambda _path: (_ for _ in ()).throw(
            FileNotFoundError("missing local lfs object")
        ),
    )
    monkeypatch.setattr(
        runtime, "_historical_lfs_object_for_repo_path", lambda _path: None
    )

    resolved = runtime._binary_path("gsnet")

    assert resolved == peer_path.resolve()


def test_prepare_anygrasp_runtime_symlinks_runtime_binaries_to_resolved_elf_sources(
    monkeypatch,
    tmp_path: Path,
) -> None:
    repo_root = tmp_path / "repo"
    vendor_root = repo_root / "third_party" / "anygrasp_sdk"
    peer_root = tmp_path / "peer"

    for rel in [
        ("grasp_detection/gsnet_versions", "gsnet"),
        ("license_registration/lib_cxx_versions", "lib_cxx"),
        ("grasp_tracking/tracker_versions", "tracker"),
    ]:
        subdir, stem = rel
        local_path = vendor_root / subdir / f"{stem}{_SUFFIX}"
        local_path.parent.mkdir(parents=True, exist_ok=True)
        local_path.write_text(_POINTER_TEXT, encoding="utf-8")
        peer_path = (
            peer_root / "third_party" / "anygrasp_sdk" / subdir / f"{stem}{_SUFFIX}"
        )
        _write_fake_elf(peer_path)

    license_zip = tmp_path / "license.zip"
    with zipfile.ZipFile(license_zip, "w") as zf:
        zf.writestr("licenseCfg.json", "{}")

    monkeypatch.setattr(runtime, "REPO_ROOT", repo_root)
    monkeypatch.setattr(runtime, "VENDOR_ROOT", vendor_root)
    monkeypatch.setattr(runtime, "RUNTIME_ROOT", tmp_path / "runtime")
    monkeypatch.setattr(runtime, "_python_ext_suffix", lambda: _SUFFIX)
    monkeypatch.setattr(runtime, "_binary_search_roots", lambda: [tmp_path])
    monkeypatch.setattr(
        runtime,
        "_materialize_lfs_pointer",
        lambda _path: (_ for _ in ()).throw(
            FileNotFoundError("missing local lfs object")
        ),
    )
    monkeypatch.setattr(
        runtime, "_historical_lfs_object_for_repo_path", lambda _path: None
    )
    monkeypatch.setattr(
        runtime, "_prebuilt_python_root", lambda _kind: tmp_path / "prebuilt"
    )

    prepared = runtime.prepare_anygrasp_runtime(license_zip=license_zip)

    detect_root = prepared["detect_root"]
    track_root = prepared["track_root"]
    assert (detect_root / "gsnet.so").is_symlink()
    assert (detect_root / "lib_cxx.so").is_symlink()
    assert (track_root / "tracker.so").is_symlink()
    assert (track_root / "lib_cxx.so").is_symlink()
    assert (detect_root / "gsnet.so").resolve().read_bytes().startswith(b"\x7fELF")
    assert (track_root / "tracker.so").resolve().read_bytes().startswith(b"\x7fELF")
