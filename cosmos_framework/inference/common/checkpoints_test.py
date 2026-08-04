# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file

from cosmos_framework.inference.common.checkpoints import (
    _AVAE_LEGACY_CKPT_NAME,
    _AVAE_LEGACY_JSON_NAME,
    _atomic_write,
    _materialize_avae_ckpt,
)


def test_atomic_write_destination_is_never_partially_visible(tmp_path: Path) -> None:
    """The destination must not exist until it is complete.

    A peer rank tests ``exists()`` to decide whether to skip materialization; if
    an in-progress write is visible it reads a truncated archive.
    """
    dest = tmp_path / "out.bin"
    visible_during_write = []

    def write(tmp: Path) -> None:
        tmp.write_bytes(b"first")
        visible_during_write.append(dest.exists())
        tmp.write_bytes(b"first-then-complete")

    _atomic_write(dest, write)

    assert visible_during_write == [False]
    assert dest.read_bytes() == b"first-then-complete"


def test_atomic_write_leaves_no_temp_behind_on_failure(tmp_path: Path) -> None:
    dest = tmp_path / "out.bin"

    def write(tmp: Path) -> None:
        tmp.write_bytes(b"partial")
        raise RuntimeError("boom")

    with pytest.raises(RuntimeError, match="boom"):
        _atomic_write(dest, write)

    assert not dest.exists()
    assert not list(tmp_path.glob(f".{dest.name}.tmp.*"))


def _write_fake_sound_tokenizer(local: Path, num_blocks: int = 2) -> None:
    """Write the minimal ``sound_tokenizer/`` inputs the shim converts."""
    tensors = {"decoder.conv1.weight": torch.zeros(2, 2)}
    for block in range(num_blocks):
        tensors[f"decoder.block.{block}.snake1.alpha"] = torch.zeros(1, 4, 1)
        tensors[f"decoder.block.{block}.conv_t1.weight"] = torch.zeros(2, 2)
        tensors[f"decoder.block.{block}.res_unit1.conv1.weight"] = torch.zeros(2, 2)
    save_file(tensors, str(local / "diffusion_pytorch_model.safetensors"))
    (local / "config.json").write_text(json.dumps({"model_type": "avae"}))


def test_materialize_avae_ckpt_produces_loadable_legacy_layout(tmp_path: Path) -> None:
    _write_fake_sound_tokenizer(tmp_path)

    _materialize_avae_ckpt(str(tmp_path))

    checkpoint = torch.load(str(tmp_path / _AVAE_LEGACY_CKPT_NAME), weights_only=True)
    state_dict = checkpoint["state_dict"]
    assert not any(k.startswith("decoder.block.") for k in state_dict)
    assert state_dict["decoder.layers.1.layers.0.alpha"].shape == (4,)  # [1, C, 1] -> [C]
    assert (tmp_path / _AVAE_LEGACY_JSON_NAME).exists()


def test_materialize_avae_ckpt_is_idempotent(tmp_path: Path) -> None:
    _write_fake_sound_tokenizer(tmp_path)

    _materialize_avae_ckpt(str(tmp_path))
    first = (tmp_path / _AVAE_LEGACY_CKPT_NAME).read_bytes()
    _materialize_avae_ckpt(str(tmp_path))

    assert (tmp_path / _AVAE_LEGACY_CKPT_NAME).read_bytes() == first


def test_materialize_avae_ckpt_survives_concurrent_ranks(tmp_path: Path) -> None:
    """Every rank of a torchrun job runs this hook against one shared directory.

    Before the outputs were published atomically, a rank that observed a
    half-written ``.ckpt`` skipped the write and went straight to
    ``torch.load``, which failed with "PytorchStreamReader failed reading zip
    archive: failed finding central directory".
    """
    _write_fake_sound_tokenizer(tmp_path)

    ranks = 8
    with ProcessPoolExecutor(max_workers=ranks) as pool:
        # Raises here if any rank failed.
        list(pool.map(_materialize_avae_ckpt, [str(tmp_path)] * ranks))

    checkpoint = torch.load(str(tmp_path / _AVAE_LEGACY_CKPT_NAME), weights_only=True)
    assert "state_dict" in checkpoint
