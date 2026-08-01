# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

from wfm_inference.lilypad_entrypoint import _parallelism_axis_flags


def test_parallelism_axis_flags_omitted_when_unconfigured() -> None:
    """An unpinned axis must stay on the preset's own default."""
    assert _parallelism_axis_flags({"parallelism_preset": "latency", "num_gpus": 8}) == []


def test_parallelism_axis_flags_emits_configured_axes() -> None:
    flags = _parallelism_axis_flags(
        {"parallelism_preset": "latency", "num_gpus": 8, "dp_shard_size": 1, "cp_size": 4}
    )

    assert flags == ["--dp-shard-size=1", "--cp-size=4"]


def test_parallelism_axis_flags_emits_all_four_in_stable_order() -> None:
    flags = _parallelism_axis_flags(
        {
            "cfgp_size": 2,
            "cp_size": 4,
            "dp_replicate_size": 1,
            "dp_shard_size": 1,
        }
    )

    assert flags == [
        "--dp-shard-size=1",
        "--dp-replicate-size=1",
        "--cp-size=4",
        "--cfgp-size=2",
    ]


def test_parallelism_axis_flags_keeps_explicit_zero() -> None:
    """0 means "auto" to the inference CLI, and is distinct from unset."""
    assert _parallelism_axis_flags({"dp_shard_size": 0}) == ["--dp-shard-size=0"]
