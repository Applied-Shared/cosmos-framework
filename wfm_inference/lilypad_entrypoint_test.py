# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

from wfm_inference.lilypad_entrypoint import _apply_recipe_overrides, _parallelism_axis_flags


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


def test_apply_recipe_overrides_defaults_max_frames_to_num_frames() -> None:
    """A caller who set num_frames almost always meant to bound the output too.

    Transfer inference reads output length off the control video and clamps it
    only with max_frames, so num_frames alone leaves a 350-frame control bundle
    generating in full.
    """
    spec: dict = {}

    _apply_recipe_overrides(spec, {"num_frames": 121})

    assert spec == {"num_frames": 121, "max_frames": 121}


def test_apply_recipe_overrides_respects_explicit_max_frames() -> None:
    """Explicit max_frames wins over the num_frames-derived default."""
    spec: dict = {}

    _apply_recipe_overrides(spec, {"num_frames": 121, "max_frames": 200})

    assert spec["max_frames"] == 200


def test_apply_recipe_overrides_leaves_max_frames_unset_without_num_frames() -> None:
    spec: dict = {}

    _apply_recipe_overrides(spec, {"resolution": "720"})

    assert "max_frames" not in spec


def test_apply_recipe_overrides_defaults_from_spec_json_num_frames() -> None:
    """num_frames may come from the spec.json base, not the overrides dict."""
    spec: dict = {"num_frames": 93}

    _apply_recipe_overrides(spec, {"resolution": "720"})

    assert spec["max_frames"] == 93


def test_apply_recipe_overrides_preserves_existing_max_frames_from_spec_json() -> None:
    spec: dict = {"num_frames": 121, "max_frames": 200}

    _apply_recipe_overrides(spec, {})

    assert spec["max_frames"] == 200
