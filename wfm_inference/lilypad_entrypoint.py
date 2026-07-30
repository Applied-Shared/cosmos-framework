"""Lilypad entrypoint for Cosmos 3 Transfer WFM inference.

Ported from cosmos-transfer2.5. Key differences from 2.5:

- Inference command is ``python -m cosmos_framework.scripts.inference`` (Nano,
  single GPU) or ``torchrun --nproc-per-node=N -m cosmos_framework.scripts.inference``
  (Super, multi-GPU). Framework is installed from this repo's pyproject.toml.
- ``--checkpoint-path`` is a HuggingFace model ID (``Cosmos3-Nano`` /
  ``Cosmos3-Super``), not a local file path. Weights are resolved through the
  offline HF cache pre-staged in OCI.
- ``--experiment`` no longer exists; ``--parallelism-preset=latency`` and
  ``--no-guardrails`` replace 2.5's ``--disable-guardrails``.
- Recipe overrides are a plain top-level dict merge; the ``camera_conditional_frames``
  key was multiview-specific and is dropped.
"""

import json
import logging
import os
import re
import subprocess
import tempfile
from pathlib import Path

import boto3
import botocore.config
import ray

logger = logging.getLogger(__name__)

# Lilypad SDK provides an AIStore-cached read-only boto client that
# accelerates cross-region GETs (e.g. Phoenix → Chicago cache). If the SDK
# isn't available (no cp313 wheel published as of 2026-07-20), fall back to
# plain boto3 — correct but slower on the HF-cache preload path.
try:
    from lilypad.public.sdk_py.cached_file_access.boto import (
        get_readonly_boto_client as _cached_client_factory,
    )
except ImportError:
    _cached_client_factory = None

# OCI S3-compat requires payload signing and disables the default AWS SDK v4
# checksum headers that OCI doesn't support.
_OCI_BOTO_CONFIG = botocore.config.Config(
    s3={"payload_signing_enabled": True},
    request_checksum_calculation="when_required",
    response_checksum_validation="when_required",
)

# Persistent directory on the worker node for shared resources that survive
# across jobs in a batch (HF cache). Lives outside tempdir so it is not cleaned
# up between jobs.
_WORKER_CACHE_DIR = Path("/tmp/wfm_worker_cache")


def _apply_recipe_overrides(spec: dict, recipe_overrides: dict) -> None:
    """Apply recipe overrides from the WFM InferenceRecipe onto a spec.json dict in-place.

    An inline ``prompt`` key replaces ``prompt_path``; every other key is applied
    at the top level.
    """
    for key, value in recipe_overrides.items():
        if key == "prompt":
            spec["prompt"] = value
            spec.pop("prompt_path", None)
        else:
            spec[key] = value


def _remap_hf_snapshot(
    hf_cache_dir: Path,
    repo: str,
    expected_revision: str,
    logger: "logging.Logger",
) -> None:
    """Copy ``snapshots/<actual_rev>/`` to ``snapshots/<expected_rev>/`` when they differ.

    The OCI cache may have been staged at a different commit than what the
    framework requests. Since file content is identical, we alias the snapshot
    directory. HF hub with ``HF_HUB_OFFLINE=1`` looks up files by snapshot path,
    not by blob hash, so this is sufficient.
    """
    import shutil

    model_dir = hf_cache_dir / ("models--" + repo.replace("/", "--"))
    refs_main = model_dir / "refs" / "main"
    if not refs_main.exists():
        logger.warning("_remap_hf_snapshot: refs/main not found for %s, skipping", repo)
        return

    actual_revision = refs_main.read_text().strip()
    if actual_revision == expected_revision:
        return

    actual_snapshot = model_dir / "snapshots" / actual_revision
    expected_snapshot = model_dir / "snapshots" / expected_revision

    if not actual_snapshot.exists():
        logger.warning("_remap_hf_snapshot: snapshot %s not found for %s, skipping", actual_revision[:8], repo)
        return

    if not expected_snapshot.exists():
        shutil.copytree(str(actual_snapshot), str(expected_snapshot), symlinks=True)
        logger.info("Remapped %s: %s -> %s", repo, actual_revision[:8], expected_revision[:8])


_OCI_ENDPOINT_REGION_RE = re.compile(r"objectstorage\.[a-z0-9-]+\.oraclecloud\.com")


def _endpoint_for_region(base_endpoint: str, region: str | None) -> str:
    """Rewrite the OCI S3-compat endpoint FQDN to point at ``region``.

    OCI S3-compatibility endpoints are per-region: buckets live inside a single
    region's namespace and are not visible across regions. Given the base
    endpoint (from ``AWS_ENDPOINT_URL_S3``) and a target region, swap the
    ``objectstorage.<region>.oraclecloud.com`` segment. If ``region`` is None,
    return the base endpoint unchanged.
    """
    if not region:
        return base_endpoint
    return _OCI_ENDPOINT_REGION_RE.sub(
        f"objectstorage.{region}.oraclecloud.com", base_endpoint
    )


def _setup_hf_cache(
    plain_client: "boto3.client",
    cached_client: "boto3.client",
    hf_cache_bucket: str,
    hf_cache_prefix: str,
    hf_cache_dir: Path,
    hf_snapshots: list[dict],
    logger: "logging.Logger",
) -> None:
    """Download the HF model cache from OCI; skip if the expected snapshots already exist.

    ``hf_snapshots`` is a list of ``{"repo": "...", "revision": "..."}`` entries
    naming every HF model the framework needs at runtime. All must be pre-staged
    under ``s3://{hf_cache_bucket}/{hf_cache_prefix}/`` in standard HF hub layout
    (``models--<org>--<name>/snapshots/<rev>/``).
    """
    expected_snapshots = [
        hf_cache_dir
        / ("models--" + entry["repo"].replace("/", "--"))
        / "snapshots"
        / entry["revision"]
        for entry in hf_snapshots
    ]
    if all(p.exists() for p in expected_snapshots):
        logger.info("HF cache already populated, skipping download")
        return

    hf_cache_prefix = hf_cache_prefix.rstrip("/")
    logger.info("Downloading HF model cache from s3://%s/%s", hf_cache_bucket, hf_cache_prefix)
    paginator = plain_client.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=hf_cache_bucket, Prefix=hf_cache_prefix + "/"):
        for obj in page.get("Contents", []):
            key = obj["Key"]
            relative = key[len(hf_cache_prefix):].lstrip("/")
            dest = hf_cache_dir / relative
            dest.parent.mkdir(parents=True, exist_ok=True)
            cached_client.download_file(hf_cache_bucket, key, str(dest))

    for entry in hf_snapshots:
        _remap_hf_snapshot(
            hf_cache_dir,
            repo=entry["repo"],
            expected_revision=entry["revision"],
            logger=logger,
        )


@ray.remote
def _run_batch_on_gpu(base_config: dict, jobs: list[dict]) -> None:
    """Runs on the GPU worker. Downloads shared resources once, then runs all jobs."""
    import json
    import logging
    import os
    import subprocess
    import tempfile
    from pathlib import Path

    import boto3
    import botocore.config

    try:
        from lilypad.public.sdk_py.cached_file_access.boto import (
            get_readonly_boto_client,
        )
    except ImportError:
        get_readonly_boto_client = None

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    logger = logging.getLogger(__name__)

    _oci_config = botocore.config.Config(
        s3={"payload_signing_enabled": True},
        request_checksum_calculation="when_required",
        response_checksum_validation="when_required",
    )
    _base_endpoint = os.environ["AWS_ENDPOINT_URL_S3"]
    _default_region = os.environ["AWS_DEFAULT_REGION"]

    _plain_clients: dict[str | None, "boto3.client"] = {}

    def plain_client_for(region: str | None) -> "boto3.client":
        """Return a cached boto3 S3 client whose endpoint targets ``region``.

        ``region=None`` falls back to ``AWS_DEFAULT_REGION`` / ``AWS_ENDPOINT_URL_S3``
        verbatim, preserving today's single-region behavior when nothing is set.
        """
        key = region or None
        if key not in _plain_clients:
            _plain_clients[key] = boto3.client(
                "s3",
                endpoint_url=_endpoint_for_region(_base_endpoint, region),
                region_name=region or _default_region,
                aws_access_key_id=os.environ["AWS_ACCESS_KEY_ID"],
                aws_secret_access_key=os.environ["AWS_SECRET_ACCESS_KEY"],
                config=_oci_config,
            )
        return _plain_clients[key]

    hf_cache_region = base_config.get("hf_cache_region")
    hf_plain = plain_client_for(hf_cache_region)
    if get_readonly_boto_client is not None:
        try:
            cached_client = get_readonly_boto_client(region_name=hf_cache_region) if hf_cache_region else get_readonly_boto_client()
        except TypeError:
            # Older SDK: no region_name kwarg. Fall back to argless call — HF cache
            # download works but may be routed through the default region.
            cached_client = get_readonly_boto_client()
    else:
        logger.warning(
            "lilypad SDK not installed; falling back to plain boto3 client for HF cache downloads"
        )
        cached_client = hf_plain

    hf_cache_dir = Path(os.environ.get("HF_HUB_CACHE", Path.home() / ".cache" / "huggingface" / "hub"))

    _setup_hf_cache(
        hf_plain,
        cached_client,
        base_config["hf_cache_bucket"],
        base_config["hf_cache_prefix"],
        hf_cache_dir,
        base_config["hf_snapshots"],
        logger,
    )
    os.environ["HF_HUB_OFFLINE"] = "1"

    num_gpus = base_config.get("num_gpus", 1)
    checkpoint_path = base_config.get("checkpoint_path", "Cosmos3-Nano")
    parallelism_preset = base_config.get("parallelism_preset", "latency")
    seed = base_config.get("seed", 2026)
    logger.info("Shared resources ready; running %d job(s) with %s on %d GPU(s)",
                len(jobs), checkpoint_path, num_gpus)

    for i, job in enumerate(jobs):
        input_bucket = job["input_bucket"]
        input_prefix = job["input_prefix"]
        output_bucket = job["output_bucket"]
        output_prefix = job["output_prefix"]
        spec_json = job.get("spec_json", "spec.json")
        recipe_overrides = job.get("recipe_overrides", {})
        input_client = plain_client_for(job.get("input_region"))
        output_client = plain_client_for(job.get("output_region"))

        logger.info("Job %d/%d: s3://%s/%s -> s3://%s/%s",
                    i + 1, len(jobs), input_bucket, input_prefix, output_bucket, output_prefix)

        with tempfile.TemporaryDirectory() as tmpdir:
            work = Path(tmpdir)
            assets_dir = work / "assets"
            output_dir = work / "outputs"
            output_dir.mkdir(parents=True)

            paginator = input_client.get_paginator("list_objects_v2")
            for page in paginator.paginate(Bucket=input_bucket, Prefix=input_prefix):
                for obj in page.get("Contents", []):
                    key = obj["Key"]
                    relative = key[len(input_prefix):].lstrip("/")
                    dest = assets_dir / relative
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    input_client.download_file(input_bucket, key, str(dest))

            if recipe_overrides:
                spec_path = assets_dir / spec_json
                with open(spec_path) as f:
                    spec_data = json.load(f)
                _apply_recipe_overrides(spec_data, recipe_overrides)
                with open(spec_path, "w") as f:
                    json.dump(spec_data, f, indent=2)
                logger.info("Applied recipe overrides to %s", spec_json)

            # Nano (single-GPU) uses plain python; Super needs torchrun.
            if num_gpus > 1:
                cmd = [
                    "torchrun",
                    f"--nproc-per-node={num_gpus}",
                    "--master-addr=127.0.0.1",
                    "--master-port=29500",
                    "-m", "cosmos_framework.scripts.inference",
                ]
            else:
                cmd = ["python", "-m", "cosmos_framework.scripts.inference"]

            cmd += [
                f"--parallelism-preset={parallelism_preset}",
                "-i", str(assets_dir / spec_json),
                "-o", str(output_dir),
                "--checkpoint-path", checkpoint_path,
                "--seed", str(seed),
                # Guardrail (nvidia/Cosmos-1.0-Guardrail) is not staged in OCI;
                # not needed for internal inference on controlled robotics data.
                "--no-guardrails",
            ]
            logger.info("Running: %s", " ".join(cmd))
            result = subprocess.run(cmd, capture_output=False)

            if result.returncode != 0:
                console_log = output_dir / "console.log"
                if console_log.exists():
                    debug_key = f"{output_prefix}/_debug/console.log"
                    try:
                        plain_client.upload_file(str(console_log), output_bucket, debug_key)
                        logger.info("Uploaded console.log to s3://%s/%s", output_bucket, debug_key)
                    except Exception as upload_err:
                        logger.warning("Could not upload console.log: %s", upload_err)
                raise RuntimeError(f"Job {i + 1}/{len(jobs)} inference exited with code {result.returncode}")

            logger.info("Job %d/%d inference finished successfully", i + 1, len(jobs))

            output_files = [p for p in sorted(output_dir.rglob("*")) if p.is_file()]
            logger.info("Uploading %d file(s) to s3://%s/%s", len(output_files), output_bucket, output_prefix)
            for path in output_files:
                key = f"{output_prefix}/{path.relative_to(output_dir)}".lstrip("/")
                plain_client.upload_file(str(path), output_bucket, key)

            plain_client.put_object(
                Body=b"",
                Bucket=output_bucket,
                Key=f"{output_prefix}/succeed.txt",
            )
            logger.info("Job %d/%d upload complete", i + 1, len(jobs))


def run(config: dict) -> None:
    """Lilypad entrypoint for Cosmos 3 Transfer inference.

    Accepts either a single job (flat config) or a batch (jobs list). Shared
    resources (HF model cache) are downloaded once per batch and reused across
    all jobs.

    Base config keys (shared across all jobs):
        hf_cache_bucket:    OCI bucket containing the pre-staged HF model cache
        hf_cache_prefix:    prefix under which the HF cache tree is stored
        hf_snapshots:       list of {"repo": "nvidia/Cosmos3-Nano", "revision": "..."}
                            entries naming every HF snapshot the framework loads.
                            All must be pre-staged under hf_cache_prefix.
        checkpoint_path:    HF model ID passed to --checkpoint-path
                            (default: "Cosmos3-Nano")
        parallelism_preset: --parallelism-preset value (default: "latency")
        seed:               --seed value (default: 2026)
        num_gpus:           GPUs to use; 1 -> python, >1 -> torchrun (default: 1)

    Per-job keys (under the jobs list, or at top level for a single job):
        input_bucket:      OCI bucket containing input assets
        input_prefix:      prefix under which the assets/ tree is stored
        output_bucket:     OCI bucket to upload inference outputs to
        output_prefix:     prefix under which outputs will be written
        spec_json:         spec file path relative to assets root
                           (default: spec.json)
        recipe_overrides:  optional top-level merge into spec_json before inference
    """
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    num_gpus = config.get("num_gpus", 1)

    if "jobs" in config:
        jobs = config["jobs"]
        base_config = {k: v for k, v in config.items() if k != "jobs"}
    else:
        # Single-job flat format for backward compatibility.
        jobs = [{
            "input_bucket": config["input_bucket"],
            "input_prefix": config["input_prefix"],
            "output_bucket": config["output_bucket"],
            "output_prefix": config["output_prefix"],
            "spec_json": config.get("spec_json", "spec.json"),
        }]
        base_config = config

    ray.init(address="auto")
    logger.info("Dispatching batch of %d job(s) to GPU worker (num_gpus=%d)", len(jobs), num_gpus)
    ref = _run_batch_on_gpu.options(num_gpus=num_gpus).remote(base_config, jobs)
    ray.get(ref)
    logger.info("Batch complete.")
