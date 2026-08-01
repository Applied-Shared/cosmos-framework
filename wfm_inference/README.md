# WFM Inference — Lilypad Entrypoint

Runs Cosmos 3 Transfer inference on Lilypad generic workloads. The applied3
workload config lives at
`adp/services/wfm/lilypad_workload_configs/cosmos_transfer3_inference.yaml`.

## Architecture

Lilypad generic workloads start a Ray cluster with two node types:

- **Head node** (`InstanceTypeVMStandardE5Flex`, CPU-only) — runs `lilypad_entrypoint.run()`.
- **GPU worker nodes** — where all the heavy work happens. Default is 1 GPU
  (Cosmos3-Nano fits on a single A100/H100); set `num_gpus > 1` to use `torchrun`
  and drive Cosmos3-Super across multiple GPUs.

`run()` calls `_run_batch_on_gpu.remote(base_config, jobs)` and blocks on
`ray.get()`. All download, inference, and upload logic lives inside the
`@ray.remote` function so it executes on a GPU worker, not the CPU head.

## Usage

Export OCI credentials, then launch:

```bash
export AWS_ACCESS_KEY_ID=<oci-access-key>
export AWS_SECRET_ACCESS_KEY=<oci-secret-key>
lilypad workload launch adp/services/wfm/lilypad_workload_configs/cosmos_transfer3_inference.yaml
```

The workload config contains a base `entrypoint_fn_config` that the WFM gRPC
service will override per-job at submission time. For manual test runs the
defaults point at `sensor-sim-wfm/robotics/episodes/example_episode`.

## Config Keys

Shared fields (top-level, same for every job in the batch):

| Key | Description |
|-----|-------------|
| `hf_cache_bucket` / `hf_cache_prefix` | OCI prefix holding the pre-staged HuggingFace model cache |
| `hf_cache_region` | Optional OCI region for `hf_cache_bucket`; swaps the boto3 endpoint when the cache lives outside `AWS_ENDPOINT_URL_S3` |
| `hf_snapshots` | List of `{repo, revision}` naming every HF snapshot the framework loads. Must be present under `hf_cache_prefix`. |
| `checkpoint_path` | HF alias passed to `--checkpoint-path` (default: `Cosmos3-Nano`). Resolved by the framework registry at `cosmos_framework/inference/args.py:1144`. |
| `parallelism_preset` | `--parallelism-preset` value (default: `latency`) |
| `seed` | `--seed` value (default: `2026`) |
| `num_gpus` | GPUs to claim on the worker. `1` → plain `python`; `>1` → `torchrun --nproc-per-node=N` |

Per-job fields (under `jobs` list, or at top level for a single job):

| Key | Description |
|-----|-------------|
| `control_bucket` / `control_prefix` | OCI location of the control bundle (`spec.json`, control videos, `prompt.json`, `negative_prompt.json`) |
| `control_region` | Optional OCI region for `control_bucket`; swaps the boto3 endpoint for input LIST/GET |
| `output_bucket` / `output_prefix` | OCI destination for the generated MP4 |
| `output_region` | Optional OCI region for `output_bucket`; swaps the boto3 endpoint for output PUT |
| `spec_json` | Spec file path relative to the assets root (default: `spec.json`) |
| `recipe_overrides` | Optional top-level dict merge applied to `spec_json` before inference. An inline `prompt` key replaces `prompt_path`. |

### Single-job example (Cosmos3-Nano, edge control)

```yaml
entrypoint_fn_config:
  hf_cache_bucket: sensor-sim-wfm
  hf_cache_prefix: checkpoints/hf-cache-cosmos3
  hf_snapshots:
    - repo: nvidia/Cosmos3-Nano
      revision: main
  checkpoint_path: Cosmos3-Nano
  parallelism_preset: latency
  seed: 2026
  num_gpus: 1
  control_bucket: sensor-sim-wfm
  control_prefix: robotics/episodes/pickup_drill_v3
  output_bucket: sensor-sim-wfm
  output_prefix: robotics/inferences/pickup_drill_v3
  spec_json: spec.json
  recipe_overrides:
    resolution: "720"
    aspect_ratio: "16,9"
    num_frames: 121
    fps: 30
    guidance: 3.0
    control_guidance: 1.5
```

The HF cache is downloaded once per batch; each job then downloads its own
assets, runs inference, and uploads outputs before the next job starts.

### Spec.json shape

Cosmos 3 spec files follow the transfer cookbook at
[`cookbooks/cosmos3/generator/transfer/`](../cookbooks/cosmos3/generator/transfer)
in this repo. Minimal edge-control spec:

```json
{
  "name": "transfer_edge",
  "model_mode": "video2video",
  "resolution": "720",
  "aspect_ratio": "16,9",
  "num_frames": 121,
  "fps": 30,
  "guidance": 3.0,
  "control_guidance": 1.5,
  "negative_prompt_file": "negative_prompt.json",
  "prompt_path": "prompt.json",
  "edge": {
    "vision_path": "source.mp4",
    "preset_edge_threshold": "medium"
  }
}
```

`vision_path` derives the Canny edge control on the fly. Use `control_path`
instead when passing a pre-computed control video (required for `depth` and
`seg` — those depend on DepthAnything and SAM2 which are not bundled).

## Building and Pushing the Docker Image

All inference code is baked into the image at build time (`STANDALONE=true`).

```bash
cd <path to this repo checkout>
docker build -f Dockerfile \
  --build-arg CUDA_NAME=cu128 \
  --build-arg STANDALONE=true \
  -t us-phoenix-1.ocir.io/idskhu5vqvtl/lilypad/sds:cosmos_transfer3_v<VERSION> .

docker push us-phoenix-1.ocir.io/idskhu5vqvtl/lilypad/sds:cosmos_transfer3_v<VERSION>
```

Log in first if you haven't already:

```bash
docker login us-phoenix-1.ocir.io -u idskhu5vqvtl/<user>@applied.co
```

Paste an OCI auth token (not password) when prompted.

After push, bump `docker_image` in
`adp/services/wfm/lilypad_workload_configs/cosmos_transfer3_inference.yaml`.
Most layers are shared with previous tags, so rebuilds are fast when only
Python files changed.

## OCI S3-compat Gotcha

OCI's S3-compatible API requires payload signing and does not accept the
default AWS SDK v4 checksum headers. Any boto3 client used for PUT/LIST
against OCI must use:

```python
botocore.config.Config(
    s3={"payload_signing_enabled": True},
    request_checksum_calculation="when_required",
    response_checksum_validation="when_required",
)
```

There are two client factories in the worker:

- **`plain_client_for(region)`** — returns a cached direct OCI boto3 client (above
  config) whose endpoint targets `region`. Used for control-bundle LIST/GET,
  output PUT, and HF-cache LIST. When `region` is omitted, the client uses
  `AWS_ENDPOINT_URL_S3` / `AWS_DEFAULT_REGION`.
- **`cached_client`** (`get_readonly_boto_client()`) — routes GETs through the
  AIStore cross-region cache at the Chicago edge. Used for HF cache downloads
  to avoid repeated cross-region transfer costs.

## Pre-staging the HuggingFace Model Cache

Cosmos 3 resolves `--checkpoint-path Cosmos3-Nano` to HF repo
[`nvidia/Cosmos3-Nano`](https://huggingface.co/nvidia/Cosmos3-Nano) via the
registry at `cosmos_framework/inference/args.py:1144`. That repo ships its own
processor/tokenizer, so no upstream `Qwen/Qwen3-VL-*-Instruct` side-download is
needed — only **one** HF repo has to be pre-staged.

Pre-stage the cache in OCI under `sensor-sim-wfm/checkpoints/hf-cache-cosmos3/`
as a standard HF hub cache tree:

```
models--nvidia--Cosmos3-Nano/
  refs/main
  snapshots/<rev>/<all model files>
  blobs/...
```

Upload with `aws s3 sync` pointed at `~/.cache/huggingface/hub/` after
downloading the model locally. OCI upload needs the same checksum overrides:

```bash
export AWS_REQUEST_CHECKSUM_CALCULATION=when_required
export AWS_RESPONSE_CHECKSUM_VALIDATION=when_required
aws s3 sync ~/.cache/huggingface/hub/ \
  s3://sensor-sim-wfm/checkpoints/hf-cache-cosmos3/ \
  --endpoint-url https://idskhu5vqvtl.compat.objectstorage.us-phoenix-1.oraclecloud.com
```

Note the `hf-cache-cosmos3` prefix — kept separate from the 2.5 `hf-cache/`
prefix so the two model families don't collide.

### Revision handling

The workload YAML's `hf_snapshots` list names the revision each repo is
expected to load. Cosmos 3's built-in registry uses `revision: main` (a
floating ref) for `Cosmos3-Nano`, so `_remap_hf_snapshot()` aliases whatever
commit was staged to the `main`-resolved revision on the worker. If the
framework's registry is later pinned to a specific commit, update the
`revision` field in the workload YAML to match and re-stage the OCI cache if
the model weights actually changed.

## Content Safety Guardrails

Cosmos 3 enables guardrails by default, which would download
[`nvidia/Cosmos-1.0-Guardrail`](https://huggingface.co/nvidia/Cosmos-1.0-Guardrail)
(gated) from HuggingFace at startup. That model is not staged in OCI and is
not needed for internal inference on controlled robotics data.

Guardrails are disabled by passing `--no-guardrails` to
`cosmos_framework.scripts.inference`. If guardrail checks are ever needed,
request HF access to `nvidia/Cosmos-1.0-Guardrail`, stage it under
`hf-cache-cosmos3/`, add an entry to the workload YAML's `hf_snapshots`, and
remove `--no-guardrails` from the entrypoint.
