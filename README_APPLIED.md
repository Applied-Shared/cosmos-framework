# README (Applied additions)

This is Applied's fork of [`NVIDIA/cosmos-framework`](https://github.com/NVIDIA/cosmos-framework),
extended with the pieces needed to run Cosmos 3 Transfer as a Lilypad workload
on WFM. See [`wfm_inference/README.md`](wfm_inference/README.md) for the full
architecture, config schema, HF pre-staging, and image build docs.

## Quick reference — build & push the Lilypad image

```bash
docker build -f Dockerfile -q \
  --build-arg CUDA_NAME=cu128 \
  --build-arg STANDALONE=true \
  -t us-phoenix-1.ocir.io/idskhu5vqvtl/lilypad/sds:cosmos_transfer3_v0.0.1 .

docker push us-phoenix-1.ocir.io/idskhu5vqvtl/lilypad/sds:cosmos_transfer3_v0.0.1
```

Then bump `docker_image` in
[`cosmos_transfer3_inference.yaml`](https://github.com/AppliedIntuition/applied3/blob/master/adp/services/wfm/lilypad_workload_configs/cosmos_transfer3_inference.yaml)
in the applied3 repo.
