# Tiramisu-rs 

A Rust implementation of Tiramisu Model for segmentation of MPRAGE and MP2RAGE anatomical MRI Volumes

## Status
- [x] MP2RAGE Slow mode segmentation
- [x] MP2RAGE Fast mode segmentation
- [x] MPRAGE Slow mode segmentation
- [x] MPRAGE Fast mode segmentation

ONNX Runtime is the default inference backend. The legacy TensorFlow backend is
deprecated and will be removed in a future release; it remains available with
`--use-tf` during the transition.

CPU inference is the default. `--gpu` runs the complete Conv3D graph through
Burn's native WGPU backend. Convolutions use FP16 while the source model's
training-mode BatchNormalization is represented as FP32 per-patch
InstanceNormalization to avoid reduced-precision overflow. On representative
central patches this experimental path retained 99.924% MP2RAGE and 99.735%
MPRAGE voxel-label agreement with the faithful ONNX model.

## Install


## Develop
### Requirement:
1. Rust tool chain: Cargo, Rustc ...
2. libtensorflow:
    * On Mac: `brew install libtensorflow`

### Build
```{bash}
cargo build
```

### Run

```bash
cargo run --release -- --input volume.nii.gz --type mp2rage --mode slow
```

Use native GPU inference:

```bash
cargo run --release -- --input volume.nii.gz --type mp2rage --mode slow --gpu
```

Use `--reframe` to symmetrically zero-pad the standardized volume to a complete
patch grid and crop the predictions back to the original dimensions. Reframing
is disabled by default.

To temporarily use the deprecated TensorFlow backend:

```bash
cargo run --release -- --input volume.nii.gz --type mp2rage --mode slow --use-tf
```
