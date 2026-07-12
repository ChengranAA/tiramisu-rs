# Tiramisu-rs 

A Rust implementation of Tiramisu Model for segmentation of MPRAGE and MP2RAGE anatomical MRI Volumes

## Status
- [x] MP2RAGE Slow mode segmentation 
- [x] MP2RAGE Fast mode segmentation
- [x] MPRAGE Slow model segmentation 
- [x] MPRAGE Fast mode segmentation 

ONNX Runtime is the default inference backend. The legacy TensorFlow backend is
deprecated and will be removed in a future release; it remains available with
`--use-tf` during the transition.

On Apple platforms, the ONNX backend detects CoreML support automatically and
uses the GPU when available, with CPU fallback for unsupported operations.

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

Use `--reframe` to symmetrically zero-pad the standardized volume to a complete
patch grid and crop the predictions back to the original dimensions. Reframing
is disabled by default.

To temporarily use the deprecated TensorFlow backend:

```bash
cargo run --release -- --input volume.nii.gz --type mp2rage --mode slow --use-tf
```
