use ndarray::{Array, Array3, Array4, ArrayD, ArrayViewD, Axis, Ix3, Zip, s};
use nifti::{IntoNdArray, NiftiObject, ReaderOptions};

pub fn load_nifti_3d(path: &str) -> Result<Array3<f32>, Box<dyn std::error::Error>> {
    let obj = ReaderOptions::new().read_file(path).unwrap();
    let _header = obj.header().clone();
    let volume: ArrayD<f32> = obj.into_volume().into_ndarray::<f32>()?;
    let volume: Array3<f32> = volume.into_dimensionality::<Ix3>()?;
    Ok(volume)
}

pub fn squeeze<A>(view: ArrayViewD<'_, A>) -> ArrayViewD<'_, A> {
    let mut v = view;
    let mut axis = 0;

    while axis < v.ndim() {
        if v.shape()[axis] == 1 {
            v = v.remove_axis(Axis(axis));
        } else {
            axis += 1;
        }
    }
    v
}

pub fn func_stnd_ima(ary_ima: &Array3<f32>) -> Array3<f32> {
    // Convert to f32 (like astype(dtype))
    // let data: Array3<f32> = ary_ima.map(|&v| v.into());
    let data: Array3<f32> = ary_ima.clone();
    let mean = data.mean().unwrap();
    let std = data.std(0.0);

    // Standardize: (x - mean) / std
    if std > 0.0 {
        data.map(|x| (x - mean) / std)
    } else {
        eprintln!("Standard deviation was less/equal zero");
        Array::zeros(data.raw_dim())
    }
}

pub fn reframe_volume(
    volume: &Array3<f32>,
    patch: (usize, usize, usize),
    stride: (usize, usize, usize),
) -> (Array3<f32>, (usize, usize, usize)) {
    fn framed_size(size: usize, patch: usize, stride: usize) -> usize {
        if size <= patch {
            patch
        } else {
            patch + (size - patch).div_ceil(stride) * stride
        }
    }

    let (nx, ny, nz) = volume.dim();
    let (px, py, pz) = patch;
    let (sx, sy, sz) = stride;
    let framed = (
        framed_size(nx, px, sx),
        framed_size(ny, py, sy),
        framed_size(nz, pz, sz),
    );
    let offset = (
        (framed.0 - nx) / 2,
        (framed.1 - ny) / 2,
        (framed.2 - nz) / 2,
    );

    let mut reframed = Array3::<f32>::zeros(framed);
    reframed
        .slice_mut(s![
            offset.0..offset.0 + nx,
            offset.1..offset.1 + ny,
            offset.2..offset.2 + nz
        ])
        .assign(volume);

    (reframed, offset)
}

pub fn postprocess(
    ary_out: Array4<f32>,     // (nx, ny, nz, C)
    ary_counter: Array3<f32>, // (nx, ny, nz)
) -> (Array4<f32>, Array3<i32>, Array3<f32>) {
    let (nx, ny, nz, n_channels) = ary_out.dim();

    // --- ary_mean_prob = ary_out / ary_counter[:,:,:,None] ---
    // Broadcast counter to shape (nx, ny, nz, 1)
    let ary_counter_b = ary_counter.insert_axis(Axis(3)); // (nx, ny, nz, 1)
    let ary_mean_prob = &ary_out / &ary_counter_b;

    // --- ary_mean_prob_norm = ary_mean_prob / sum(ary_mean_prob, axis=-1)[:, :, :, None] ---
    let sum_over_channels = ary_mean_prob.sum_axis(Axis(3)); // (nx, ny, nz)
    let sum_over_channels_b = sum_over_channels.insert_axis(Axis(3)); // (nx, ny, nz, 1)
    let ary_mean_prob_norm = &ary_mean_prob / &sum_over_channels_b;

    // --- ary_pred = argmax(ary_mean_prob, axis=-1)
    // --- ary_prob = max(ary_mean_prob, axis=-1) / sum(ary_mean_prob, axis=-1) ---
    let mut ary_pred = Array3::<i32>::zeros((nx, ny, nz));
    let mut ary_prob = Array3::<f32>::zeros((nx, ny, nz));

    Zip::indexed(&mut ary_pred)
        .and(&mut ary_prob)
        .par_for_each(|(x, y, z), pred, prob| {
            let mut max_val = f32::MIN;
            let mut max_idx = 0usize;
            let mut sum = 0.0f32;

            for c in 0..n_channels {
                let v = ary_mean_prob[(x, y, z, c)];
                sum += v;
                if v > max_val {
                    max_val = v;
                    max_idx = c;
                }
            }

            *pred = max_idx as i32;
            *prob = if sum > 0.0 { max_val / sum } else { 0.0 };
        });

    (ary_mean_prob_norm, ary_pred, ary_prob)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reframes_and_centers_volume_on_patch_grid() {
        let volume = Array3::from_elem((70, 80, 90), 1.0);
        let (reframed, offset) = reframe_volume(&volume, (64, 64, 64), (32, 32, 32));

        assert_eq!(reframed.dim(), (96, 96, 96));
        assert_eq!(offset, (13, 8, 3));
        assert_eq!(reframed[(0, 0, 0)], 0.0);
        assert!(
            reframed
                .slice(s![13..83, 8..88, 3..93])
                .iter()
                .all(|&voxel| voxel == 1.0)
        );
    }
}
