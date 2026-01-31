use ndarray::{Array, Array3, Array4, ArrayD, ArrayViewD, Axis, Ix3};
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

    for x in 0..nx {
        for y in 0..ny {
            for z in 0..nz {
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

                ary_pred[(x, y, z)] = max_idx as i32;
                ary_prob[(x, y, z)] = if sum > 0.0 { max_val / sum } else { 0.0 };
            }
        }
    }

    (ary_mean_prob_norm, ary_pred, ary_prob)
}
