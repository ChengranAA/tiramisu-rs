use ndarray::{Array3, Array4, Axis};

pub fn postprocess(
    ary_out: Array4<f32>,     // (nx, ny, nz, C)
    ary_counter: Array3<f32>, // (nx, ny, nz)
) -> (Array4<f32>, Array3<f32>, Array3<f32>) {
    let (nx, ny, nz, n_channels) = ary_out.dim();

    // --- ary_mean_prob = ary_out / ary_counter[:,:,:,None] ---
    // Broadcast counter to shape (nx, ny, nz, 1)
    let ary_counter_b = ary_counter.insert_axis(Axis(3)); // (nx, ny, nz, 1)
    let ary_mean_prob = &ary_out / &ary_counter_b; // broadcasting

    // --- ary_mean_prob_norm = ary_mean_prob / sum(ary_mean_prob, axis=-1)[:, :, :, None] ---
    let sum_over_channels = ary_mean_prob.sum_axis(Axis(3)); // (nx, ny, nz)
    let sum_over_channels_b = sum_over_channels.insert_axis(Axis(3)); // (nx, ny, nz, 1)
    let ary_mean_prob_norm = &ary_mean_prob / &sum_over_channels_b;

    // --- ary_pred = argmax(ary_mean_prob, axis=-1)
    // --- ary_prob = max(ary_mean_prob, axis=-1) / sum(ary_mean_prob, axis=-1) ---

    let mut ary_pred = Array3::<f32>::zeros((nx, ny, nz)); // class indices
    let mut ary_prob = Array3::<f32>::zeros((nx, ny, nz)); // normalized prob

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

                ary_pred[(x, y, z)] = max_idx as f32;
                ary_prob[(x, y, z)] = if sum > 0.0 { max_val / sum } else { 0.0 };
            }
        }
    }

    (ary_mean_prob_norm, ary_pred, ary_prob)
}
