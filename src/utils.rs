use ndarray::{Array, Array3, ArrayViewD, Axis};

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
    let data: Array3<f32> = ary_ima.map(|&v| v.into());

    // let n = data.len() as f32;
    let mean = data.mean().unwrap();
    let std = data.std(0.0);

    // Compute standard deviation
    //let var_sum: f32 = data
    //    .iter()
    //    .map(|v| {
    //        let diff = *v - mean;
    //        diff * diff
    //    })
    //    .sum();
    //let variance = if n > 0.0 { var_sum / n } else { 0.0 };
    //let std = variance.sqrt();

    // Standardize: (x - mean) / std
    if std > 0.0 {
        data.map(|x| (x - mean) / std)
    } else {
        eprintln!("Standard deviation was less/equal zero");
        Array::zeros(data.raw_dim())
    }
}
