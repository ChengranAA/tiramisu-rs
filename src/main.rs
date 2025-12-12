use ndarray::{Array, Array3, Array4, ArrayD, ArrayView3, ArrayViewMut4, Ix3, s};
use nifti::writer::WriterOptions;
use nifti::{IntoNdArray, NiftiObject, ReaderOptions, volume};
use std::string::String;
use tensorflow::{
    self as tf, DataType, Graph, Scope, Session, SessionOptions, SessionRunArgs, Shape, Tensor,
};
mod postpro;
mod prepro;
mod utils;
use utils::squeeze;

fn func_stnd_ima(ary_ima: &Array3<f32>) -> Array3<f32> {
    // Convert to f32 (like astype(dtype))
    let data: Array3<f32> = ary_ima.map(|&v| v.into());

    // Compute mean manually (no extra crates)
    let n = data.len() as f32;
    let sum: f32 = data.iter().sum();
    let mean = if n > 0.0 { sum / n } else { 0.0 };

    // Compute standard deviation
    let var_sum: f32 = data
        .iter()
        .map(|v| {
            let diff = *v - mean;
            diff * diff
        })
        .sum();
    let variance = if n > 0.0 { var_sum / n } else { 0.0 };
    let std = variance.sqrt();

    // Standardize: (x - mean) / std
    if std > 0.0 {
        data.map(|x| (x - mean) / std)
    } else {
        eprintln!("Standard deviation was less/equal zero");
        Array::zeros(data.raw_dim())
    }
}

// const MODEL_INP_NAME: String = String::from("inp");
// const MODEL_OUT_NAME: String = String::from("out");

fn main() {
    const VAR_OUT_CHN: usize = 8;
    const VAR_INP_CHN: usize = 1;

    let tpl_strides: (usize, usize, usize) = (32, 32, 16);
    let tpl_inp_shp: (usize, usize, usize) = (64, 64, 64);

    let (px, py, pz) = tpl_inp_shp;
    let (sx, sy, sz) = tpl_strides;

    let nifti_file_path = String::from("test-data/S02_UNI_reframed.nii.gz");
    let volume: Array3<f32> = prepro::load_nifti_3d(&nifti_file_path).unwrap();

    println!(
        "Input Nifti 3D Volume file has the shape {:?}",
        volume.shape()
    );

    let ary_data_x: Array3<f32> = func_stnd_ima(&volume);
    let (nx, ny, nz) = ary_data_x.dim();

    // Expand dimension to 1, x, y, z, 1
    let ary_data_x_expanded = ary_data_x.insert_axis(ndarray::Axis(0));
    let ary_data_x_expanded = ary_data_x_expanded.insert_axis(ndarray::Axis(4));

    let ary_data_x_expanded = ary_data_x_expanded.as_standard_layout().into_owned();
    assert!(ary_data_x_expanded.is_standard_layout());

    let input_tensor = Tensor::from(ary_data_x_expanded); // Note:  here there is a dependency issue with ndarray between nifti-rs and tensorflow-rs

    // Get Patches
    let nx_patches = (nx - px) / sx + 1;
    let ny_patches = (ny - py) / sy + 1;
    let nz_patches = (nz - pz) / sz + 1;
    let num_patches = nx_patches * ny_patches * nz_patches;

    // build a graph for extract_volume_patches
    let mut scope = Scope::new_root_scope();

    let placeholder = tf::ops::Placeholder::new()
        .dtype(DataType::Float)
        .shape(Shape::from(input_tensor.shape()))
        .build(&mut scope.with_op_name("input_volume"))
        .unwrap();
    let ksizes = vec![1_i64, px as i64, py as i64, pz as i64, 1_i64];
    let strides = vec![1_i64, sx as i64, sy as i64, sz as i64, 1_i64];

    let extract_builder = tf::ops::ExtractVolumePatches::new()
        .T(DataType::Float)
        .ksizes(ksizes)
        .strides(strides)
        .padding("VALID".to_string());

    let extract_op = extract_builder.build(placeholder, &mut scope).unwrap();
    let session = Session::new(&SessionOptions::new(), &scope.graph()).unwrap();
    let mut run_args = SessionRunArgs::new();
    run_args.add_feed(
        &scope
            .graph()
            .operation_by_name_required("input_volume")
            .unwrap()
            .into(),
        0,
        &input_tensor,
    );

    let patches_token = run_args.request_fetch(&extract_op, 0);
    session.run(&mut run_args).unwrap();
    let patches_tensor: Tensor<f32> = run_args.fetch(patches_token).unwrap();
    let flat_patch_len = px * py * pz;
    assert_eq!(patches_tensor.len(), num_patches * flat_patch_len);
    //let patches_array = Array::from(patches_tensor)
    //    .to_shape((num_patches, flat_patch_len, 1))
    //    .unwrap();
    let patches_array = Array::from(patches_tensor);
    println!("{:?}", patches_array.shape());
    let patches_array = squeeze(patches_array.view()).to_owned();
    println!("{:?}", patches_array.shape());

    // We’ll keep a view-by-index function: (ix,iy,iz) -> ArrayView3<f32> (px,py,pz)
    let patches_flat = patches_array.into_raw_vec(); // back to a Vec<f32> we can index
    let patch_storage = &patches_flat;

    let patch_view = |idx: usize| -> ArrayView3<'_, f32> {
        // patch idx in [0, num_patches)
        let start = idx * flat_patch_len;
        let end = start + flat_patch_len;
        // create a view of shape (px, py, pz)
        ArrayView3::from_shape((px, py, pz), &patch_storage[start..end]).unwrap()
    };

    // prepare output array
    let mut ary_out = Array4::<f32>::zeros((nx, ny, nz, VAR_OUT_CHN));
    let mut ary_counter = Array3::<f32>::zeros((nx, ny, nz));

    // load model
    let mut model_graph = Graph::new();
    let model_dir = String::from("./model/tf_model_infer");
    let bundle = tf::SavedModelBundle::load(
        &SessionOptions::new(),
        &["serve"],
        &mut model_graph,
        model_dir,
    )
    .unwrap();

    let model_session = &bundle.session;

    let input_op = model_graph
        .operation_by_name_required("serving_default_inp")
        .unwrap();
    let output_op = model_graph
        .operation_by_name_required("StatefulPartitionedCall")
        .unwrap();

    let mut counter = 0;
    let out_len = px * py * pz * VAR_OUT_CHN;

    // debug

    for ixp in 0..nx_patches {
        let ind_x1 = ixp * sx;
        let ind_x2 = ind_x1 + px;

        for iyp in 0..ny_patches {
            let ind_y1 = iyp * sy;
            let ind_y2 = ind_y1 + py;

            for izp in 0..nz_patches {
                let ind_z1 = izp * sz;
                let ind_z2 = ind_z1 + pz;

                let patch_idx = ixp * (ny_patches * nz_patches) + iyp * nz_patches + izp;
                println!("Working on patch {} out of {}", counter, num_patches);

                let patch = patch_view(patch_idx);
                let mut patch_tensor = Tensor::<f32>::new(&[1, px as u64, py as u64, pz as u64, 1]);
                patch_tensor.copy_from_slice(patch.as_slice().unwrap());

                // Run model
                let mut run_args = SessionRunArgs::new();
                run_args.add_feed(&input_op, 0, &patch_tensor);
                let out_token = run_args.request_fetch(&output_op, 0);

                model_session.run(&mut run_args).unwrap();

                let pred: Tensor<f32> = run_args.fetch(out_token).unwrap();
                assert_eq!(pred.len(), out_len);
                let pred_array = Array::from(pred);
                let pred_view = squeeze(pred_array.view()).to_owned();

                {
                    let mut out_sub: ArrayViewMut4<f32> =
                        ary_out.slice_mut(s![ind_x1..ind_x2, ind_y1..ind_y2, ind_z1..ind_z2, ..]);

                    out_sub += &pred_view;

                    let mut cnt_sub =
                        ary_counter.slice_mut(s![ind_x1..ind_x2, ind_y1..ind_y2, ind_z1..ind_z2]);
                    cnt_sub += 1.0;
                }

                counter += 1;
            }
        }
    }

    WriterOptions::new("./output/ary_out.nii.gz")
        .write_nifti(&ary_out)
        .unwrap();

    WriterOptions::new("./output/ary_counter.nii.gz")
        .write_nifti(&ary_counter)
        .unwrap();

    let (ary_mean_prob_norm, ary_pred, ary_prob) = postpro::postprocess(ary_out, ary_counter);

    // save results
    let ary_data_x: Array3<f32> = func_stnd_ima(&volume);
    WriterOptions::new("./output/data.nii.gz")
        .write_nifti(&ary_data_x)
        .unwrap();

    WriterOptions::new("./output/pred.nii.gz")
        .write_nifti(&ary_pred)
        .unwrap();

    WriterOptions::new("./output/prob.nii.gz")
        .write_nifti(&ary_prob)
        .unwrap();

    WriterOptions::new("./output/pred_per_class.nii.gz")
        .write_nifti(&ary_mean_prob_norm)
        .unwrap();
}
