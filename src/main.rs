use burn::{
    backend::{Wgpu, wgpu::WgpuDevice},
    tensor::{Tensor as BurnTensor, TensorData},
};
use indicatif::{ProgressBar, ProgressStyle};
use ndarray::{Array, Array3, Array4, ArrayD, ArrayViewMut4, s};
use nifti::writer::WriterOptions;
use ort::{session::Session as OnnxSession, value::Tensor as OnnxTensor};
use std::{
    f32,
    fs::{DirBuilder, File},
    path::Path,
    string::String,
};
use tensorflow::{Graph, Operation, SavedModelBundle, SessionOptions, SessionRunArgs, Tensor};
mod models;
mod utils;
use clap::{Arg, ArgMatches, Command};
use utils::{func_stnd_ima, load_nifti_3d, postprocess, reframe_volume, squeeze};

// const MODEL_INP_NAME: String = String::from("inp");
// const MODEL_OUT_NAME: String = String::from("out");

const MP2RAGE_MODEL: &str = "./model/tf_legacy/mp2rage";
const MPRAGE_MODEL: &str = "./model/tf_legacy/mprage";
const MP2RAGE_ONNX_MODEL: &str = "./model/onnx_cpu/mp2rage.onnx";
const MPRAGE_ONNX_MODEL: &str = "./model/onnx_cpu/mprage.onnx";
const VAR_OUT_CHN: usize = 8;
const TPL_INP_SHP: (usize, usize, usize) = (64, 64, 64);
const INPUT_OP_NAME: &str = "serving_default_inp";
const OUTPUT_OP_NAME: &str = "StatefulPartitionedCall";
const MP2RAGE_WGPU_WEIGHTS: &str = "./model/wgpu/mp2rage.bpk";
const MPRAGE_WGPU_WEIGHTS: &str = "./model/wgpu/mprage.bpk";

type WgpuBackend = Wgpu<burn::tensor::f16>;

enum InferenceBackend {
    Onnx(OnnxSession),
    Mp2rageWgpu {
        model: models::mp2rage::Model<WgpuBackend>,
        device: WgpuDevice,
    },
    MprageWgpu {
        model: models::mprage::Model<WgpuBackend>,
        device: WgpuDevice,
    },
    TensorFlow {
        _graph: Graph,
        bundle: SavedModelBundle,
        input: Operation,
        output: Operation,
    },
}

impl InferenceBackend {
    fn load(use_tf: bool, use_gpu: bool, run_type: &str, tf_model: &str, onnx_model: &str) -> Self {
        if use_tf {
            let mut graph = Graph::new();
            let bundle =
                SavedModelBundle::load(&SessionOptions::new(), ["serve"], &mut graph, tf_model)
                    .unwrap();
            let input = graph.operation_by_name_required(INPUT_OP_NAME).unwrap();
            let output = graph.operation_by_name_required(OUTPUT_OP_NAME).unwrap();
            Self::TensorFlow {
                _graph: graph,
                bundle,
                input,
                output,
            }
        } else if use_gpu {
            let device = WgpuDevice::default();
            println!("GPU mode: using native Burn/WGPU Conv3D execution");
            match run_type {
                "mp2rage" => Self::Mp2rageWgpu {
                    model: models::mp2rage::Model::from_file(MP2RAGE_WGPU_WEIGHTS, &device),
                    device,
                },
                "mprage" => Self::MprageWgpu {
                    model: models::mprage::Model::from_file(MPRAGE_WGPU_WEIGHTS, &device),
                    device,
                },
                _ => unreachable!(),
            }
        } else {
            let mut builder = OnnxSession::builder().unwrap();
            Self::Onnx(builder.commit_from_file(onnx_model).unwrap())
        }
    }

    fn run(&mut self, patch: &Array3<f32>) -> ArrayD<f32> {
        let (px, py, pz) = patch.dim();
        match self {
            Self::Onnx(session) => {
                let input = OnnxTensor::from_array((
                    [1, px, py, pz, 1],
                    patch.iter().copied().collect::<Vec<_>>(),
                ))
                .unwrap();
                let outputs = session.run(ort::inputs!["inp" => input]).unwrap();
                let (_, data) = outputs["out"].try_extract_tensor::<f32>().unwrap();
                Array::from_shape_vec(vec![1, px, py, pz, VAR_OUT_CHN], data.to_vec()).unwrap()
            }
            Self::TensorFlow {
                bundle,
                input,
                output,
                ..
            } => {
                let mut input_tensor = Tensor::<f32>::new(&[1, px as u64, py as u64, pz as u64, 1]);
                input_tensor.copy_from_slice(patch.as_slice().unwrap());
                let mut args = SessionRunArgs::new();
                args.add_feed(input, 0, &input_tensor);
                let token = args.request_fetch(output, 0);
                bundle.session.run(&mut args).unwrap();
                Array::from(args.fetch::<f32>(token).unwrap())
            }
            Self::Mp2rageWgpu { model, device } => run_wgpu(model, device, patch),
            Self::MprageWgpu { model, device } => run_wgpu(model, device, patch),
        }
    }
}

fn run_wgpu<M>(model: &M, device: &WgpuDevice, patch: &Array3<f32>) -> ArrayD<f32>
where
    M: WgpuModel,
{
    let input = BurnTensor::<WgpuBackend, 5>::from_data(
        TensorData::new(
            patch
                .iter()
                .map(|&value| burn::tensor::f16::from_f32(value))
                .collect::<Vec<_>>(),
            [1, 64, 64, 64, 1],
        ),
        device,
    );
    let values = model
        .forward(input)
        .into_data()
        .to_vec::<burn::tensor::f16>()
        .unwrap()
        .into_iter()
        .map(f32::from)
        .collect();
    Array::from_shape_vec(vec![1, 64, 64, 64, VAR_OUT_CHN], values).unwrap()
}

trait WgpuModel {
    fn forward(&self, input: BurnTensor<WgpuBackend, 5>) -> BurnTensor<WgpuBackend, 5>;
}

impl WgpuModel for models::mp2rage::Model<WgpuBackend> {
    fn forward(&self, input: BurnTensor<WgpuBackend, 5>) -> BurnTensor<WgpuBackend, 5> {
        self.forward(input)
    }
}

impl WgpuModel for models::mprage::Model<WgpuBackend> {
    fn forward(&self, input: BurnTensor<WgpuBackend, 5>) -> BurnTensor<WgpuBackend, 5> {
        self.forward(input)
    }
}

// BUG: REGION MISSES IN FAST MODE

fn interface() -> ArgMatches {
    Command::new("tiramisu")
        .about("An Implementation of the Tiramisu Anatomical Segmentation Inference Utility")
        .version("0.0.1")
        .arg(
            Arg::new("input")
                .long("input")
                .short('i')
                .value_name("INPUT_PATH")
                .help("Sets input nifti file")
                .required(true)
                .value_parser(clap::value_parser!(String)),
        )
        .arg(
            Arg::new("mode")
                .long("mode")
                .short('m')
                .default_value("slow")
                .value_name("RUN_MODE")
                .help("Sets model running mode")
                .value_parser(clap::value_parser!(String)),
        )
        .arg(
            Arg::new("type")
                .long("type")
                .short('t')
                .default_value("mp2rage")
                .value_name("FILE_TYPE")
                .help("Sets model running type")
                .value_parser(clap::value_parser!(String)),
        )
        .arg(
            Arg::new("use-tf")
                .long("use-tf")
                .help("Use the deprecated TensorFlow inference backend")
                .conflicts_with("gpu")
                .action(clap::ArgAction::SetTrue),
        )
        .arg(
            Arg::new("gpu")
                .long("gpu")
                .help("Use native Burn/WGPU inference")
                .conflicts_with("use-tf")
                .action(clap::ArgAction::SetTrue),
        )
        .arg(
            Arg::new("reframe")
                .long("reframe")
                .help("Symmetrically pad the volume to a complete patch grid")
                .action(clap::ArgAction::SetTrue),
        )
        .get_matches()
}

fn main() {
    // CONFIG

    let arg_matches: ArgMatches = interface();
    let run_mode: &str = arg_matches.get_one::<String>("mode").expect("").as_str();
    let run_type: &str = arg_matches.get_one::<String>("type").expect("").as_str();
    let use_tf = arg_matches.get_flag("use-tf");
    let use_gpu = arg_matches.get_flag("gpu");
    let use_reframe = arg_matches.get_flag("reframe");

    let (tf_model, onnx_model): (&str, &str) = match run_type {
        "mp2rage" => (MP2RAGE_MODEL, MP2RAGE_ONNX_MODEL),
        "mprage" => (MPRAGE_MODEL, MPRAGE_ONNX_MODEL),
        other => {
            eprintln!("Error: '{}' is not a valid run type", other);
            std::process::exit(1);
        }
    };

    if use_tf {
        eprintln!(
            "DEPRECATION WARNING: the TensorFlow backend is deprecated and will be removed in a future release; omit --use-tf to use ONNX Runtime."
        );
    }
    if use_gpu {
        eprintln!("GPU mode: experimental FP16 inference with FP32 per-patch InstanceNorm.");
    }
    let nifti_path_str: &str = arg_matches
        .get_one::<String>("input")
        .expect("input file path is required")
        .as_str();

    let input_nifti_path = Path::new(&nifti_path_str).to_owned();

    let tpl_strides: (usize, usize, usize) = match run_mode {
        "fast" => (64, 64, 64),
        "slow" => (32, 32, 32),
        other => {
            eprintln!("Error: '{}' is not a valid run mode", other);
            std::process::exit(1);
        }
    };

    let (sx, sy, sz) = tpl_strides;
    let (px, py, pz) = TPL_INP_SHP;

    let volume: Array3<f32> = load_nifti_3d(input_nifti_path.to_str().unwrap()).unwrap();
    let (nx, ny, nz) = volume.dim();

    // CONFIG

    println!(
        "Input Nifti 3D Volume file has the shape {:?}",
        volume.shape()
    );

    let ary_data_x: Array3<f32> = func_stnd_ima(&volume);

    let (reframed, frame_offset) = if use_reframe {
        let (reframed, offset) = reframe_volume(&ary_data_x, TPL_INP_SHP, tpl_strides);
        println!(
            "Reframed standardized volume from {:?} to {:?}",
            ary_data_x.shape(),
            reframed.shape()
        );
        (Some(reframed), offset)
    } else {
        (None, (0, 0, 0))
    };
    let inference_volume = reframed.as_ref().unwrap_or(&ary_data_x);
    let (fx, fy, fz) = inference_volume.dim();

    let nx_patches = (fx - px) / sx + 1;
    let ny_patches = (fy - py) / sy + 1;
    let nz_patches = (fz - pz) / sz + 1;
    let num_patches = nx_patches * ny_patches * nz_patches;

    // prepare output array
    let mut ary_out = Array4::<f32>::zeros((fx, fy, fz, VAR_OUT_CHN));
    let mut ary_counter = Array3::<f32>::zeros((fx, fy, fz));

    let mut backend = InferenceBackend::load(use_tf, use_gpu, run_type, tf_model, onnx_model);

    let out_len = px * py * pz * VAR_OUT_CHN;

    let mut patch_positions = Vec::with_capacity(num_patches);
    for ixp in 0..nx_patches {
        for iyp in 0..ny_patches {
            for izp in 0..nz_patches {
                patch_positions.push((ixp * sx, iyp * sy, izp * sz));
            }
        }
    }

    let progress = ProgressBar::new(num_patches as u64)
        .with_style(
            ProgressStyle::with_template(
                "{msg} {spinner:.cyan} [{elapsed_precise}] [{bar:40.cyan/blue}] {pos}/{len} ({percent}%) {per_sec} ETA {eta_precise}",
            )
            .unwrap()
            .progress_chars("━╸─"),
        )
        .with_message("Segmenting");

    for &(ind_x1, ind_y1, ind_z1) in &patch_positions {
        let patch = inference_volume
            .slice(s![
                ind_x1..ind_x1 + px,
                ind_y1..ind_y1 + py,
                ind_z1..ind_z1 + pz
            ])
            .as_standard_layout()
            .into_owned();
        let pred_array = backend.run(&patch);

        assert_eq!(pred_array.len(), out_len);
        let pred_array = squeeze(pred_array.view()).to_owned();
        let ind_x2 = ind_x1 + px;
        let ind_y2 = ind_y1 + py;
        let ind_z2 = ind_z1 + pz;
        {
            let mut out_sub: ArrayViewMut4<f32> =
                ary_out.slice_mut(s![ind_x1..ind_x2, ind_y1..ind_y2, ind_z1..ind_z2, ..]);
            out_sub += &pred_array;

            let mut cnt_sub =
                ary_counter.slice_mut(s![ind_x1..ind_x2, ind_y1..ind_y2, ind_z1..ind_z2]);
            cnt_sub += 1.0;
        }
        progress.inc(1);
    }
    progress.finish_with_message("Segmentation complete");

    let lgc_zeros = ary_counter.mapv(|x| if x.abs() < f32::EPSILON { 1.0 } else { 0.0 });
    if lgc_zeros.sum() > 0.0 {
        println!(
            "!!!WARNING: Voxels were unvisitied!!! ', 'Change tpl_strides input (see WARNING messages above)"
        )
    }

    let (ary_out, ary_counter) = if use_reframe {
        let (ox, oy, oz) = frame_offset;
        (
            ary_out
                .slice(s![ox..ox + nx, oy..oy + ny, oz..oz + nz, ..])
                .to_owned(),
            ary_counter
                .slice(s![ox..ox + nx, oy..oy + ny, oz..oz + nz])
                .to_owned(),
        )
    } else {
        (ary_out, ary_counter)
    };

    let (ary_mean_prob_norm, ary_pred, ary_prob) = postprocess(ary_out, ary_counter);

    // save results

    // Create a directory if not exit else skip
    match DirBuilder::new().create("./output") {
        Ok(()) => println!("LOG: output folder does not exist, create the folder now."),
        Err(_e) => println!("LOG: output folder already exist, skip creation. "),
    }

    File::create("./output/data.nii.gz").unwrap();
    WriterOptions::new("./output/data.nii.gz")
        .write_nifti(&ary_data_x)
        .unwrap();

    File::create("./output/pred.nii.gz").unwrap();
    WriterOptions::new("./output/pred.nii.gz")
        .write_nifti(&ary_pred)
        .unwrap();

    File::create("./output/prob.nii.gz").unwrap();
    WriterOptions::new("./output/prob.nii.gz")
        .write_nifti(&ary_prob)
        .unwrap();

    File::create("./output/pred_per_class.nii.gz").unwrap();
    WriterOptions::new("./output/pred_per_class.nii.gz")
        .write_nifti(&ary_mean_prob_norm)
        .unwrap();
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[ignore = "slow WGPU integration test"]
    fn gpu_matches_onnx_labels_on_real_patches() {
        for (run_type, tf_model, onnx_model, test_volume) in [
            (
                "mp2rage",
                MP2RAGE_MODEL,
                MP2RAGE_ONNX_MODEL,
                "./test-data/mp2rage.nii",
            ),
            (
                "mprage",
                MPRAGE_MODEL,
                MPRAGE_ONNX_MODEL,
                "./test-data/mprage.nii",
            ),
        ] {
            let volume = func_stnd_ima(&load_nifti_3d(test_volume).unwrap());
            let (nx, ny, nz) = volume.dim();
            let start = ((nx - 64) / 2, (ny - 64) / 2, (nz - 64) / 2);
            let patch = volume
                .slice(s![
                    start.0..start.0 + 64,
                    start.1..start.1 + 64,
                    start.2..start.2 + 64
                ])
                .as_standard_layout()
                .into_owned();

            let mut backend = InferenceBackend::load(false, true, run_type, tf_model, onnx_model);
            let output = backend.run(&patch);
            let mut reference = OnnxSession::builder()
                .unwrap()
                .commit_from_file(onnx_model)
                .unwrap();

            let input = OnnxTensor::from_array((
                [1, 64, 64, 64, 1],
                patch.iter().copied().collect::<Vec<_>>(),
            ))
            .unwrap();
            let expected_output = reference.run(ort::inputs!["inp" => input]).unwrap();
            let (_, expected) = expected_output["out"].try_extract_tensor::<f32>().unwrap();

            let mut max_difference = 0.0_f32;
            let mut matching_labels = 0usize;
            for (&actual, &expected) in output.iter().zip(expected.iter()) {
                max_difference = max_difference.max((actual - expected).abs());
            }
            for (actual, expected) in output
                .as_slice()
                .unwrap()
                .chunks_exact(VAR_OUT_CHN)
                .zip(expected.chunks_exact(VAR_OUT_CHN))
            {
                let argmax = |values: &[f32]| {
                    values
                        .iter()
                        .enumerate()
                        .max_by(|(_, a), (_, b)| a.total_cmp(b))
                        .unwrap()
                        .0
                };
                matching_labels += usize::from(argmax(actual) == argmax(expected));
            }
            let label_agreement = matching_labels as f32 / (64 * 64 * 64) as f32;
            assert!(output.iter().all(|value| value.is_finite()));
            assert!(
                label_agreement > 0.99,
                "{run_type}: max difference {max_difference}, label agreement {label_agreement:.4}"
            );
            println!(
                "{run_type}: max difference {max_difference:.6}, label agreement {:.3}%",
                label_agreement * 100.0
            );
        }
    }
}
