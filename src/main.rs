use ndarray::{Array, Array3, Array4, ArrayD, ArrayViewMut4, s};
use nifti::writer::WriterOptions;
use ort::{
    ep::{self, ExecutionProvider},
    session::Session as OnnxSession,
    value::Tensor as OnnxTensor,
};
use std::{
    f32,
    fs::{DirBuilder, File},
    path::Path,
    string::String,
    time::Instant,
};
use tensorflow::{Graph, Operation, SavedModelBundle, SessionOptions, SessionRunArgs, Tensor};
mod utils;
use clap::{Arg, ArgMatches, Command};
use utils::{func_stnd_ima, load_nifti_3d, postprocess, reframe_volume, squeeze};

// const MODEL_INP_NAME: String = String::from("inp");
// const MODEL_OUT_NAME: String = String::from("out");

const MP2RAGE_MODEL: &str = "./model/tf_model_mp2rage";
const MPRAGE_MODEL: &str = "./model/tf_model_mprage";
const MP2RAGE_ONNX_MODEL: &str = "./model/mp2rage.onnx";
const MPRAGE_ONNX_MODEL: &str = "./model/mprage.onnx";
const VAR_OUT_CHN: usize = 8;
const TPL_INP_SHP: (usize, usize, usize) = (64, 64, 64);
const INPUT_OP_NAME: &str = "serving_default_inp";
const OUTPUT_OP_NAME: &str = "StatefulPartitionedCall";

enum InferenceBackend {
    Onnx(OnnxSession),
    TensorFlow {
        _graph: Graph,
        bundle: SavedModelBundle,
        input: Operation,
        output: Operation,
    },
}

impl InferenceBackend {
    fn load(use_tf: bool, tf_model: &str, onnx_model: &str) -> Self {
        if use_tf {
            let mut graph = Graph::new();
            let bundle =
                SavedModelBundle::load(&SessionOptions::new(), &["serve"], &mut graph, tf_model)
                    .unwrap();
            let input = graph.operation_by_name_required(INPUT_OP_NAME).unwrap();
            let output = graph.operation_by_name_required(OUTPUT_OP_NAME).unwrap();
            Self::TensorFlow {
                _graph: graph,
                bundle,
                input,
                output,
            }
        } else {
            let mut builder = OnnxSession::builder().unwrap();

            #[cfg(target_vendor = "apple")]
            {
                let coreml =
                    ep::CoreML::default().with_compute_units(ep::coreml::ComputeUnits::CPUAndGPU);
                if coreml.is_available().unwrap_or(false) {
                    println!("GPU detected: enabling the CoreML execution provider");
                    builder = builder
                        .with_execution_providers([coreml.build().fail_silently()])
                        .unwrap();
                } else {
                    println!("No supported GPU execution provider detected; using CPU");
                }
            }

            #[cfg(not(target_vendor = "apple"))]
            println!("No supported GPU execution provider detected; using CPU");

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
                let (shape, data) = outputs["out"].try_extract_tensor::<f32>().unwrap();
                Array::from_shape_vec(
                    shape
                        .iter()
                        .map(|&dimension| dimension as usize)
                        .collect::<Vec<_>>(),
                    data.to_vec(),
                )
                .unwrap()
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
        }
    }
}

// BUG: REGION MISSES IN FAST MODE

fn interface() -> ArgMatches {
    let matches = Command::new("tiramisu")
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
                .action(clap::ArgAction::SetTrue),
        )
        .arg(
            Arg::new("reframe")
                .long("reframe")
                .help("Symmetrically pad the volume to a complete patch grid")
                .action(clap::ArgAction::SetTrue),
        )
        .get_matches();
    matches
}

fn main() {
    // CONFIG

    let arg_matches: ArgMatches = interface();
    let run_mode: &str = arg_matches.get_one::<String>("mode").expect("").as_str();
    let run_type: &str = arg_matches.get_one::<String>("type").expect("").as_str();
    let use_tf = arg_matches.get_flag("use-tf");
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

    let mut backend = InferenceBackend::load(use_tf, tf_model, onnx_model);

    let mut counter = 0;
    let out_len = px * py * pz * VAR_OUT_CHN;

    for ixp in 0..nx_patches {
        let ind_x1 = ixp * sx;
        let ind_x2 = ind_x1 + px;

        for iyp in 0..ny_patches {
            let ind_y1 = iyp * sy;
            let ind_y2 = ind_y1 + py;

            for izp in 0..nz_patches {
                let ind_z1 = izp * sz;
                let ind_z2 = ind_z1 + pz;

                let now = Instant::now();
                print!("Working on patch {} out of {}", counter, num_patches);

                let patch = inference_volume
                    .slice(s![ind_x1..ind_x2, ind_y1..ind_y2, ind_z1..ind_z2])
                    .as_standard_layout()
                    .into_owned();

                let pred_array = backend.run(&patch);
                assert_eq!(pred_array.len(), out_len);
                let pred_array = squeeze(pred_array.view()).to_owned();

                {
                    let mut out_sub: ArrayViewMut4<f32> =
                        ary_out.slice_mut(s![ind_x1..ind_x2, ind_y1..ind_y2, ind_z1..ind_z2, ..]);

                    out_sub += &pred_array;

                    let mut cnt_sub =
                        ary_counter.slice_mut(s![ind_x1..ind_x2, ind_y1..ind_y2, ind_z1..ind_z2]);

                    cnt_sub += 1.0;
                }

                counter += 1;
                println!(" took {} ms", now.elapsed().as_millis());
            }
        }
    }

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
