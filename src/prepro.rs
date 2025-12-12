use ndarray::{Array3, ArrayD, Ix3};
use nifti::{IntoNdArray, NiftiObject, ReaderOptions};

pub fn load_nifti_3d(path: &str) -> Result<Array3<f32>, Box<dyn std::error::Error>> {
    let obj = ReaderOptions::new().read_file(path).unwrap();
    let _header = obj.header().clone();
    let volume: ArrayD<f32> = obj.into_volume().into_ndarray::<f32>()?;
    let volume: Array3<f32> = volume.into_dimensionality::<Ix3>()?;
    Ok(volume)
}
