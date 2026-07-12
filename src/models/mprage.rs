// Burn model implementation for MPRAGE inference.
use burn::nn::InstanceNorm;
use burn::nn::InstanceNormConfig;
use burn::nn::PaddingConfig3d;
use burn::nn::conv::Conv3d;
use burn::nn::conv::Conv3dConfig;
use burn::nn::conv::ConvTranspose3d;
use burn::nn::conv::ConvTranspose3dConfig;
use burn::prelude::*;
use burn::tensor::Bytes;
use burn_store::BurnpackStore;
use burn_store::ModuleSnapshot;

#[derive(Module, Debug)]
pub struct Submodule1<B: Backend> {
    conv3d1: Conv3d<B>,
    instancenormalization1: InstanceNorm<B>,
    conv3d2: Conv3d<B>,
    instancenormalization2: InstanceNorm<B>,
    conv3d3: Conv3d<B>,
    conv3d4: Conv3d<B>,
    conv3d5: Conv3d<B>,
    instancenormalization3: InstanceNorm<B>,
    conv3d6: Conv3d<B>,
    instancenormalization4: InstanceNorm<B>,
    conv3d7: Conv3d<B>,
    instancenormalization5: InstanceNorm<B>,
    conv3d8: Conv3d<B>,
    instancenormalization6: InstanceNorm<B>,
    conv3d9: Conv3d<B>,
    phantom: core::marker::PhantomData<B>,
    #[module(skip)]
    device: B::Device,
}
impl<B: Backend> Submodule1<B> {
    #[allow(unused_variables)]
    pub fn new(device: &B::Device) -> Self {
        let conv3d1 = Conv3dConfig::new([1, 32], [3, 3, 3])
            .with_stride([1, 1, 1])
            .with_padding(PaddingConfig3d::Explicit(1, 1, 1))
            .with_dilation([1, 1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let instancenormalization1 = InstanceNormConfig::new(32)
            .with_epsilon(0.0010000000474974513f64)
            .init(device);
        let conv3d2 = Conv3dConfig::new([32, 16], [3, 3, 3])
            .with_stride([1, 1, 1])
            .with_padding(PaddingConfig3d::Explicit(1, 1, 1))
            .with_dilation([1, 1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let instancenormalization2 = InstanceNormConfig::new(48)
            .with_epsilon(0.0010000000474974513f64)
            .init(device);
        let conv3d3 = Conv3dConfig::new([48, 16], [3, 3, 3])
            .with_stride([1, 1, 1])
            .with_padding(PaddingConfig3d::Explicit(1, 1, 1))
            .with_dilation([1, 1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv3d4 = Conv3dConfig::new([64, 32], [1, 1, 1])
            .with_stride([1, 1, 1])
            .with_padding(PaddingConfig3d::Valid)
            .with_dilation([1, 1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv3d5 = Conv3dConfig::new([32, 64], [1, 1, 1])
            .with_stride([1, 1, 1])
            .with_padding(PaddingConfig3d::Valid)
            .with_dilation([1, 1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let instancenormalization3 = InstanceNormConfig::new(64)
            .with_epsilon(0.0010000000474974513f64)
            .init(device);
        let conv3d6 = Conv3dConfig::new([64, 16], [3, 3, 3])
            .with_stride([1, 1, 1])
            .with_padding(PaddingConfig3d::Explicit(1, 1, 1))
            .with_dilation([1, 1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let instancenormalization4 = InstanceNormConfig::new(80)
            .with_epsilon(0.0010000000474974513f64)
            .init(device);
        let conv3d7 = Conv3dConfig::new([80, 16], [3, 3, 3])
            .with_stride([1, 1, 1])
            .with_padding(PaddingConfig3d::Explicit(1, 1, 1))
            .with_dilation([1, 1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let instancenormalization5 = InstanceNormConfig::new(96)
            .with_epsilon(0.0010000000474974513f64)
            .init(device);
        let conv3d8 = Conv3dConfig::new([96, 16], [3, 3, 3])
            .with_stride([1, 1, 1])
            .with_padding(PaddingConfig3d::Explicit(1, 1, 1))
            .with_dilation([1, 1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let instancenormalization6 = InstanceNormConfig::new(112)
            .with_epsilon(0.0010000000474974513f64)
            .init(device);
        let conv3d9 = Conv3dConfig::new([112, 16], [3, 3, 3])
            .with_stride([1, 1, 1])
            .with_padding(PaddingConfig3d::Explicit(1, 1, 1))
            .with_dilation([1, 1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        Self {
            conv3d1,
            instancenormalization1,
            conv3d2,
            instancenormalization2,
            conv3d3,
            conv3d4,
            conv3d5,
            instancenormalization3,
            conv3d6,
            instancenormalization4,
            conv3d7,
            instancenormalization5,
            conv3d8,
            instancenormalization6,
            conv3d9,
            phantom: core::marker::PhantomData,
            device: device.clone(),
        }
    }
    #[allow(clippy::let_and_return, clippy::approx_constant)]
    pub fn forward(&self, inp: Tensor<B, 5>) -> (Tensor<B, 5>, Tensor<B, 5>) {
        let reshape1_out1 = inp.reshape([-1, 1, 64, 64, 64]);
        let conv3d1_out1 = self.conv3d1.forward(reshape1_out1);
        let cast1_out1 = conv3d1_out1.clone().cast(burn::tensor::DType::F32);
        let instancenormalization1_out1 = self.instancenormalization1.forward(cast1_out1);
        let cast2_out1 = instancenormalization1_out1.cast(burn::tensor::DType::F16);
        let relu1_out1 = burn::tensor::activation::relu(cast2_out1);
        let conv3d2_out1 = self.conv3d2.forward(relu1_out1);
        let concat1_out1 =
            burn::tensor::Tensor::cat([conv3d1_out1.clone(), conv3d2_out1.clone()].into(), 1);
        let cast3_out1 = concat1_out1.cast(burn::tensor::DType::F32);
        let instancenormalization2_out1 = self.instancenormalization2.forward(cast3_out1);
        let cast4_out1 = instancenormalization2_out1.cast(burn::tensor::DType::F16);
        let relu2_out1 = burn::tensor::activation::relu(cast4_out1);
        let conv3d3_out1 = self.conv3d3.forward(relu2_out1);
        let concat2_out1 =
            burn::tensor::Tensor::cat([conv3d2_out1, conv3d3_out1, conv3d1_out1].into(), 1);
        let reducemean1_out1 = { concat2_out1.clone().mean_dim(2usize).mean_dim(3usize) };
        let reducemean2_out1 = { concat2_out1.clone().mean_dim(2usize).mean_dim(4usize) };
        let reducemean3_out1 = { concat2_out1.clone().mean_dim(3usize).mean_dim(4usize) };
        let add1_out1 = reducemean3_out1.add(reducemean2_out1);
        let add2_out1 = add1_out1.add(reducemean1_out1);
        let conv3d4_out1 = self.conv3d4.forward(add2_out1);
        let relu3_out1 = burn::tensor::activation::relu(conv3d4_out1);
        let conv3d5_out1 = self.conv3d5.forward(relu3_out1);
        let sigmoid1_out1 = burn::tensor::activation::sigmoid(conv3d5_out1);
        let mul1_out1 = concat2_out1.mul(sigmoid1_out1);
        let slice1_out1 = mul1_out1
            .clone()
            .slice(s![.., .., 0..64; 2, 0..64; 2, 0..64; 2]);
        let slice2_out1 = mul1_out1
            .clone()
            .slice(s![.., .., 0..64; 2, 0..64; 2, 1..64; 2]);
        let slice3_out1 = mul1_out1
            .clone()
            .slice(s![.., .., 0..64; 2, 1..64; 2, 0..64; 2]);
        let slice4_out1 = mul1_out1
            .clone()
            .slice(s![.., .., 0..64; 2, 1..64; 2, 1..64; 2]);
        let slice5_out1 = mul1_out1
            .clone()
            .slice(s![.., .., 1..64; 2, 0..64; 2, 0..64; 2]);
        let slice6_out1 = mul1_out1
            .clone()
            .slice(s![.., .., 1..64; 2, 0..64; 2, 1..64; 2]);
        let slice7_out1 = mul1_out1
            .clone()
            .slice(s![.., .., 1..64; 2, 1..64; 2, 0..64; 2]);
        let slice8_out1 = mul1_out1
            .clone()
            .slice(s![.., .., 1..64; 2, 1..64; 2, 1..64; 2]);
        let cast5_out1 = slice1_out1.cast(burn::tensor::DType::F32);
        let cast6_out1 = slice2_out1.cast(burn::tensor::DType::F32);
        let max1_out1 = cast5_out1.max_pair(cast6_out1);
        let cast7_out1 = slice3_out1.cast(burn::tensor::DType::F32);
        let max2_out1 = max1_out1.max_pair(cast7_out1);
        let cast8_out1 = slice4_out1.cast(burn::tensor::DType::F32);
        let max3_out1 = max2_out1.max_pair(cast8_out1);
        let cast9_out1 = slice5_out1.cast(burn::tensor::DType::F32);
        let max4_out1 = max3_out1.max_pair(cast9_out1);
        let cast10_out1 = slice6_out1.cast(burn::tensor::DType::F32);
        let max5_out1 = max4_out1.max_pair(cast10_out1);
        let cast11_out1 = slice7_out1.cast(burn::tensor::DType::F32);
        let max6_out1 = max5_out1.max_pair(cast11_out1);
        let cast12_out1 = slice8_out1.cast(burn::tensor::DType::F32);
        let max7_out1 = max6_out1.max_pair(cast12_out1);
        let cast13_out1 = max7_out1.cast(burn::tensor::DType::F16);
        let cast14_out1 = cast13_out1.clone().cast(burn::tensor::DType::F32);
        let instancenormalization3_out1 = self.instancenormalization3.forward(cast14_out1);
        let cast15_out1 = instancenormalization3_out1.cast(burn::tensor::DType::F16);
        let relu4_out1 = burn::tensor::activation::relu(cast15_out1);
        let conv3d6_out1 = self.conv3d6.forward(relu4_out1);
        let concat3_out1 =
            burn::tensor::Tensor::cat([cast13_out1.clone(), conv3d6_out1.clone()].into(), 1);
        let cast16_out1 = concat3_out1.clone().cast(burn::tensor::DType::F32);
        let instancenormalization4_out1 = self.instancenormalization4.forward(cast16_out1);
        let cast17_out1 = instancenormalization4_out1.cast(burn::tensor::DType::F16);
        let relu5_out1 = burn::tensor::activation::relu(cast17_out1);
        let conv3d7_out1 = self.conv3d7.forward(relu5_out1);
        let concat4_out1 =
            burn::tensor::Tensor::cat([concat3_out1, conv3d7_out1.clone()].into(), 1);
        let cast18_out1 = concat4_out1.clone().cast(burn::tensor::DType::F32);
        let instancenormalization5_out1 = self.instancenormalization5.forward(cast18_out1);
        let cast19_out1 = instancenormalization5_out1.cast(burn::tensor::DType::F16);
        let relu6_out1 = burn::tensor::activation::relu(cast19_out1);
        let conv3d8_out1 = self.conv3d8.forward(relu6_out1);
        let concat5_out1 =
            burn::tensor::Tensor::cat([concat4_out1, conv3d8_out1.clone()].into(), 1);
        let cast20_out1 = concat5_out1.cast(burn::tensor::DType::F32);
        let instancenormalization6_out1 = self.instancenormalization6.forward(cast20_out1);
        let cast21_out1 = instancenormalization6_out1.cast(burn::tensor::DType::F16);
        let relu7_out1 = burn::tensor::activation::relu(cast21_out1);
        let conv3d9_out1 = self.conv3d9.forward(relu7_out1);
        let concat6_out1 = burn::tensor::Tensor::cat(
            [
                conv3d6_out1,
                conv3d7_out1,
                conv3d8_out1,
                conv3d9_out1,
                cast13_out1,
            ]
            .into(),
            1,
        );
        (concat6_out1, mul1_out1)
    }
}
#[derive(Module, Debug)]
pub struct Submodule2<B: Backend> {
    conv3d10: Conv3d<B>,
    conv3d11: Conv3d<B>,
    instancenormalization7: InstanceNorm<B>,
    conv3d12: Conv3d<B>,
    instancenormalization8: InstanceNorm<B>,
    conv3d13: Conv3d<B>,
    instancenormalization9: InstanceNorm<B>,
    conv3d14: Conv3d<B>,
    instancenormalization10: InstanceNorm<B>,
    conv3d15: Conv3d<B>,
    instancenormalization11: InstanceNorm<B>,
    conv3d16: Conv3d<B>,
    instancenormalization12: InstanceNorm<B>,
    conv3d17: Conv3d<B>,
    instancenormalization13: InstanceNorm<B>,
    conv3d18: Conv3d<B>,
    instancenormalization14: InstanceNorm<B>,
    conv3d19: Conv3d<B>,
    phantom: core::marker::PhantomData<B>,
    #[module(skip)]
    device: B::Device,
}
impl<B: Backend> Submodule2<B> {
    #[allow(unused_variables)]
    pub fn new(device: &B::Device) -> Self {
        let conv3d10 = Conv3dConfig::new([128, 64], [1, 1, 1])
            .with_stride([1, 1, 1])
            .with_padding(PaddingConfig3d::Valid)
            .with_dilation([1, 1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv3d11 = Conv3dConfig::new([64, 128], [1, 1, 1])
            .with_stride([1, 1, 1])
            .with_padding(PaddingConfig3d::Valid)
            .with_dilation([1, 1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let instancenormalization7 = InstanceNormConfig::new(128)
            .with_epsilon(0.0010000000474974513f64)
            .init(device);
        let conv3d12 = Conv3dConfig::new([128, 16], [3, 3, 3])
            .with_stride([1, 1, 1])
            .with_padding(PaddingConfig3d::Explicit(1, 1, 1))
            .with_dilation([1, 1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let instancenormalization8 = InstanceNormConfig::new(144)
            .with_epsilon(0.0010000000474974513f64)
            .init(device);
        let conv3d13 = Conv3dConfig::new([144, 16], [3, 3, 3])
            .with_stride([1, 1, 1])
            .with_padding(PaddingConfig3d::Explicit(1, 1, 1))
            .with_dilation([1, 1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let instancenormalization9 = InstanceNormConfig::new(160)
            .with_epsilon(0.0010000000474974513f64)
            .init(device);
        let conv3d14 = Conv3dConfig::new([160, 16], [3, 3, 3])
            .with_stride([1, 1, 1])
            .with_padding(PaddingConfig3d::Explicit(1, 1, 1))
            .with_dilation([1, 1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let instancenormalization10 = InstanceNormConfig::new(176)
            .with_epsilon(0.0010000000474974513f64)
            .init(device);
        let conv3d15 = Conv3dConfig::new([176, 16], [3, 3, 3])
            .with_stride([1, 1, 1])
            .with_padding(PaddingConfig3d::Explicit(1, 1, 1))
            .with_dilation([1, 1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let instancenormalization11 = InstanceNormConfig::new(192)
            .with_epsilon(0.0010000000474974513f64)
            .init(device);
        let conv3d16 = Conv3dConfig::new([192, 16], [3, 3, 3])
            .with_stride([1, 1, 1])
            .with_padding(PaddingConfig3d::Explicit(1, 1, 1))
            .with_dilation([1, 1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let instancenormalization12 = InstanceNormConfig::new(208)
            .with_epsilon(0.0010000000474974513f64)
            .init(device);
        let conv3d17 = Conv3dConfig::new([208, 16], [3, 3, 3])
            .with_stride([1, 1, 1])
            .with_padding(PaddingConfig3d::Explicit(1, 1, 1))
            .with_dilation([1, 1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let instancenormalization13 = InstanceNormConfig::new(224)
            .with_epsilon(0.0010000000474974513f64)
            .init(device);
        let conv3d18 = Conv3dConfig::new([224, 16], [3, 3, 3])
            .with_stride([1, 1, 1])
            .with_padding(PaddingConfig3d::Explicit(1, 1, 1))
            .with_dilation([1, 1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let instancenormalization14 = InstanceNormConfig::new(240)
            .with_epsilon(0.0010000000474974513f64)
            .init(device);
        let conv3d19 = Conv3dConfig::new([240, 16], [3, 3, 3])
            .with_stride([1, 1, 1])
            .with_padding(PaddingConfig3d::Explicit(1, 1, 1))
            .with_dilation([1, 1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        Self {
            conv3d10,
            conv3d11,
            instancenormalization7,
            conv3d12,
            instancenormalization8,
            conv3d13,
            instancenormalization9,
            conv3d14,
            instancenormalization10,
            conv3d15,
            instancenormalization11,
            conv3d16,
            instancenormalization12,
            conv3d17,
            instancenormalization13,
            conv3d18,
            instancenormalization14,
            conv3d19,
            phantom: core::marker::PhantomData,
            device: device.clone(),
        }
    }
    #[allow(clippy::let_and_return, clippy::approx_constant)]
    pub fn forward(&self, concat6_out1: Tensor<B, 5>) -> (Tensor<B, 5>, Tensor<B, 5>) {
        let reducemean4_out1 = { concat6_out1.clone().mean_dim(2usize).mean_dim(3usize) };
        let reducemean5_out1 = { concat6_out1.clone().mean_dim(2usize).mean_dim(4usize) };
        let reducemean6_out1 = { concat6_out1.clone().mean_dim(3usize).mean_dim(4usize) };
        let add3_out1 = reducemean6_out1.add(reducemean5_out1);
        let add4_out1 = add3_out1.add(reducemean4_out1);
        let conv3d10_out1 = self.conv3d10.forward(add4_out1);
        let relu8_out1 = burn::tensor::activation::relu(conv3d10_out1);
        let conv3d11_out1 = self.conv3d11.forward(relu8_out1);
        let sigmoid2_out1 = burn::tensor::activation::sigmoid(conv3d11_out1);
        let mul2_out1 = concat6_out1.mul(sigmoid2_out1);
        let slice9_out1 = mul2_out1
            .clone()
            .slice(s![.., .., 0..32; 2, 0..32; 2, 0..32; 2]);
        let slice10_out1 = mul2_out1
            .clone()
            .slice(s![.., .., 0..32; 2, 0..32; 2, 1..32; 2]);
        let slice11_out1 = mul2_out1
            .clone()
            .slice(s![.., .., 0..32; 2, 1..32; 2, 0..32; 2]);
        let slice12_out1 = mul2_out1
            .clone()
            .slice(s![.., .., 0..32; 2, 1..32; 2, 1..32; 2]);
        let slice13_out1 = mul2_out1
            .clone()
            .slice(s![.., .., 1..32; 2, 0..32; 2, 0..32; 2]);
        let slice14_out1 = mul2_out1
            .clone()
            .slice(s![.., .., 1..32; 2, 0..32; 2, 1..32; 2]);
        let slice15_out1 = mul2_out1
            .clone()
            .slice(s![.., .., 1..32; 2, 1..32; 2, 0..32; 2]);
        let slice16_out1 = mul2_out1
            .clone()
            .slice(s![.., .., 1..32; 2, 1..32; 2, 1..32; 2]);
        let cast22_out1 = slice9_out1.cast(burn::tensor::DType::F32);
        let cast23_out1 = slice10_out1.cast(burn::tensor::DType::F32);
        let max8_out1 = cast22_out1.max_pair(cast23_out1);
        let cast24_out1 = slice11_out1.cast(burn::tensor::DType::F32);
        let max9_out1 = max8_out1.max_pair(cast24_out1);
        let cast25_out1 = slice12_out1.cast(burn::tensor::DType::F32);
        let max10_out1 = max9_out1.max_pair(cast25_out1);
        let cast26_out1 = slice13_out1.cast(burn::tensor::DType::F32);
        let max11_out1 = max10_out1.max_pair(cast26_out1);
        let cast27_out1 = slice14_out1.cast(burn::tensor::DType::F32);
        let max12_out1 = max11_out1.max_pair(cast27_out1);
        let cast28_out1 = slice15_out1.cast(burn::tensor::DType::F32);
        let max13_out1 = max12_out1.max_pair(cast28_out1);
        let cast29_out1 = slice16_out1.cast(burn::tensor::DType::F32);
        let max14_out1 = max13_out1.max_pair(cast29_out1);
        let cast30_out1 = max14_out1.cast(burn::tensor::DType::F16);
        let cast31_out1 = cast30_out1.clone().cast(burn::tensor::DType::F32);
        let instancenormalization7_out1 = self.instancenormalization7.forward(cast31_out1);
        let cast32_out1 = instancenormalization7_out1.cast(burn::tensor::DType::F16);
        let relu9_out1 = burn::tensor::activation::relu(cast32_out1);
        let conv3d12_out1 = self.conv3d12.forward(relu9_out1);
        let concat7_out1 =
            burn::tensor::Tensor::cat([cast30_out1.clone(), conv3d12_out1.clone()].into(), 1);
        let cast33_out1 = concat7_out1.clone().cast(burn::tensor::DType::F32);
        let instancenormalization8_out1 = self.instancenormalization8.forward(cast33_out1);
        let cast34_out1 = instancenormalization8_out1.cast(burn::tensor::DType::F16);
        let relu10_out1 = burn::tensor::activation::relu(cast34_out1);
        let conv3d13_out1 = self.conv3d13.forward(relu10_out1);
        let concat8_out1 =
            burn::tensor::Tensor::cat([concat7_out1, conv3d13_out1.clone()].into(), 1);
        let cast35_out1 = concat8_out1.clone().cast(burn::tensor::DType::F32);
        let instancenormalization9_out1 = self.instancenormalization9.forward(cast35_out1);
        let cast36_out1 = instancenormalization9_out1.cast(burn::tensor::DType::F16);
        let relu11_out1 = burn::tensor::activation::relu(cast36_out1);
        let conv3d14_out1 = self.conv3d14.forward(relu11_out1);
        let concat9_out1 =
            burn::tensor::Tensor::cat([concat8_out1, conv3d14_out1.clone()].into(), 1);
        let cast37_out1 = concat9_out1.clone().cast(burn::tensor::DType::F32);
        let instancenormalization10_out1 = self.instancenormalization10.forward(cast37_out1);
        let cast38_out1 = instancenormalization10_out1.cast(burn::tensor::DType::F16);
        let relu12_out1 = burn::tensor::activation::relu(cast38_out1);
        let conv3d15_out1 = self.conv3d15.forward(relu12_out1);
        let concat10_out1 =
            burn::tensor::Tensor::cat([concat9_out1, conv3d15_out1.clone()].into(), 1);
        let cast39_out1 = concat10_out1.clone().cast(burn::tensor::DType::F32);
        let instancenormalization11_out1 = self.instancenormalization11.forward(cast39_out1);
        let cast40_out1 = instancenormalization11_out1.cast(burn::tensor::DType::F16);
        let relu13_out1 = burn::tensor::activation::relu(cast40_out1);
        let conv3d16_out1 = self.conv3d16.forward(relu13_out1);
        let concat11_out1 =
            burn::tensor::Tensor::cat([concat10_out1, conv3d16_out1.clone()].into(), 1);
        let cast41_out1 = concat11_out1.clone().cast(burn::tensor::DType::F32);
        let instancenormalization12_out1 = self.instancenormalization12.forward(cast41_out1);
        let cast42_out1 = instancenormalization12_out1.cast(burn::tensor::DType::F16);
        let relu14_out1 = burn::tensor::activation::relu(cast42_out1);
        let conv3d17_out1 = self.conv3d17.forward(relu14_out1);
        let concat12_out1 =
            burn::tensor::Tensor::cat([concat11_out1, conv3d17_out1.clone()].into(), 1);
        let cast43_out1 = concat12_out1.clone().cast(burn::tensor::DType::F32);
        let instancenormalization13_out1 = self.instancenormalization13.forward(cast43_out1);
        let cast44_out1 = instancenormalization13_out1.cast(burn::tensor::DType::F16);
        let relu15_out1 = burn::tensor::activation::relu(cast44_out1);
        let conv3d18_out1 = self.conv3d18.forward(relu15_out1);
        let concat13_out1 =
            burn::tensor::Tensor::cat([concat12_out1, conv3d18_out1.clone()].into(), 1);
        let cast45_out1 = concat13_out1.cast(burn::tensor::DType::F32);
        let instancenormalization14_out1 = self.instancenormalization14.forward(cast45_out1);
        let cast46_out1 = instancenormalization14_out1.cast(burn::tensor::DType::F16);
        let relu16_out1 = burn::tensor::activation::relu(cast46_out1);
        let conv3d19_out1 = self.conv3d19.forward(relu16_out1);
        let concat14_out1 = burn::tensor::Tensor::cat(
            [
                conv3d12_out1,
                conv3d13_out1,
                conv3d14_out1,
                conv3d15_out1,
                conv3d16_out1,
                conv3d17_out1,
                conv3d18_out1,
                conv3d19_out1,
                cast30_out1,
            ]
            .into(),
            1,
        );
        (concat14_out1, mul2_out1)
    }
}
#[derive(Module, Debug)]
pub struct Submodule3<B: Backend> {
    conv3d20: Conv3d<B>,
    conv3d21: Conv3d<B>,
    instancenormalization15: InstanceNorm<B>,
    conv3d22: Conv3d<B>,
    instancenormalization16: InstanceNorm<B>,
    conv3d23: Conv3d<B>,
    instancenormalization17: InstanceNorm<B>,
    conv3d24: Conv3d<B>,
    instancenormalization18: InstanceNorm<B>,
    conv3d25: Conv3d<B>,
    instancenormalization19: InstanceNorm<B>,
    conv3d26: Conv3d<B>,
    instancenormalization20: InstanceNorm<B>,
    conv3d27: Conv3d<B>,
    instancenormalization21: InstanceNorm<B>,
    conv3d28: Conv3d<B>,
    instancenormalization22: InstanceNorm<B>,
    conv3d29: Conv3d<B>,
    instancenormalization23: InstanceNorm<B>,
    conv3d30: Conv3d<B>,
    instancenormalization24: InstanceNorm<B>,
    conv3d31: Conv3d<B>,
    instancenormalization25: InstanceNorm<B>,
    conv3d32: Conv3d<B>,
    instancenormalization26: InstanceNorm<B>,
    conv3d33: Conv3d<B>,
    instancenormalization27: InstanceNorm<B>,
    conv3d34: Conv3d<B>,
    instancenormalization28: InstanceNorm<B>,
    conv3d35: Conv3d<B>,
    instancenormalization29: InstanceNorm<B>,
    conv3d36: Conv3d<B>,
    instancenormalization30: InstanceNorm<B>,
    conv3d37: Conv3d<B>,
    conv3d38: Conv3d<B>,
    conv3d39: Conv3d<B>,
    convtranspose3d1: ConvTranspose3d<B>,
    phantom: core::marker::PhantomData<B>,
    #[module(skip)]
    device: B::Device,
}
impl<B: Backend> Submodule3<B> {
    #[allow(unused_variables)]
    pub fn new(device: &B::Device) -> Self {
        let conv3d20 = Conv3dConfig::new([256, 128], [1, 1, 1])
            .with_stride([1, 1, 1])
            .with_padding(PaddingConfig3d::Valid)
            .with_dilation([1, 1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv3d21 = Conv3dConfig::new([128, 256], [1, 1, 1])
            .with_stride([1, 1, 1])
            .with_padding(PaddingConfig3d::Valid)
            .with_dilation([1, 1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let instancenormalization15 = InstanceNormConfig::new(256)
            .with_epsilon(0.0010000000474974513f64)
            .init(device);
        let conv3d22 = Conv3dConfig::new([256, 16], [3, 3, 3])
            .with_stride([1, 1, 1])
            .with_padding(PaddingConfig3d::Explicit(1, 1, 1))
            .with_dilation([1, 1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let instancenormalization16 = InstanceNormConfig::new(272)
            .with_epsilon(0.0010000000474974513f64)
            .init(device);
        let conv3d23 = Conv3dConfig::new([272, 16], [3, 3, 3])
            .with_stride([1, 1, 1])
            .with_padding(PaddingConfig3d::Explicit(1, 1, 1))
            .with_dilation([1, 1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let instancenormalization17 = InstanceNormConfig::new(288)
            .with_epsilon(0.0010000000474974513f64)
            .init(device);
        let conv3d24 = Conv3dConfig::new([288, 16], [3, 3, 3])
            .with_stride([1, 1, 1])
            .with_padding(PaddingConfig3d::Explicit(1, 1, 1))
            .with_dilation([1, 1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let instancenormalization18 = InstanceNormConfig::new(304)
            .with_epsilon(0.0010000000474974513f64)
            .init(device);
        let conv3d25 = Conv3dConfig::new([304, 16], [3, 3, 3])
            .with_stride([1, 1, 1])
            .with_padding(PaddingConfig3d::Explicit(1, 1, 1))
            .with_dilation([1, 1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let instancenormalization19 = InstanceNormConfig::new(320)
            .with_epsilon(0.0010000000474974513f64)
            .init(device);
        let conv3d26 = Conv3dConfig::new([320, 16], [3, 3, 3])
            .with_stride([1, 1, 1])
            .with_padding(PaddingConfig3d::Explicit(1, 1, 1))
            .with_dilation([1, 1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let instancenormalization20 = InstanceNormConfig::new(336)
            .with_epsilon(0.0010000000474974513f64)
            .init(device);
        let conv3d27 = Conv3dConfig::new([336, 16], [3, 3, 3])
            .with_stride([1, 1, 1])
            .with_padding(PaddingConfig3d::Explicit(1, 1, 1))
            .with_dilation([1, 1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let instancenormalization21 = InstanceNormConfig::new(352)
            .with_epsilon(0.0010000000474974513f64)
            .init(device);
        let conv3d28 = Conv3dConfig::new([352, 16], [3, 3, 3])
            .with_stride([1, 1, 1])
            .with_padding(PaddingConfig3d::Explicit(1, 1, 1))
            .with_dilation([1, 1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let instancenormalization22 = InstanceNormConfig::new(368)
            .with_epsilon(0.0010000000474974513f64)
            .init(device);
        let conv3d29 = Conv3dConfig::new([368, 16], [3, 3, 3])
            .with_stride([1, 1, 1])
            .with_padding(PaddingConfig3d::Explicit(1, 1, 1))
            .with_dilation([1, 1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let instancenormalization23 = InstanceNormConfig::new(384)
            .with_epsilon(0.0010000000474974513f64)
            .init(device);
        let conv3d30 = Conv3dConfig::new([384, 16], [3, 3, 3])
            .with_stride([1, 1, 1])
            .with_padding(PaddingConfig3d::Explicit(1, 1, 1))
            .with_dilation([1, 1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let instancenormalization24 = InstanceNormConfig::new(400)
            .with_epsilon(0.0010000000474974513f64)
            .init(device);
        let conv3d31 = Conv3dConfig::new([400, 16], [3, 3, 3])
            .with_stride([1, 1, 1])
            .with_padding(PaddingConfig3d::Explicit(1, 1, 1))
            .with_dilation([1, 1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let instancenormalization25 = InstanceNormConfig::new(416)
            .with_epsilon(0.0010000000474974513f64)
            .init(device);
        let conv3d32 = Conv3dConfig::new([416, 16], [3, 3, 3])
            .with_stride([1, 1, 1])
            .with_padding(PaddingConfig3d::Explicit(1, 1, 1))
            .with_dilation([1, 1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let instancenormalization26 = InstanceNormConfig::new(432)
            .with_epsilon(0.0010000000474974513f64)
            .init(device);
        let conv3d33 = Conv3dConfig::new([432, 16], [3, 3, 3])
            .with_stride([1, 1, 1])
            .with_padding(PaddingConfig3d::Explicit(1, 1, 1))
            .with_dilation([1, 1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let instancenormalization27 = InstanceNormConfig::new(448)
            .with_epsilon(0.0010000000474974513f64)
            .init(device);
        let conv3d34 = Conv3dConfig::new([448, 16], [3, 3, 3])
            .with_stride([1, 1, 1])
            .with_padding(PaddingConfig3d::Explicit(1, 1, 1))
            .with_dilation([1, 1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let instancenormalization28 = InstanceNormConfig::new(464)
            .with_epsilon(0.0010000000474974513f64)
            .init(device);
        let conv3d35 = Conv3dConfig::new([464, 16], [3, 3, 3])
            .with_stride([1, 1, 1])
            .with_padding(PaddingConfig3d::Explicit(1, 1, 1))
            .with_dilation([1, 1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let instancenormalization29 = InstanceNormConfig::new(480)
            .with_epsilon(0.0010000000474974513f64)
            .init(device);
        let conv3d36 = Conv3dConfig::new([480, 16], [3, 3, 3])
            .with_stride([1, 1, 1])
            .with_padding(PaddingConfig3d::Explicit(1, 1, 1))
            .with_dilation([1, 1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let instancenormalization30 = InstanceNormConfig::new(496)
            .with_epsilon(0.0010000000474974513f64)
            .init(device);
        let conv3d37 = Conv3dConfig::new([496, 16], [3, 3, 3])
            .with_stride([1, 1, 1])
            .with_padding(PaddingConfig3d::Explicit(1, 1, 1))
            .with_dilation([1, 1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv3d38 = Conv3dConfig::new([256, 128], [1, 1, 1])
            .with_stride([1, 1, 1])
            .with_padding(PaddingConfig3d::Valid)
            .with_dilation([1, 1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv3d39 = Conv3dConfig::new([128, 256], [1, 1, 1])
            .with_stride([1, 1, 1])
            .with_padding(PaddingConfig3d::Valid)
            .with_dilation([1, 1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let convtranspose3d1 = ConvTranspose3dConfig::new([256, 256], [2, 2, 2])
            .with_stride([2, 2, 2])
            .with_padding([0, 0, 0])
            .with_padding_out([0, 0, 0])
            .with_dilation([1, 1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        Self {
            conv3d20,
            conv3d21,
            instancenormalization15,
            conv3d22,
            instancenormalization16,
            conv3d23,
            instancenormalization17,
            conv3d24,
            instancenormalization18,
            conv3d25,
            instancenormalization19,
            conv3d26,
            instancenormalization20,
            conv3d27,
            instancenormalization21,
            conv3d28,
            instancenormalization22,
            conv3d29,
            instancenormalization23,
            conv3d30,
            instancenormalization24,
            conv3d31,
            instancenormalization25,
            conv3d32,
            instancenormalization26,
            conv3d33,
            instancenormalization27,
            conv3d34,
            instancenormalization28,
            conv3d35,
            instancenormalization29,
            conv3d36,
            instancenormalization30,
            conv3d37,
            conv3d38,
            conv3d39,
            convtranspose3d1,
            phantom: core::marker::PhantomData,
            device: device.clone(),
        }
    }
    #[allow(clippy::let_and_return, clippy::approx_constant)]
    pub fn forward(&self, concat14_out1: Tensor<B, 5>) -> Tensor<B, 5> {
        let reducemean7_out1 = { concat14_out1.clone().mean_dim(2usize).mean_dim(3usize) };
        let reducemean8_out1 = { concat14_out1.clone().mean_dim(2usize).mean_dim(4usize) };
        let reducemean9_out1 = { concat14_out1.clone().mean_dim(3usize).mean_dim(4usize) };
        let add5_out1 = reducemean9_out1.add(reducemean8_out1);
        let add6_out1 = add5_out1.add(reducemean7_out1);
        let conv3d20_out1 = self.conv3d20.forward(add6_out1);
        let relu17_out1 = burn::tensor::activation::relu(conv3d20_out1);
        let conv3d21_out1 = self.conv3d21.forward(relu17_out1);
        let sigmoid3_out1 = burn::tensor::activation::sigmoid(conv3d21_out1);
        let mul3_out1 = concat14_out1.mul(sigmoid3_out1);
        let slice17_out1 = mul3_out1
            .clone()
            .slice(s![.., .., 0..16; 2, 0..16; 2, 0..16; 2]);
        let slice18_out1 = mul3_out1
            .clone()
            .slice(s![.., .., 0..16; 2, 0..16; 2, 1..16; 2]);
        let slice19_out1 = mul3_out1
            .clone()
            .slice(s![.., .., 0..16; 2, 1..16; 2, 0..16; 2]);
        let slice20_out1 = mul3_out1
            .clone()
            .slice(s![.., .., 0..16; 2, 1..16; 2, 1..16; 2]);
        let slice21_out1 = mul3_out1
            .clone()
            .slice(s![.., .., 1..16; 2, 0..16; 2, 0..16; 2]);
        let slice22_out1 = mul3_out1
            .clone()
            .slice(s![.., .., 1..16; 2, 0..16; 2, 1..16; 2]);
        let slice23_out1 = mul3_out1
            .clone()
            .slice(s![.., .., 1..16; 2, 1..16; 2, 0..16; 2]);
        let slice24_out1 = mul3_out1
            .clone()
            .slice(s![.., .., 1..16; 2, 1..16; 2, 1..16; 2]);
        let cast47_out1 = slice17_out1.cast(burn::tensor::DType::F32);
        let cast48_out1 = slice18_out1.cast(burn::tensor::DType::F32);
        let max15_out1 = cast47_out1.max_pair(cast48_out1);
        let cast49_out1 = slice19_out1.cast(burn::tensor::DType::F32);
        let max16_out1 = max15_out1.max_pair(cast49_out1);
        let cast50_out1 = slice20_out1.cast(burn::tensor::DType::F32);
        let max17_out1 = max16_out1.max_pair(cast50_out1);
        let cast51_out1 = slice21_out1.cast(burn::tensor::DType::F32);
        let max18_out1 = max17_out1.max_pair(cast51_out1);
        let cast52_out1 = slice22_out1.cast(burn::tensor::DType::F32);
        let max19_out1 = max18_out1.max_pair(cast52_out1);
        let cast53_out1 = slice23_out1.cast(burn::tensor::DType::F32);
        let max20_out1 = max19_out1.max_pair(cast53_out1);
        let cast54_out1 = slice24_out1.cast(burn::tensor::DType::F32);
        let max21_out1 = max20_out1.max_pair(cast54_out1);
        let cast55_out1 = max21_out1.cast(burn::tensor::DType::F16);
        let cast56_out1 = cast55_out1.clone().cast(burn::tensor::DType::F32);
        let instancenormalization15_out1 = self.instancenormalization15.forward(cast56_out1);
        let cast57_out1 = instancenormalization15_out1.cast(burn::tensor::DType::F16);
        let relu18_out1 = burn::tensor::activation::relu(cast57_out1);
        let conv3d22_out1 = self.conv3d22.forward(relu18_out1);
        let concat15_out1 =
            burn::tensor::Tensor::cat([cast55_out1, conv3d22_out1.clone()].into(), 1);
        let cast58_out1 = concat15_out1.clone().cast(burn::tensor::DType::F32);
        let instancenormalization16_out1 = self.instancenormalization16.forward(cast58_out1);
        let cast59_out1 = instancenormalization16_out1.cast(burn::tensor::DType::F16);
        let relu19_out1 = burn::tensor::activation::relu(cast59_out1);
        let conv3d23_out1 = self.conv3d23.forward(relu19_out1);
        let concat16_out1 =
            burn::tensor::Tensor::cat([concat15_out1, conv3d23_out1.clone()].into(), 1);
        let cast60_out1 = concat16_out1.clone().cast(burn::tensor::DType::F32);
        let instancenormalization17_out1 = self.instancenormalization17.forward(cast60_out1);
        let cast61_out1 = instancenormalization17_out1.cast(burn::tensor::DType::F16);
        let relu20_out1 = burn::tensor::activation::relu(cast61_out1);
        let conv3d24_out1 = self.conv3d24.forward(relu20_out1);
        let concat17_out1 =
            burn::tensor::Tensor::cat([concat16_out1, conv3d24_out1.clone()].into(), 1);
        let cast62_out1 = concat17_out1.clone().cast(burn::tensor::DType::F32);
        let instancenormalization18_out1 = self.instancenormalization18.forward(cast62_out1);
        let cast63_out1 = instancenormalization18_out1.cast(burn::tensor::DType::F16);
        let relu21_out1 = burn::tensor::activation::relu(cast63_out1);
        let conv3d25_out1 = self.conv3d25.forward(relu21_out1);
        let concat18_out1 =
            burn::tensor::Tensor::cat([concat17_out1, conv3d25_out1.clone()].into(), 1);
        let cast64_out1 = concat18_out1.clone().cast(burn::tensor::DType::F32);
        let instancenormalization19_out1 = self.instancenormalization19.forward(cast64_out1);
        let cast65_out1 = instancenormalization19_out1.cast(burn::tensor::DType::F16);
        let relu22_out1 = burn::tensor::activation::relu(cast65_out1);
        let conv3d26_out1 = self.conv3d26.forward(relu22_out1);
        let concat19_out1 =
            burn::tensor::Tensor::cat([concat18_out1, conv3d26_out1.clone()].into(), 1);
        let cast66_out1 = concat19_out1.clone().cast(burn::tensor::DType::F32);
        let instancenormalization20_out1 = self.instancenormalization20.forward(cast66_out1);
        let cast67_out1 = instancenormalization20_out1.cast(burn::tensor::DType::F16);
        let relu23_out1 = burn::tensor::activation::relu(cast67_out1);
        let conv3d27_out1 = self.conv3d27.forward(relu23_out1);
        let concat20_out1 =
            burn::tensor::Tensor::cat([concat19_out1, conv3d27_out1.clone()].into(), 1);
        let cast68_out1 = concat20_out1.clone().cast(burn::tensor::DType::F32);
        let instancenormalization21_out1 = self.instancenormalization21.forward(cast68_out1);
        let cast69_out1 = instancenormalization21_out1.cast(burn::tensor::DType::F16);
        let relu24_out1 = burn::tensor::activation::relu(cast69_out1);
        let conv3d28_out1 = self.conv3d28.forward(relu24_out1);
        let concat21_out1 =
            burn::tensor::Tensor::cat([concat20_out1, conv3d28_out1.clone()].into(), 1);
        let cast70_out1 = concat21_out1.clone().cast(burn::tensor::DType::F32);
        let instancenormalization22_out1 = self.instancenormalization22.forward(cast70_out1);
        let cast71_out1 = instancenormalization22_out1.cast(burn::tensor::DType::F16);
        let relu25_out1 = burn::tensor::activation::relu(cast71_out1);
        let conv3d29_out1 = self.conv3d29.forward(relu25_out1);
        let concat22_out1 =
            burn::tensor::Tensor::cat([concat21_out1, conv3d29_out1.clone()].into(), 1);
        let cast72_out1 = concat22_out1.clone().cast(burn::tensor::DType::F32);
        let instancenormalization23_out1 = self.instancenormalization23.forward(cast72_out1);
        let cast73_out1 = instancenormalization23_out1.cast(burn::tensor::DType::F16);
        let relu26_out1 = burn::tensor::activation::relu(cast73_out1);
        let conv3d30_out1 = self.conv3d30.forward(relu26_out1);
        let concat23_out1 =
            burn::tensor::Tensor::cat([concat22_out1, conv3d30_out1.clone()].into(), 1);
        let cast74_out1 = concat23_out1.clone().cast(burn::tensor::DType::F32);
        let instancenormalization24_out1 = self.instancenormalization24.forward(cast74_out1);
        let cast75_out1 = instancenormalization24_out1.cast(burn::tensor::DType::F16);
        let relu27_out1 = burn::tensor::activation::relu(cast75_out1);
        let conv3d31_out1 = self.conv3d31.forward(relu27_out1);
        let concat24_out1 =
            burn::tensor::Tensor::cat([concat23_out1, conv3d31_out1.clone()].into(), 1);
        let cast76_out1 = concat24_out1.clone().cast(burn::tensor::DType::F32);
        let instancenormalization25_out1 = self.instancenormalization25.forward(cast76_out1);
        let cast77_out1 = instancenormalization25_out1.cast(burn::tensor::DType::F16);
        let relu28_out1 = burn::tensor::activation::relu(cast77_out1);
        let conv3d32_out1 = self.conv3d32.forward(relu28_out1);
        let concat25_out1 =
            burn::tensor::Tensor::cat([concat24_out1, conv3d32_out1.clone()].into(), 1);
        let cast78_out1 = concat25_out1.clone().cast(burn::tensor::DType::F32);
        let instancenormalization26_out1 = self.instancenormalization26.forward(cast78_out1);
        let cast79_out1 = instancenormalization26_out1.cast(burn::tensor::DType::F16);
        let relu29_out1 = burn::tensor::activation::relu(cast79_out1);
        let conv3d33_out1 = self.conv3d33.forward(relu29_out1);
        let concat26_out1 =
            burn::tensor::Tensor::cat([concat25_out1, conv3d33_out1.clone()].into(), 1);
        let cast80_out1 = concat26_out1.clone().cast(burn::tensor::DType::F32);
        let instancenormalization27_out1 = self.instancenormalization27.forward(cast80_out1);
        let cast81_out1 = instancenormalization27_out1.cast(burn::tensor::DType::F16);
        let relu30_out1 = burn::tensor::activation::relu(cast81_out1);
        let conv3d34_out1 = self.conv3d34.forward(relu30_out1);
        let concat27_out1 =
            burn::tensor::Tensor::cat([concat26_out1, conv3d34_out1.clone()].into(), 1);
        let cast82_out1 = concat27_out1.clone().cast(burn::tensor::DType::F32);
        let instancenormalization28_out1 = self.instancenormalization28.forward(cast82_out1);
        let cast83_out1 = instancenormalization28_out1.cast(burn::tensor::DType::F16);
        let relu31_out1 = burn::tensor::activation::relu(cast83_out1);
        let conv3d35_out1 = self.conv3d35.forward(relu31_out1);
        let concat28_out1 =
            burn::tensor::Tensor::cat([concat27_out1, conv3d35_out1.clone()].into(), 1);
        let cast84_out1 = concat28_out1.clone().cast(burn::tensor::DType::F32);
        let instancenormalization29_out1 = self.instancenormalization29.forward(cast84_out1);
        let cast85_out1 = instancenormalization29_out1.cast(burn::tensor::DType::F16);
        let relu32_out1 = burn::tensor::activation::relu(cast85_out1);
        let conv3d36_out1 = self.conv3d36.forward(relu32_out1);
        let concat29_out1 =
            burn::tensor::Tensor::cat([concat28_out1, conv3d36_out1.clone()].into(), 1);
        let cast86_out1 = concat29_out1.cast(burn::tensor::DType::F32);
        let instancenormalization30_out1 = self.instancenormalization30.forward(cast86_out1);
        let cast87_out1 = instancenormalization30_out1.cast(burn::tensor::DType::F16);
        let relu33_out1 = burn::tensor::activation::relu(cast87_out1);
        let conv3d37_out1 = self.conv3d37.forward(relu33_out1);
        let concat30_out1 = burn::tensor::Tensor::cat(
            [
                conv3d22_out1,
                conv3d23_out1,
                conv3d24_out1,
                conv3d25_out1,
                conv3d26_out1,
                conv3d27_out1,
                conv3d28_out1,
                conv3d29_out1,
                conv3d30_out1,
                conv3d31_out1,
                conv3d32_out1,
                conv3d33_out1,
                conv3d34_out1,
                conv3d35_out1,
                conv3d36_out1,
                conv3d37_out1,
            ]
            .into(),
            1,
        );
        let reducemean10_out1 = { concat30_out1.clone().mean_dim(2usize).mean_dim(3usize) };
        let reducemean11_out1 = { concat30_out1.clone().mean_dim(2usize).mean_dim(4usize) };
        let reducemean12_out1 = { concat30_out1.clone().mean_dim(3usize).mean_dim(4usize) };
        let add7_out1 = reducemean12_out1.add(reducemean11_out1);
        let add8_out1 = add7_out1.add(reducemean10_out1);
        let conv3d38_out1 = self.conv3d38.forward(add8_out1);
        let relu34_out1 = burn::tensor::activation::relu(conv3d38_out1);
        let conv3d39_out1 = self.conv3d39.forward(relu34_out1);
        let sigmoid4_out1 = burn::tensor::activation::sigmoid(conv3d39_out1);
        let mul4_out1 = concat30_out1.mul(sigmoid4_out1);
        let convtranspose3d1_out1 = self.convtranspose3d1.forward(mul4_out1);
        let concat31_out1 = burn::tensor::Tensor::cat([convtranspose3d1_out1, mul3_out1].into(), 1);
        concat31_out1
    }
}
#[derive(Module, Debug)]
pub struct Submodule4<B: Backend> {
    instancenormalization31: InstanceNorm<B>,
    conv3d40: Conv3d<B>,
    instancenormalization32: InstanceNorm<B>,
    conv3d41: Conv3d<B>,
    instancenormalization33: InstanceNorm<B>,
    conv3d42: Conv3d<B>,
    instancenormalization34: InstanceNorm<B>,
    conv3d43: Conv3d<B>,
    instancenormalization35: InstanceNorm<B>,
    conv3d44: Conv3d<B>,
    instancenormalization36: InstanceNorm<B>,
    conv3d45: Conv3d<B>,
    instancenormalization37: InstanceNorm<B>,
    conv3d46: Conv3d<B>,
    instancenormalization38: InstanceNorm<B>,
    conv3d47: Conv3d<B>,
    conv3d48: Conv3d<B>,
    conv3d49: Conv3d<B>,
    convtranspose3d2: ConvTranspose3d<B>,
    instancenormalization39: InstanceNorm<B>,
    conv3d50: Conv3d<B>,
    instancenormalization40: InstanceNorm<B>,
    conv3d51: Conv3d<B>,
    instancenormalization41: InstanceNorm<B>,
    conv3d52: Conv3d<B>,
    instancenormalization42: InstanceNorm<B>,
    conv3d53: Conv3d<B>,
    conv3d54: Conv3d<B>,
    conv3d55: Conv3d<B>,
    convtranspose3d3: ConvTranspose3d<B>,
    instancenormalization43: InstanceNorm<B>,
    conv3d56: Conv3d<B>,
    instancenormalization44: InstanceNorm<B>,
    conv3d57: Conv3d<B>,
    conv3d58: Conv3d<B>,
    conv3d59: Conv3d<B>,
    conv3d60: Conv3d<B>,
    phantom: core::marker::PhantomData<B>,
    #[module(skip)]
    device: B::Device,
}
impl<B: Backend> Submodule4<B> {
    #[allow(unused_variables)]
    pub fn new(device: &B::Device) -> Self {
        let instancenormalization31 = InstanceNormConfig::new(512)
            .with_epsilon(0.0010000000474974513f64)
            .init(device);
        let conv3d40 = Conv3dConfig::new([512, 16], [3, 3, 3])
            .with_stride([1, 1, 1])
            .with_padding(PaddingConfig3d::Explicit(1, 1, 1))
            .with_dilation([1, 1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let instancenormalization32 = InstanceNormConfig::new(528)
            .with_epsilon(0.0010000000474974513f64)
            .init(device);
        let conv3d41 = Conv3dConfig::new([528, 16], [3, 3, 3])
            .with_stride([1, 1, 1])
            .with_padding(PaddingConfig3d::Explicit(1, 1, 1))
            .with_dilation([1, 1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let instancenormalization33 = InstanceNormConfig::new(544)
            .with_epsilon(0.0010000000474974513f64)
            .init(device);
        let conv3d42 = Conv3dConfig::new([544, 16], [3, 3, 3])
            .with_stride([1, 1, 1])
            .with_padding(PaddingConfig3d::Explicit(1, 1, 1))
            .with_dilation([1, 1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let instancenormalization34 = InstanceNormConfig::new(560)
            .with_epsilon(0.0010000000474974513f64)
            .init(device);
        let conv3d43 = Conv3dConfig::new([560, 16], [3, 3, 3])
            .with_stride([1, 1, 1])
            .with_padding(PaddingConfig3d::Explicit(1, 1, 1))
            .with_dilation([1, 1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let instancenormalization35 = InstanceNormConfig::new(576)
            .with_epsilon(0.0010000000474974513f64)
            .init(device);
        let conv3d44 = Conv3dConfig::new([576, 16], [3, 3, 3])
            .with_stride([1, 1, 1])
            .with_padding(PaddingConfig3d::Explicit(1, 1, 1))
            .with_dilation([1, 1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let instancenormalization36 = InstanceNormConfig::new(592)
            .with_epsilon(0.0010000000474974513f64)
            .init(device);
        let conv3d45 = Conv3dConfig::new([592, 16], [3, 3, 3])
            .with_stride([1, 1, 1])
            .with_padding(PaddingConfig3d::Explicit(1, 1, 1))
            .with_dilation([1, 1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let instancenormalization37 = InstanceNormConfig::new(608)
            .with_epsilon(0.0010000000474974513f64)
            .init(device);
        let conv3d46 = Conv3dConfig::new([608, 16], [3, 3, 3])
            .with_stride([1, 1, 1])
            .with_padding(PaddingConfig3d::Explicit(1, 1, 1))
            .with_dilation([1, 1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let instancenormalization38 = InstanceNormConfig::new(624)
            .with_epsilon(0.0010000000474974513f64)
            .init(device);
        let conv3d47 = Conv3dConfig::new([624, 16], [3, 3, 3])
            .with_stride([1, 1, 1])
            .with_padding(PaddingConfig3d::Explicit(1, 1, 1))
            .with_dilation([1, 1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv3d48 = Conv3dConfig::new([128, 64], [1, 1, 1])
            .with_stride([1, 1, 1])
            .with_padding(PaddingConfig3d::Valid)
            .with_dilation([1, 1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv3d49 = Conv3dConfig::new([64, 128], [1, 1, 1])
            .with_stride([1, 1, 1])
            .with_padding(PaddingConfig3d::Valid)
            .with_dilation([1, 1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let convtranspose3d2 = ConvTranspose3dConfig::new([128, 128], [2, 2, 2])
            .with_stride([2, 2, 2])
            .with_padding([0, 0, 0])
            .with_padding_out([0, 0, 0])
            .with_dilation([1, 1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let instancenormalization39 = InstanceNormConfig::new(256)
            .with_epsilon(0.0010000000474974513f64)
            .init(device);
        let conv3d50 = Conv3dConfig::new([256, 16], [3, 3, 3])
            .with_stride([1, 1, 1])
            .with_padding(PaddingConfig3d::Explicit(1, 1, 1))
            .with_dilation([1, 1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let instancenormalization40 = InstanceNormConfig::new(272)
            .with_epsilon(0.0010000000474974513f64)
            .init(device);
        let conv3d51 = Conv3dConfig::new([272, 16], [3, 3, 3])
            .with_stride([1, 1, 1])
            .with_padding(PaddingConfig3d::Explicit(1, 1, 1))
            .with_dilation([1, 1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let instancenormalization41 = InstanceNormConfig::new(288)
            .with_epsilon(0.0010000000474974513f64)
            .init(device);
        let conv3d52 = Conv3dConfig::new([288, 16], [3, 3, 3])
            .with_stride([1, 1, 1])
            .with_padding(PaddingConfig3d::Explicit(1, 1, 1))
            .with_dilation([1, 1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let instancenormalization42 = InstanceNormConfig::new(304)
            .with_epsilon(0.0010000000474974513f64)
            .init(device);
        let conv3d53 = Conv3dConfig::new([304, 16], [3, 3, 3])
            .with_stride([1, 1, 1])
            .with_padding(PaddingConfig3d::Explicit(1, 1, 1))
            .with_dilation([1, 1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv3d54 = Conv3dConfig::new([64, 32], [1, 1, 1])
            .with_stride([1, 1, 1])
            .with_padding(PaddingConfig3d::Valid)
            .with_dilation([1, 1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv3d55 = Conv3dConfig::new([32, 64], [1, 1, 1])
            .with_stride([1, 1, 1])
            .with_padding(PaddingConfig3d::Valid)
            .with_dilation([1, 1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let convtranspose3d3 = ConvTranspose3dConfig::new([64, 64], [2, 2, 2])
            .with_stride([2, 2, 2])
            .with_padding([0, 0, 0])
            .with_padding_out([0, 0, 0])
            .with_dilation([1, 1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let instancenormalization43 = InstanceNormConfig::new(128)
            .with_epsilon(0.0010000000474974513f64)
            .init(device);
        let conv3d56 = Conv3dConfig::new([128, 16], [3, 3, 3])
            .with_stride([1, 1, 1])
            .with_padding(PaddingConfig3d::Explicit(1, 1, 1))
            .with_dilation([1, 1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let instancenormalization44 = InstanceNormConfig::new(144)
            .with_epsilon(0.0010000000474974513f64)
            .init(device);
        let conv3d57 = Conv3dConfig::new([144, 16], [3, 3, 3])
            .with_stride([1, 1, 1])
            .with_padding(PaddingConfig3d::Explicit(1, 1, 1))
            .with_dilation([1, 1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv3d58 = Conv3dConfig::new([32, 16], [1, 1, 1])
            .with_stride([1, 1, 1])
            .with_padding(PaddingConfig3d::Valid)
            .with_dilation([1, 1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv3d59 = Conv3dConfig::new([16, 32], [1, 1, 1])
            .with_stride([1, 1, 1])
            .with_padding(PaddingConfig3d::Valid)
            .with_dilation([1, 1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv3d60 = Conv3dConfig::new([32, 8], [1, 1, 1])
            .with_stride([1, 1, 1])
            .with_padding(PaddingConfig3d::Valid)
            .with_dilation([1, 1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        Self {
            instancenormalization31,
            conv3d40,
            instancenormalization32,
            conv3d41,
            instancenormalization33,
            conv3d42,
            instancenormalization34,
            conv3d43,
            instancenormalization35,
            conv3d44,
            instancenormalization36,
            conv3d45,
            instancenormalization37,
            conv3d46,
            instancenormalization38,
            conv3d47,
            conv3d48,
            conv3d49,
            convtranspose3d2,
            instancenormalization39,
            conv3d50,
            instancenormalization40,
            conv3d51,
            instancenormalization41,
            conv3d52,
            instancenormalization42,
            conv3d53,
            conv3d54,
            conv3d55,
            convtranspose3d3,
            instancenormalization43,
            conv3d56,
            instancenormalization44,
            conv3d57,
            conv3d58,
            conv3d59,
            conv3d60,
            phantom: core::marker::PhantomData,
            device: device.clone(),
        }
    }
    #[allow(clippy::let_and_return, clippy::approx_constant)]
    pub fn forward(
        &self,
        concat31_out1: Tensor<B, 5>,
        mul2_out1: Tensor<B, 5>,
        mul1_out1: Tensor<B, 5>,
    ) -> Tensor<B, 5> {
        let cast88_out1 = concat31_out1.clone().cast(burn::tensor::DType::F32);
        let instancenormalization31_out1 = self.instancenormalization31.forward(cast88_out1);
        let cast89_out1 = instancenormalization31_out1.cast(burn::tensor::DType::F16);
        let relu35_out1 = burn::tensor::activation::relu(cast89_out1);
        let conv3d40_out1 = self.conv3d40.forward(relu35_out1);
        let concat32_out1 =
            burn::tensor::Tensor::cat([concat31_out1, conv3d40_out1.clone()].into(), 1);
        let cast90_out1 = concat32_out1.clone().cast(burn::tensor::DType::F32);
        let instancenormalization32_out1 = self.instancenormalization32.forward(cast90_out1);
        let cast91_out1 = instancenormalization32_out1.cast(burn::tensor::DType::F16);
        let relu36_out1 = burn::tensor::activation::relu(cast91_out1);
        let conv3d41_out1 = self.conv3d41.forward(relu36_out1);
        let concat33_out1 =
            burn::tensor::Tensor::cat([concat32_out1, conv3d41_out1.clone()].into(), 1);
        let cast92_out1 = concat33_out1.clone().cast(burn::tensor::DType::F32);
        let instancenormalization33_out1 = self.instancenormalization33.forward(cast92_out1);
        let cast93_out1 = instancenormalization33_out1.cast(burn::tensor::DType::F16);
        let relu37_out1 = burn::tensor::activation::relu(cast93_out1);
        let conv3d42_out1 = self.conv3d42.forward(relu37_out1);
        let concat34_out1 =
            burn::tensor::Tensor::cat([concat33_out1, conv3d42_out1.clone()].into(), 1);
        let cast94_out1 = concat34_out1.clone().cast(burn::tensor::DType::F32);
        let instancenormalization34_out1 = self.instancenormalization34.forward(cast94_out1);
        let cast95_out1 = instancenormalization34_out1.cast(burn::tensor::DType::F16);
        let relu38_out1 = burn::tensor::activation::relu(cast95_out1);
        let conv3d43_out1 = self.conv3d43.forward(relu38_out1);
        let concat35_out1 =
            burn::tensor::Tensor::cat([concat34_out1, conv3d43_out1.clone()].into(), 1);
        let cast96_out1 = concat35_out1.clone().cast(burn::tensor::DType::F32);
        let instancenormalization35_out1 = self.instancenormalization35.forward(cast96_out1);
        let cast97_out1 = instancenormalization35_out1.cast(burn::tensor::DType::F16);
        let relu39_out1 = burn::tensor::activation::relu(cast97_out1);
        let conv3d44_out1 = self.conv3d44.forward(relu39_out1);
        let concat36_out1 =
            burn::tensor::Tensor::cat([concat35_out1, conv3d44_out1.clone()].into(), 1);
        let cast98_out1 = concat36_out1.clone().cast(burn::tensor::DType::F32);
        let instancenormalization36_out1 = self.instancenormalization36.forward(cast98_out1);
        let cast99_out1 = instancenormalization36_out1.cast(burn::tensor::DType::F16);
        let relu40_out1 = burn::tensor::activation::relu(cast99_out1);
        let conv3d45_out1 = self.conv3d45.forward(relu40_out1);
        let concat37_out1 =
            burn::tensor::Tensor::cat([concat36_out1, conv3d45_out1.clone()].into(), 1);
        let cast100_out1 = concat37_out1.clone().cast(burn::tensor::DType::F32);
        let instancenormalization37_out1 = self.instancenormalization37.forward(cast100_out1);
        let cast101_out1 = instancenormalization37_out1.cast(burn::tensor::DType::F16);
        let relu41_out1 = burn::tensor::activation::relu(cast101_out1);
        let conv3d46_out1 = self.conv3d46.forward(relu41_out1);
        let concat38_out1 =
            burn::tensor::Tensor::cat([concat37_out1, conv3d46_out1.clone()].into(), 1);
        let cast102_out1 = concat38_out1.cast(burn::tensor::DType::F32);
        let instancenormalization38_out1 = self.instancenormalization38.forward(cast102_out1);
        let cast103_out1 = instancenormalization38_out1.cast(burn::tensor::DType::F16);
        let relu42_out1 = burn::tensor::activation::relu(cast103_out1);
        let conv3d47_out1 = self.conv3d47.forward(relu42_out1);
        let concat39_out1 = burn::tensor::Tensor::cat(
            [
                conv3d40_out1,
                conv3d41_out1,
                conv3d42_out1,
                conv3d43_out1,
                conv3d44_out1,
                conv3d45_out1,
                conv3d46_out1,
                conv3d47_out1,
            ]
            .into(),
            1,
        );
        let reducemean13_out1 = { concat39_out1.clone().mean_dim(2usize).mean_dim(3usize) };
        let reducemean14_out1 = { concat39_out1.clone().mean_dim(2usize).mean_dim(4usize) };
        let reducemean15_out1 = { concat39_out1.clone().mean_dim(3usize).mean_dim(4usize) };
        let add9_out1 = reducemean15_out1.add(reducemean14_out1);
        let add10_out1 = add9_out1.add(reducemean13_out1);
        let conv3d48_out1 = self.conv3d48.forward(add10_out1);
        let relu43_out1 = burn::tensor::activation::relu(conv3d48_out1);
        let conv3d49_out1 = self.conv3d49.forward(relu43_out1);
        let sigmoid5_out1 = burn::tensor::activation::sigmoid(conv3d49_out1);
        let mul5_out1 = concat39_out1.mul(sigmoid5_out1);
        let convtranspose3d2_out1 = self.convtranspose3d2.forward(mul5_out1);
        let concat40_out1 = burn::tensor::Tensor::cat([convtranspose3d2_out1, mul2_out1].into(), 1);
        let cast104_out1 = concat40_out1.clone().cast(burn::tensor::DType::F32);
        let instancenormalization39_out1 = self.instancenormalization39.forward(cast104_out1);
        let cast105_out1 = instancenormalization39_out1.cast(burn::tensor::DType::F16);
        let relu44_out1 = burn::tensor::activation::relu(cast105_out1);
        let conv3d50_out1 = self.conv3d50.forward(relu44_out1);
        let concat41_out1 =
            burn::tensor::Tensor::cat([concat40_out1, conv3d50_out1.clone()].into(), 1);
        let cast106_out1 = concat41_out1.clone().cast(burn::tensor::DType::F32);
        let instancenormalization40_out1 = self.instancenormalization40.forward(cast106_out1);
        let cast107_out1 = instancenormalization40_out1.cast(burn::tensor::DType::F16);
        let relu45_out1 = burn::tensor::activation::relu(cast107_out1);
        let conv3d51_out1 = self.conv3d51.forward(relu45_out1);
        let concat42_out1 =
            burn::tensor::Tensor::cat([concat41_out1, conv3d51_out1.clone()].into(), 1);
        let cast108_out1 = concat42_out1.clone().cast(burn::tensor::DType::F32);
        let instancenormalization41_out1 = self.instancenormalization41.forward(cast108_out1);
        let cast109_out1 = instancenormalization41_out1.cast(burn::tensor::DType::F16);
        let relu46_out1 = burn::tensor::activation::relu(cast109_out1);
        let conv3d52_out1 = self.conv3d52.forward(relu46_out1);
        let concat43_out1 =
            burn::tensor::Tensor::cat([concat42_out1, conv3d52_out1.clone()].into(), 1);
        let cast110_out1 = concat43_out1.cast(burn::tensor::DType::F32);
        let instancenormalization42_out1 = self.instancenormalization42.forward(cast110_out1);
        let cast111_out1 = instancenormalization42_out1.cast(burn::tensor::DType::F16);
        let relu47_out1 = burn::tensor::activation::relu(cast111_out1);
        let conv3d53_out1 = self.conv3d53.forward(relu47_out1);
        let concat44_out1 = burn::tensor::Tensor::cat(
            [conv3d50_out1, conv3d51_out1, conv3d52_out1, conv3d53_out1].into(),
            1,
        );
        let reducemean16_out1 = { concat44_out1.clone().mean_dim(2usize).mean_dim(3usize) };
        let reducemean17_out1 = { concat44_out1.clone().mean_dim(2usize).mean_dim(4usize) };
        let reducemean18_out1 = { concat44_out1.clone().mean_dim(3usize).mean_dim(4usize) };
        let add11_out1 = reducemean18_out1.add(reducemean17_out1);
        let add12_out1 = add11_out1.add(reducemean16_out1);
        let conv3d54_out1 = self.conv3d54.forward(add12_out1);
        let relu48_out1 = burn::tensor::activation::relu(conv3d54_out1);
        let conv3d55_out1 = self.conv3d55.forward(relu48_out1);
        let sigmoid6_out1 = burn::tensor::activation::sigmoid(conv3d55_out1);
        let mul6_out1 = concat44_out1.mul(sigmoid6_out1);
        let convtranspose3d3_out1 = self.convtranspose3d3.forward(mul6_out1);
        let concat45_out1 = burn::tensor::Tensor::cat([convtranspose3d3_out1, mul1_out1].into(), 1);
        let cast112_out1 = concat45_out1.clone().cast(burn::tensor::DType::F32);
        let instancenormalization43_out1 = self.instancenormalization43.forward(cast112_out1);
        let cast113_out1 = instancenormalization43_out1.cast(burn::tensor::DType::F16);
        let relu49_out1 = burn::tensor::activation::relu(cast113_out1);
        let conv3d56_out1 = self.conv3d56.forward(relu49_out1);
        let concat46_out1 =
            burn::tensor::Tensor::cat([concat45_out1, conv3d56_out1.clone()].into(), 1);
        let cast114_out1 = concat46_out1.cast(burn::tensor::DType::F32);
        let instancenormalization44_out1 = self.instancenormalization44.forward(cast114_out1);
        let cast115_out1 = instancenormalization44_out1.cast(burn::tensor::DType::F16);
        let relu50_out1 = burn::tensor::activation::relu(cast115_out1);
        let conv3d57_out1 = self.conv3d57.forward(relu50_out1);
        let concat47_out1 = burn::tensor::Tensor::cat([conv3d56_out1, conv3d57_out1].into(), 1);
        let reducemean19_out1 = { concat47_out1.clone().mean_dim(2usize).mean_dim(3usize) };
        let reducemean20_out1 = { concat47_out1.clone().mean_dim(2usize).mean_dim(4usize) };
        let reducemean21_out1 = { concat47_out1.clone().mean_dim(3usize).mean_dim(4usize) };
        let add13_out1 = reducemean21_out1.add(reducemean20_out1);
        let add14_out1 = add13_out1.add(reducemean19_out1);
        let conv3d58_out1 = self.conv3d58.forward(add14_out1);
        let relu51_out1 = burn::tensor::activation::relu(conv3d58_out1);
        let conv3d59_out1 = self.conv3d59.forward(relu51_out1);
        let sigmoid7_out1 = burn::tensor::activation::sigmoid(conv3d59_out1);
        let mul7_out1 = concat47_out1.mul(sigmoid7_out1);
        let conv3d60_out1 = self.conv3d60.forward(mul7_out1);
        let softmax1_out1 = burn::tensor::activation::softmax(conv3d60_out1, 1);
        let transpose1_out1 = softmax1_out1.permute([0, 2, 3, 4, 1]);
        transpose1_out1
    }
}

#[derive(Module, Debug)]
pub struct Model<B: Backend> {
    submodule1: Submodule1<B>,
    submodule2: Submodule2<B>,
    submodule3: Submodule3<B>,
    submodule4: Submodule4<B>,
    phantom: core::marker::PhantomData<B>,
    #[module(skip)]
    device: B::Device,
}

extern crate std;

impl<B: Backend> Default for Model<B> {
    fn default() -> Self {
        Self::from_file("model/wgpu/mprage.bpk", &Default::default())
    }
}

impl<B: Backend> Model<B> {
    /// Load model weights from a burnpack file.
    pub fn from_file<P: AsRef<std::path::Path>>(file: P, device: &B::Device) -> Self {
        let mut model = Self::new(device);
        let mut store = BurnpackStore::from_file(file);
        model
            .load_from(&mut store)
            .expect("Failed to load burnpack file");
        model
    }

    /// Load model weights from in-memory bytes.
    ///
    /// The bytes must be the contents of a `.bpk` file.
    pub fn from_bytes(bytes: Bytes, device: &B::Device) -> Self {
        let mut model = Self::new(device);
        let mut store = BurnpackStore::from_bytes(Some(bytes));
        model
            .load_from(&mut store)
            .expect("Failed to load burnpack bytes");
        model
    }
}

impl<B: Backend> Model<B> {
    #[allow(unused_variables)]
    pub fn new(device: &B::Device) -> Self {
        let submodule1 = Submodule1::new(device);
        let submodule2 = Submodule2::new(device);
        let submodule3 = Submodule3::new(device);
        let submodule4 = Submodule4::new(device);
        Self {
            submodule1,
            submodule2,
            submodule3,
            submodule4,
            phantom: core::marker::PhantomData,
            device: device.clone(),
        }
    }

    #[allow(clippy::let_and_return, clippy::approx_constant)]
    pub fn forward(&self, inp: Tensor<B, 5>) -> Tensor<B, 5> {
        let (concat6_out1, mul1_out1) = self.submodule1.forward(inp);
        let (concat14_out1, mul2_out1) = self.submodule2.forward(concat6_out1);
        let concat31_out1 = self.submodule3.forward(concat14_out1);
        let transpose1_out1 = self.submodule4.forward(concat31_out1, mul2_out1, mul1_out1);
        transpose1_out1
    }
}
