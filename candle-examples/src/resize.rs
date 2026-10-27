//! Image resizing that matches the preprocessing of `transformers` exactly.
//!
//! The image processors of `transformers` resize `u8` images with torchvision's
//! `resize(..., antialias=True)`. The filters of the `image` crate are close but not equal,
//! which shifts pixels by a few levels and can change the generated text.

use candle::{DType, Result, Tensor};

/// A resampling filter, as the `resample` field of a `preprocessor_config.json`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Resample {
    Bilinear,
    Bicubic,
}

impl Resample {
    /// From a PIL resampling code: 2 for bilinear, 3 for bicubic.
    pub fn from_pil(code: usize) -> Result<Self> {
        match code {
            2 => Ok(Self::Bilinear),
            3 => Ok(Self::Bicubic),
            c => candle::bail!(
                "unsupported resample filter {c}, expected 2 (bilinear) or 3 (bicubic)"
            ),
        }
    }

    /// The filter and its support radius.
    fn filter(&self) -> (fn(f64) -> f64, f64) {
        fn bilinear(x: f64) -> f64 {
            (1. - x.abs()).max(0.)
        }
        fn bicubic(x: f64) -> f64 {
            // Keys' cubic convolution with a = -0.5, as in PIL and torchvision.
            let a = -0.5;
            let x = x.abs();
            if x < 1. {
                ((a + 2.) * x - (a + 3.)) * x * x + 1.
            } else if x < 2. {
                (((x - 5.) * x + 8.) * x - 4.) * a
            } else {
                0.
            }
        }
        match self {
            Self::Bilinear => (bilinear, 1.),
            Self::Bicubic => (bicubic, 2.),
        }
    }
}

/// For each output pixel, the first input pixel it reads and the weights of the following ones,
/// in fixed point with the returned number of fractional bits. The filter is widened when
/// downscaling, which averages all the input pixels (antialiasing).
fn coeffs(in_size: usize, out_size: usize, resample: Resample) -> (Vec<(usize, Vec<i64>)>, u32) {
    let (filter, support) = resample.filter();
    let scale = in_size as f64 / out_size as f64;
    let filter_scale = scale.max(1.);
    let support = support * filter_scale;
    let inv_filter_scale = 1. / filter_scale;
    let coeffs: Vec<(usize, Vec<f64>)> = (0..out_size)
        .map(|i| {
            let center = (i as f64 + 0.5) * scale;
            // `as` truncates toward zero like C's `(int)` casts.
            let min = ((center - support + 0.5) as i64).max(0) as usize;
            let max = ((center + support + 0.5) as i64).min(in_size as i64) as usize;
            let weights: Vec<f64> = (min..max)
                .map(|x| filter((x as f64 - center + 0.5) * inv_filter_scale))
                .collect();
            let total: f64 = weights.iter().sum();
            (min, weights.iter().map(|w| w / total).collect())
        })
        .collect();

    // The most fractional bits (up to 22) for which the largest weight fits in an i16.
    let max_weight = coeffs
        .iter()
        .flat_map(|(_, w)| w.iter().copied())
        .fold(f64::MIN, f64::max);
    let precision = (0..22)
        .find(|p| (0.5 + max_weight * (1u64 << (p + 1)) as f64) as i64 >= 1 << 15)
        .unwrap_or(22);
    let one = (1u64 << precision) as f64;
    let coeffs = coeffs
        .into_iter()
        .map(|(min, w)| (min, w.iter().map(|w| (w * one).round() as i64).collect()))
        .collect();
    (coeffs, precision)
}

/// Resamples the last dim of a `(C, rows, len)` u8 tensor to `out_len`.
fn resample_last_dim(image: &Tensor, out_len: usize, resample: Resample) -> Result<Tensor> {
    let (c, rows, len) = image.dims3()?;
    let (coeffs, precision) = coeffs(len, out_len, resample);
    let src = image.flatten_all()?.to_vec1::<u8>()?;
    let mut dst = Vec::with_capacity(c * rows * out_len);
    for row in src.chunks_exact(len) {
        for (start, weights) in coeffs.iter() {
            let acc = weights
                .iter()
                .zip(&row[*start..])
                .fold(1i64 << (precision - 1), |acc, (w, &p)| acc + w * p as i64);
            dst.push((acc >> precision).clamp(0, 255) as u8);
        }
    }
    Tensor::from_vec(dst, (c, rows, out_len), image.device())
}

/// Resizes a `(C, H, W)` u8 image to `(height, width)` exactly like torchvision's
/// `resize(image, (height, width), interpolation, antialias=True)` on a uint8 tensor: a
/// horizontal then a vertical pass in fixed point. (PIL's `Image.resize` differs by 1 or 2
/// levels on some pixels.)
pub fn resize(
    image: &Tensor,
    (height, width): (usize, usize),
    resample: Resample,
) -> Result<Tensor> {
    let (_, h, w) = image.dims3()?;
    let mut image = image.to_dtype(DType::U8)?;
    if width != w {
        image = resample_last_dim(&image, width, resample)?;
    }
    if height != h {
        let columns = image.transpose(1, 2)?.contiguous()?;
        image = resample_last_dim(&columns, height, resample)?
            .transpose(1, 2)?
            .contiguous()?;
    }
    Ok(image)
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle::Device;

    #[test]
    fn resize_matches_torchvision() -> Result<()> {
        // (c, y, x) -> (c * 50 + y * y * 13 + x * 37) % 256, a 5x7 RGB image.
        let data: Vec<u8> = (0..3)
            .flat_map(|c| {
                (0..5).flat_map(move |y| (0..7).map(move |x| c * 50 + y * y * 13 + x * 37))
            })
            .map(|v| (v % 256) as u8)
            .collect();
        let image = Tensor::from_vec(data, (3, 5, 7), &Device::Cpu)?;
        // Expected values from torchvision's `resize(..., antialias=True)`.
        let out = resize(&image, (3, 4), Resample::Bicubic)?
            .flatten_all()?
            .to_vec1::<u8>()?;
        let expected = [
            17, 79, 153, 217, 69, 142, 170, 124, 190, 130, 83, 112, 67, 131, 216, 122, 132, 177,
            171, 78, 93, 119, 100, 171, 117, 194, 148, 48, 185, 150, 45, 108, 140, 105, 163, 142,
        ];
        assert_eq!(out, expected);
        let out = resize(&image, (3, 9), Resample::Bilinear)?
            .flatten_all()?
            .to_vec1::<u8>()?;
        let expected = [
            6, 31, 59, 88, 117, 146, 175, 203, 228, 58, 83, 111, 140, 169, 154, 170, 151, 81, 169,
            194, 157, 105, 134, 77, 82, 110, 135, 56, 81, 109, 138, 167, 196, 225, 168, 22, 108,
            133, 161, 177, 162, 191, 140, 87, 74, 73, 98, 126, 130, 74, 103, 132, 160, 185, 106,
            131, 159, 188, 217, 160, 83, 47, 72, 158, 183, 186, 151, 70, 54, 71, 99, 124, 123, 148,
            127, 95, 124, 153, 182, 161, 89,
        ];
        assert_eq!(out, expected);
        Ok(())
    }
}
