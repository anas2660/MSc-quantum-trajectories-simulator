use crate::num::*;

use std::simd::{u32x8, SimdOrd};

// TODO: Control bounds
#[derive(Clone, Copy, Debug)]
pub struct Histogram<const BIN_COUNT: usize> {
    pub bins: [u32; BIN_COUNT],
}

impl<const BIN_COUNT: usize> Histogram<BIN_COUNT> {
    pub fn new() -> Histogram<BIN_COUNT> {
        Histogram {
            bins: [0u32; BIN_COUNT],
        }
    }

    pub fn add_values(&mut self, v: &Real) {
        let indices: u32x8 = unsafe { (v * Real::splat(BIN_COUNT as f32)).to_int_unchecked() }
            .simd_clamp(u32x8::splat(0), u32x8::splat(BIN_COUNT as u32 - 1));

        for index in indices.as_array() {
            self.bins[*index as usize] += 1;
        }
    }

    pub fn add_histogram(&mut self, other: &Histogram<BIN_COUNT>) {
        for (s, t) in self.bins.iter_mut().zip(other.bins.iter()) {
            *s += t;
        }
    }
}
