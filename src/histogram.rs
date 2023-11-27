use crate::num::*;

use std::simd::SimdOrd;

// TODO: Control bounds
#[derive(Clone, Copy, Debug)]
pub struct Histogram<const BIN_COUNT: usize> {
    pub bins: [u32; BIN_COUNT],
}

impl<const BIN_COUNT: usize> Histogram<BIN_COUNT> {
    // pub fn new() -> Histogram<BIN_COUNT> {
    //     Histogram {
    //         bins: [0u32; BIN_COUNT],
    //     }
    // }

    pub fn add_values(&mut self, v: &Real) {
        let indices: SV = unsafe { std::simd::SimdFloat::to_int_unchecked(v * Real::splat(BIN_COUNT as fp)) }
            .simd_clamp(SV::splat(0), SV::splat(BIN_COUNT as Int - 1));

        for index in indices.as_array() {
            unsafe { *self.bins.get_unchecked_mut(*index as usize) += 1 };
            // self.bins[*index as usize] += 1;
        }
    }

    pub fn add_histogram(&mut self, other: &Histogram<BIN_COUNT>) {
        for (s, t) in self.bins.iter_mut().zip(other.bins.iter()) {
            *s += t;
        }
    }
}
