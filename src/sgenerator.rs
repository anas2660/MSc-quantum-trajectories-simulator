use rand::{rngs::ThreadRng, Rng};

use crate::num::SV32;

pub struct SGenerator {
    v: SV32,
    counter: u8
}

// Shift right once
#[inline(always)]
fn shift_right(v: SV32) -> SV32 {
    // Shifts in zeros
    #[cfg(not(feature="double-precision"))]
    return unsafe {std::arch::x86_64::_mm256_srli_epi32::<1>(v.into())}.into();
    #[cfg(feature="double-precision")]
    return unsafe {std::arch::x86_64::_mm_srli_epi32::<1>(v.into())}.into();
}

// Multiply SV32 by 2... but actually shift left once... but actually compiles to adding itself to itself.
#[inline(always)]
fn mul2(v: SV32) -> SV32 {
    // Shifts in zeros
    #[cfg(not(feature="double-precision"))]
    return unsafe {std::arch::x86_64::_mm256_slli_epi32::<1>(v.into())}.into();
    #[cfg(feature="double-precision")]
    return unsafe {std::arch::x86_64::_mm_slli_epi32::<1>(v.into())}.into();
}


impl SGenerator {
    #[inline]
    fn regen(&mut self, rng: &mut ThreadRng) {
        rng.fill(&mut self.v.as_mut_array()[..]);

        // Faster when gen() is not inlined for some reason
        //for x in self.v.as_mut_array().iter_mut() {
        //    *x = rng.gen();
        //}

        self.counter = 0;
    }

    pub fn new(rng: &mut ThreadRng) -> Self {
        let mut g = Self {
            v: SV32::splat(0),
            counter: 0
        };
        g.regen(rng);
        g
    }


    #[inline(always)]
    pub fn gen(&mut self, rng: &mut ThreadRng) -> SV32 {
        const ONE: SV32 = SV32::from_array([1; SV32::LEN]);

        let result = mul2(self.v & ONE) - ONE; // Transforms rand -> -1 or 1

        self.counter += 1;
        if self.counter <= 32 { // "likely()" won't be stabilized?
            self.v = shift_right(self.v);
        } else {
            self.regen(rng);
        }
        result
    }
}
