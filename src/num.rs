use std::{
    fmt::Display,
    iter::Sum,
    ops::{Add, AddAssign, Mul, MulAssign},
    simd::f32x8,
    arch::asm,
    arch::x86_64::__m256, mem::MaybeUninit
};


pub type V = f32x8;
pub type Real = V;

#[derive(Debug, Clone, Copy)]
pub struct Complex {
    real: Real,
    imag: Real,
}

macro_rules! C {
    ($real: expr) => {
        Complex {
            real: Real::splat($real),
            imag: Real::splat(0.0),
        }
    };
    ($real: expr, $imag: expr) => {
        Complex {
            real: Real::splat($real),
            imag: Real::splat($imag),
        }
    };
}

//pub const ZERO: Complex = C!(0.0);

impl Complex {
    //    fn from_real(f32) -> Complex {}
}

impl Display for Complex {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let reals = self.real.as_array();
        let imags = self.imag.as_array();
        f.write_str("[")?;
        for (re, im) in reals.iter().zip(imags.iter()) {
            if *im < 0.0 {
                f.write_fmt(format_args!("{}{}i, ", re, im))?;
            } else {
                f.write_fmt(format_args!("{}+{}i, ", re, im))?;
            }
        }
        f.write_str("]")?;
        Ok(())
    }
}

impl From<f32> for Complex {
    fn from(value: f32) -> Self {
        //Complex {
        //        real: Real::splat(value),
        //    imag: Real::splat(0.0),
        //}
        C!(value)
    }
}

impl Add<&Complex> for &Complex {
    type Output = Complex;

    fn add(self, rhs: &Complex) -> Self::Output {
        Complex {
            real: self.real + rhs.real,
            imag: self.imag + rhs.imag,
        }
    }
}

impl Add<f32> for &Complex {
    type Output = Complex;

    fn add(self, rhs: f32) -> Self::Output {
        Complex {
            real: self.real + V::splat(rhs),
            imag: self.imag,
        }
    }
}



impl Mul<&Complex> for &Complex {
    type Output = Complex;

    #[inline]
    fn mul(self, rhs: &Complex) -> Self::Output {
        return Complex {
            real: self.real * rhs.real - self.imag * rhs.imag,
            imag: self.real * rhs.imag + self.imag * rhs.real,
        };


        // FMA implementation (slower for some reason)
        /*
        unsafe {
            let mut real: __m256;
            let mut imag: __m256;

            asm!(
                "vmulps  {real}, {rhsimag}, {selfimag}",
                "vmulps  {imag}, {rhsreal}, {selfimag}",
                "vfmsub231ps {real}, {rhsreal}, {selfreal}",
                "vfmadd231ps {imag}, {selfreal}, {rhsimag}",
                real = out(ymm_reg) real,
                imag = out(ymm_reg) imag,
                rhsreal = in(ymm_reg) Into::<__m256>::into(rhs.real),
                rhsimag = in(ymm_reg) Into::<__m256>::into(rhs.imag),
                selfreal = in(ymm_reg) Into::<__m256>::into(self.real),
                selfimag = in(ymm_reg) Into::<__m256>::into(self.imag),
            );

            Complex { real: real.into(), imag: imag.into() }
        }
        */

    }
}

impl Mul<&V> for &Complex {
    type Output = Complex;

    fn mul(self, rhs: &V) -> Self::Output {
        Complex {
            real: self.real * rhs,
            imag: self.imag * rhs,
        }
    }
}

impl MulAssign<&Complex> for Complex {
    fn mul_assign(&mut self, rhs: &Complex) {
        *self = &(*self) * rhs;
    }
}

impl MulAssign<&V> for Complex {
    fn mul_assign(&mut self, rhs: &V) {
        *self = &(*self) * rhs;
    }
}

impl AddAssign<&Complex> for Complex {
    #[inline]
    fn add_assign(&mut self, rhs: &Complex) {
        *self = &(*self) + rhs;
    }
}

impl AddAssign<Complex> for Complex {
    #[inline]
    fn add_assign(&mut self, rhs: Complex) {
        *self = &(*self) + &rhs;
    }
}

impl Sum for Complex {
    fn sum<I: Iterator<Item = Complex>>(iter: I) -> Complex {
        let mut result = Complex::from(0.0);
        for c in iter {
            result += &c;
        }
        result
    }
}

#[derive(Debug, Clone, Copy)]
pub struct Operator {
    elements: [[Complex; 2]; 2],
}

impl Operator {
    const SIZE: usize = 2;
    pub fn ident() -> Self {
        let mut result: MaybeUninit<Operator> = unsafe { std::mem::MaybeUninit::uninit() };
        for y in 0..Operator::SIZE {
            for x in 0..Operator::SIZE {
                unsafe { (*result.as_mut_ptr()).elements[y][x] = C!(((x==y) as u32) as f32) };
            }
        }

        unsafe { result.assume_init() }
    }
}

impl Display for Operator {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        for row in self.elements.iter() {
            f.write_str("  ")?;
            for element in row.iter() {
                f.write_fmt(format_args!("{} ", element))?;
            }
            f.write_str("\n")?;
        }
        Ok(())
    }
}

impl Mul<&Operator> for &Operator {
    type Output = Operator;

    #[inline]
    fn mul(self, rhs: &Operator) -> Self::Output {
        let mut result: Operator = unsafe { std::mem::zeroed() };
        for y in 0..Operator::SIZE {
            for x in 0..Operator::SIZE {
                for id in 0..Operator::SIZE {
                    result.elements[y][x] += &self.elements[y][id] * &rhs.elements[id][x];
                }
            }
        }
        result
    }
}

impl Mul<Operator> for Operator {
    type Output = Operator;
    #[inline]
    fn mul(self, rhs: Operator) -> Self::Output {
        &self * &rhs
    }
}
impl Mul<&Operator> for Operator {
    type Output = Operator;
    #[inline]
    fn mul(self, rhs: &Operator) -> Self::Output {
        &self * rhs
    }
}
impl Mul<Operator> for &Operator {
    type Output = Operator;
    #[inline]
    fn mul(self, rhs: Operator) -> Self::Output {
        self * &rhs
    }
}

impl Mul for Complex {
    type Output = Complex;
    fn mul(self, rhs: Self) -> Self::Output {
        &self * &rhs
    }
}

impl Mul<f32> for &Operator {
    type Output = Operator;
    #[inline]
    fn mul(self, rhs: f32) -> Self::Output {
        let to_mul = V::splat(rhs);
        self * &to_mul
    }
}
impl Mul<f32> for Operator {
    type Output = Operator;
    #[inline]
    fn mul(self, rhs: f32) -> Self::Output {
        &self * rhs
    }
}

impl Mul<&V> for &Operator {
    type Output = Operator;
    fn mul(self, rhs: &V) -> Self::Output {
        let mut result: Operator = self.clone();
        for y in 0..Operator::SIZE {
            for x in 0..Operator::SIZE {
                result.elements[y][x] *= rhs;
            }
        }
        result
    }
}

impl Mul<&V> for Operator {
    type Output = Operator;
    #[inline]
    fn mul(self, rhs: &V) -> Self::Output {
        &self * rhs
    }
}
impl Mul<V> for Operator {
    type Output = Operator;
    #[inline]
    fn mul(self, rhs: V) -> Self::Output {
        &self * &rhs
    }
}

