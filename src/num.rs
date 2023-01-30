#[allow(unused_imports)]
use std::{
    arch::asm,
    arch::x86_64::__m256,
    fmt::Display,
    iter::Sum,
    mem::MaybeUninit,
    ops::{Add, AddAssign, Div, Mul, MulAssign, Sub, SubAssign},
    simd::f32x8,
};

pub type V = f32x8;
pub type Real = V;

#[derive(Debug, Clone, Copy)]
pub struct Complex {
    pub real: Real,
    pub imag: Real,
}

//pub const ZERO: Complex = C!(0.0);

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
pub(crate) use C;

impl Complex {
    //    fn from_real(f32) -> Complex {}
    #[inline]
    pub const fn new(re: f32, im: f32) -> Self {
        Complex {
            real: V::from_array([re; V::LANES]),
            imag: V::from_array([im; V::LANES]),
        }
    }

    #[inline]
    pub fn conjugate(&self) -> Self {
        Complex {
            real: self.real,
            imag: -self.imag,
        }
    }

    #[inline]
    pub fn exp(&self) -> Self {
        let mut result = self.clone();
        for lane in 0..V::LANES {
            // formula: e^(a + bi) = e^a (cos(b) + i*sin(b))
            let real = result.real.as_array()[lane];
            let imag = result.imag.as_array()[lane];
            let realexp = real.exp();
            let result_real = realexp * imag.cos();
            let result_imag = realexp * imag.sin();
            result.real[lane] = result_real;
            result.imag[lane] = result_imag;
        }
        result
    }

    #[inline]
    pub fn real(&self) -> Real {
        self.real
    }

    #[inline]
    pub fn imag(&self) -> Real {
        self.imag
    }

    #[inline]
    pub fn first(&self) -> (f32, f32) {
        (self.real.as_array()[0], self.imag.as_array()[0])
    }

    #[inline]
    pub fn scale<'a, 'b, T>(&'a mut self, scalar: T) -> &'a Self
    where T: Mul<&'b Self, Output = Self>
    {
        unsafe { // cant get this to understand that this should be fine
            *self = scalar * &*(self as *mut Self);
        }
        self
    }

}

impl Display for Complex {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let reals = self.real.as_array();
        let imags = self.imag.as_array();

        if imags[0] < 0.0 {
            f.write_fmt(format_args!("{}{}i, ", reals[0], imags[0]))?;
        } else {
            f.write_fmt(format_args!("{}+{}i, ", reals[0], imags[0]))?;
        }
        return Ok(());
        //f.write_str("[")?;
        //for (re, im) in reals.iter().zip(imags.iter()) {
        //    if *im < 0.0 {
        //        f.write_fmt(format_args!("{}{}i, ", re, im))?;
        //    } else {
        //        f.write_fmt(format_args!("{}+{}i, ", re, im))?;
        //    }
        //}
        //f.write_str("]")?;
        //Ok(())
    }
}

impl From<f32> for Complex {
    #[inline]
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

    #[inline]
    fn add(self, rhs: &Complex) -> Self::Output {
        Complex {
            real: self.real + rhs.real,
            imag: self.imag + rhs.imag,
        }
    }
}

impl Add for Complex {
    type Output = Complex;

    #[inline]
    fn add(self, rhs: Complex) -> Self::Output {
        &self + &rhs
    }
}

impl Sub<&Complex> for &Complex {
    type Output = Complex;

    #[inline]
    fn sub(self, rhs: &Complex) -> Self::Output {
        Complex {
            real: self.real - rhs.real,
            imag: self.imag - rhs.imag,
        }
    }
}

impl Sub<f32> for &Complex {
    type Output = Complex;

    #[inline]
    fn sub(self, rhs: f32) -> Self::Output {
        Complex {
            real: self.real - V::splat(rhs),
            imag: self.imag,
        }
    }
}

impl Sub<Complex> for Complex {
    type Output = Complex;

    #[inline]
    fn sub(self, rhs: Complex) -> Self::Output {
        &self - &rhs
    }
}

impl Add<f32> for &Complex {
    type Output = Complex;

    #[inline]
    fn add(self, rhs: f32) -> Self::Output {
        Complex {
            real: self.real + V::splat(rhs),
            imag: self.imag,
        }
    }
}

impl Add<Complex> for f32 {
    type Output = Complex;

    #[inline]
    fn add(self, rhs: Complex) -> Self::Output {
        &rhs + self
    }
}

impl Sub<&Complex> for f32 {
    type Output = Complex;

    #[inline]
    fn sub(self, rhs: &Complex) -> Self::Output {
        Complex {
            real: V::splat(self) - rhs.real,
            imag: -rhs.imag,
        }
    }
}

impl Sub<Complex> for f32 {
    type Output = Complex;

    #[inline]
    fn sub(self, rhs: Complex) -> Self::Output {
        self - &rhs
    }
}

impl Div<&Complex> for &Complex {
    type Output = Complex;

    #[inline]
    fn div(self, rhs: &Complex) -> Self::Output {
        let denum = rhs.real * rhs.real + rhs.imag * rhs.imag;
        Complex {
            real: (self.real * rhs.real + self.imag * rhs.imag) / denum,
            imag: (self.imag * rhs.real - self.real * rhs.imag) / denum,
        }
    }
}

impl Div<Complex> for Complex {
    type Output = Complex;
    #[inline]
    fn div(self, rhs: Complex) -> Self::Output {
        &self / &rhs
    }
}

impl Div<&Complex> for f32 {
    type Output = Complex;

    #[inline]
    fn div(self, rhs: &Complex) -> Self::Output {
        let denum = rhs.real * rhs.real + rhs.imag * rhs.imag;
        Complex {
            real: V::splat(self) * rhs.real / denum,
            imag: -V::splat(self) * rhs.imag / denum,
        }
    }
}

impl Div<Complex> for f32 {
    type Output = Complex;
    #[inline]
    fn div(self, rhs: Complex) -> Self::Output {
        self / &rhs
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

    #[inline]
    fn mul(self, rhs: &V) -> Self::Output {
        Complex {
            real: self.real * rhs,
            imag: self.imag * rhs,
        }
    }
}

impl Mul<f32> for &Complex {
    type Output = Complex;

    #[inline]
    fn mul(self, rhs: f32) -> Self::Output {
        self * &V::splat(rhs)
    }
}

impl Mul<f32> for Complex {
    type Output = Complex;

    #[inline]
    fn mul(self, rhs: f32) -> Self::Output {
        &self * &V::splat(rhs)
    }
}

impl Mul<Complex> for f32 {
    type Output = Complex;

    #[inline]
    fn mul(self, rhs: Complex) -> Self::Output {
        &rhs * &V::splat(self)
    }
}

impl Mul<&Complex> for f32 {
    type Output = Complex;

    #[inline]
    fn mul(self, rhs: &Complex) -> Self::Output {
        rhs * &V::splat(self)
    }
}

impl Mul<&Complex> for &Real {
    type Output = Complex;
    #[inline]
    fn mul(self, rhs: &Complex) -> Self::Output {
        rhs * self
    }
}

impl Mul<&Complex> for Real {
    type Output = Complex;
    #[inline]
    fn mul(self, rhs: &Complex) -> Self::Output {
        Complex {
            real: rhs.real * self,
            imag: rhs.imag * self
        }
    }
}


impl MulAssign<&Complex> for Complex {
    #[inline]
    fn mul_assign(&mut self, rhs: &Complex) {
        *self = &(*self) * rhs;
    }
}

impl MulAssign<&V> for Complex {
    #[inline]
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

impl SubAssign<&Complex> for Complex {
    #[inline]
    fn sub_assign(&mut self, rhs: &Complex) {
        *self = &(*self) - rhs;
    }
}

impl SubAssign<Complex> for Complex {
    #[inline]
    fn sub_assign(&mut self, rhs: Complex) {
        *self = &(*self) - &rhs;
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

impl Mul for Complex {
    type Output = Complex;
    fn mul(self, rhs: Self) -> Self::Output {
        &self * &rhs
    }
}
