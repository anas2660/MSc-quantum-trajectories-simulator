use std::{
    arch::asm,
    arch::x86_64::__m256,
    fmt::Display,
    iter::Sum,
    mem::{MaybeUninit, swap},
    ops::{Add, AddAssign, Mul, MulAssign, Sub, SubAssign, Div, DivAssign, Index},
    simd::f32x8,
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
            use nalgebra::*;
            let tmp = nalgebra::Complex::new(result.real.as_array()[lane], result.imag.as_array()[lane]);
            let a = tmp.exp();
            result.real[lane] = a.re;
            result.imag[lane] = a.im;
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
        let denum = rhs.real*rhs.real + rhs.imag*rhs.imag;
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
        let denum = rhs.real*rhs.real + rhs.imag*rhs.imag;
        Complex {
            real:  V::splat(self) * rhs.real / denum,
            imag: -V::splat(self) * rhs.imag / denum,
        }
    }
}

impl Div<Complex> for f32 {
    type Output = Complex;
    #[inline]
    fn div(self, rhs: Complex) -> Self::Output {
        self/&rhs
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

#[derive(Debug, Clone, Copy)]
pub struct Operator {
    elements: [[Complex; 2]; 2],
}

impl Operator {
    const SIZE: usize = 2;
    pub fn identity() -> Self {
        let mut result: MaybeUninit<Operator> = std::mem::MaybeUninit::uninit();
        for y in 0..Operator::SIZE {
            for x in 0..Operator::SIZE {
                unsafe { (*result.as_mut_ptr()).elements[y][x] = C!(((x == y) as u32) as f32) };
            }
        }

        unsafe { result.assume_init() }
    }

    pub const fn new(v00: Complex, v01: Complex, v10: Complex, v11: Complex) -> Self {
        Operator {
            elements: [[v00, v01], [v10, v11]],
        }
    }

    pub fn from_fn<F: Fn(usize, usize) -> Complex >(f: F) -> Self {
        let mut result: MaybeUninit<Operator> = std::mem::MaybeUninit::uninit();
        for y in 0..Operator::SIZE {
            for x in 0..Operator::SIZE {
                unsafe { (*result.as_mut_ptr()).elements[y][x] = f(y,x) };
            }
        }
        unsafe { result.assume_init() }
    }


    pub fn from_partial_diagonal(diag: &[Complex]) -> Self {
        assert_eq!(diag.len(), Operator::SIZE);
        Operator::new(diag[0], Complex::from(0.), Complex::from(0.), diag[1])
    }

    #[inline]
    pub fn transpose(&self) -> Self {
        let mut result: MaybeUninit<Operator> = std::mem::MaybeUninit::uninit();
        for y in 0..Operator::SIZE {
            for x in 0..Operator::SIZE {
                unsafe { (*result.as_mut_ptr()).elements[y][x] = self.elements[x][y]; }
            }
        }
        unsafe { result.assume_init() }
    }

    #[inline]
    pub fn conjugate(&self) -> Self {
        let mut result = self.clone();
        for y in 0..Operator::SIZE {
            for x in 0..Operator::SIZE {
                result.elements[y][x] = result.elements[y][x].conjugate()
            }
        }
        result
    }

    pub fn dagger(&self) -> Self {
        self.conjugate().transpose()
    }

    #[inline]
    pub fn adjoint(&self) -> Self {
        self.dagger()
    }

    #[inline]
    pub fn scale(&self, rhs: f32) -> Self {
        self * rhs
    }

    #[inline]
    pub fn trace(&self) -> Complex {
        let mut result = Complex::from(0.0);
        for i in 0..Operator::SIZE {
            result += self.elements[i][i]
        }
        result
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

impl Index<(usize, usize)> for Operator {
    type Output = Complex;

    #[inline]
    fn index(&self, index: (usize, usize)) -> &Self::Output {
        &self.elements[index.0][index.1]
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
impl Mul<Operator> for f32 {
    type Output = Operator;
    #[inline]
    fn mul(self, rhs: Operator) -> Self::Output {
        &rhs * self
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

impl Mul<&Complex> for &Operator {
    type Output = Operator;

    #[inline]
    fn mul(self, rhs: &Complex) -> Self::Output {
        let mut result: Operator = self.clone();
        for y in 0..Operator::SIZE {
            for x in 0..Operator::SIZE {
                result.elements[y][x] *= rhs;
            }
        }
        result
    }
}

impl Mul<&Operator> for Complex {
    type Output = Operator;

    #[inline]
    fn mul(self, rhs: &Operator) -> Self::Output {
        rhs * &self
    }
}


impl Mul<Operator> for Complex {
    type Output = Operator;
    #[inline]
    fn mul(self, rhs: Operator) -> Self::Output {
        &rhs * &self
    }
}


impl Mul<Complex> for Operator {
    type Output = Operator;
    #[inline]
    fn mul(self, rhs: Complex) -> Self::Output {
        &self * &rhs
    }
}




impl Add<&Operator> for &Operator {
    type Output = Operator;

    #[inline]
    fn add(self, rhs: &Operator) -> Self::Output {
        let mut result: Operator = self.clone();
        for y in 0..Operator::SIZE {
            for x in 0..Operator::SIZE {
                result.elements[y][x] += rhs.elements[y][x];
            }
        }
        result
    }
}

impl Add<Operator> for Operator {
    type Output = Operator;
    #[inline]
    fn add(self, rhs: Operator) -> Self::Output {
        &self + &rhs
    }
}

impl AddAssign for Operator {
    #[inline]
    fn add_assign(&mut self, rhs: Self) {
        *self = (&(*self)) + &rhs;
    }
}


impl Sub<&Operator> for &Operator {
    type Output = Operator;

    #[inline]
    fn sub(self, rhs: &Operator) -> Self::Output {
        let mut result: Operator = self.clone();
        for y in 0..Operator::SIZE {
            for x in 0..Operator::SIZE {
                result.elements[y][x] -= rhs.elements[y][x];
            }
        }
        result
    }
}


impl Sub<Operator> for Operator {
    type Output = Operator;
    #[inline]
    fn sub(self, rhs: Operator) -> Self::Output {
        &self - &rhs
    }
}


impl Div<&Complex> for &Operator {
    type Output = Operator;
    fn div(self, rhs: &Complex) -> Self::Output {
        let mut result: Operator = self.clone();
        for y in 0..Operator::SIZE {
            for x in 0..Operator::SIZE {
                result.elements[y][x] = &result.elements[y][x] / rhs;
            }
        }
        result
    }
}

impl Div<Complex> for Operator {
    type Output = Operator;
    #[inline]
    fn div(self, rhs: Complex) -> Self::Output {
       &self / &rhs
    }
}
