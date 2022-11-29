use std::{mem::MaybeUninit, fmt::Display, ops::{Index, Mul, Add, AddAssign, Sub, Div}};

use crate::num::*;

#[derive(Debug, Clone, Copy)]
pub struct Operator {
    elements: [[Complex; 2]; 2],
}

impl Operator {
    pub const SIZE: usize = 2;
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
