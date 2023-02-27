use std::{
    fmt::Display,
    ops::{Add, Index, IndexMut, Mul, Sub},
};

use crate::num::*;
use crate::operator::*;

#[derive(Clone)]
pub struct Matrix {
    v: Vec<Vec<Complex>>,
}

impl Matrix {
    pub fn new(v00: fp, v01: fp, v10: fp, v11: fp) -> Matrix {
        Matrix {
            v: [
                [Complex::new(v00, 0.0), Complex::new(v01, 0.0)].to_vec(),
                [Complex::new(v10, 0.0), Complex::new(v11, 0.0)].to_vec(),
            ]
            .to_vec(),
        }
    }

    pub fn zeroed(rows: usize, columns: usize) -> Matrix {
        Matrix {
            v: vec![vec![C!(0.0); columns]; rows],
        }
    }


    pub fn identity(size: usize) -> Matrix {
        //let mut result = Matrix::zeros(size);
        Matrix {
            v: (0..size)
                .map(|y| (0..size).map(|x| C!(((x == y) as u32) as fp)).collect())
                .collect(),
        }
    }

    pub fn from_partial_diagonal(diag: &[Complex]) -> Self {
        let size = diag.len();
        let mut result = Matrix::zeroed(size, size);
        for i in 0..size {
            result.v[i][i] = diag[i];
        }
        result
    }

    pub fn vector(elements: &[fp]) -> Self {
        let size = elements.len();
        let mut result = Matrix::zeroed(size, 1);
        for i in 0..size {
            result.v[i][0] = elements[i].into();
        }
        result
    }

    pub fn kronecker(&self, rhs: &Matrix) -> Matrix {
        let size = self.v.len() * rhs.v.len();
        let mut result = Matrix::zeroed(size, size);
        let mut index = (0, 0);

        for j1 in 0..self.v[0].len() {
            for j2 in 0..rhs.v[0].len() {
                for i1 in 0..self.v.len() {
                    let coeff = self[(i1, j1)];

                    for i2 in 0..rhs.v.len() {
                        result[index] = coeff * rhs[(i2, j2)];

                        index.0 += 1;
                        if index.0 == size {
                            index.0 -= size;
                            index.1 += 1;
                        }
                    }
                }
            }
        }
        result
    }

    #[inline]
    pub fn to_operator(&self) -> Operator {
        assert!(Operator::SIZE == self.v.len());
        assert!(Operator::SIZE == self.v[0].len());
        Operator::new(
            self.v[0].as_slice(),
            self.v[1].as_slice(),
            self.v[2].as_slice(),
            self.v[3].as_slice(),
        )
    }

    pub fn transpose(&self) -> Self {
        let mut result = Matrix::zeroed(self.v[0].len(), self.v.len());
        for y in 0..self.v.len() {
            for x in 0..self.v[0].len() {
                result.v[x][y] = self.v[y][x];
            }
        }
        result
    }

    pub fn conjugate(&self) -> Self {
        let mut result = self.clone();
        for y in 0..self.v.len() {
            for x in 0..self.v[0].len() {
                result.v[y][x] = result.v[y][x].conjugate();
            }
        }
        result
    }

    pub fn dagger(&self) -> Self {
        self.transpose().conjugate()
    }
}

impl Display for Matrix {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        for row in self.v.iter() {
            f.write_str(" ")?;
            for element in row.iter() {
                f.write_fmt(format_args!("{}", element))?;
            }
            f.write_str("\n")?;
        }
        Ok(())
    }
}

impl Index<(usize, usize)> for Matrix {
    type Output = Complex;

    #[inline]
    fn index(&self, index: (usize, usize)) -> &Self::Output {
        &self.v[index.0][index.1]
    }
}

impl IndexMut<(usize, usize)> for Matrix {
    fn index_mut(&mut self, index: (usize, usize)) -> &mut Self::Output {
        &mut self.v[index.0][index.1]
    }
}

impl Mul<&Matrix> for &Matrix {
    type Output = Matrix;

    #[inline]
    fn mul(self, rhs: &Matrix) -> Self::Output {
        assert!(self.v[0].len() == rhs.v.len());
        let mut result = Matrix::zeroed(self.v.len(), rhs.v[0].len());
        for y in 0..self.v.len() {
            for x in 0..rhs.v[0].len() {
                for id in 0..self.v[0].len() {
                    result.v[y][x] += &self.v[y][id] * &rhs.v[id][x];
                }
            }
        }
        result
    }
}

impl Mul<Matrix> for Matrix {
    type Output = Matrix;
    #[inline]
    fn mul(self, rhs: Matrix) -> Self::Output {
        &self * &rhs
    }
}
impl Mul<&Matrix> for Matrix {
    type Output = Matrix;
    #[inline]
    fn mul(self, rhs: &Matrix) -> Self::Output {
        &self * rhs
    }
}
impl Mul<Matrix> for &Matrix {
    type Output = Matrix;
    #[inline]
    fn mul(self, rhs: Matrix) -> Self::Output {
        self * &rhs
    }
}

impl Mul<fp> for &Matrix {
    type Output = Matrix;
    #[inline]
    fn mul(self, rhs: fp) -> Self::Output {
        self * &C!(rhs)
    }
}
impl Mul<&Complex> for &Matrix {
    type Output = Matrix;
    #[inline]
    fn mul(self, rhs: &Complex) -> Self::Output {
        let mut result = self.clone();
        for row in result.v.iter_mut() {
            for element in row.iter_mut() {
                *element *= rhs;
            }
        }
        result
    }
}
impl Mul<Complex> for Matrix {
    type Output = Matrix;
    #[inline]
    fn mul(self, rhs: Complex) -> Self::Output {
        &self * &rhs
    }
}
impl Mul<&Matrix> for Complex {
    type Output = Matrix;
    #[inline]
    fn mul(self, rhs: &Matrix) -> Self::Output {
        rhs * &self
    }
}
impl Mul<Matrix> for Complex {
    type Output = Matrix;
    #[inline]
    fn mul(self, rhs: Matrix) -> Self::Output {
        &rhs * &self
    }
}
impl Mul<fp> for Matrix {
    type Output = Matrix;
    #[inline]
    fn mul(self, rhs: fp) -> Self::Output {
        &self * rhs
    }
}

impl Mul<Matrix> for fp {
    type Output = Matrix;
    #[inline]
    fn mul(self, rhs: Matrix) -> Self::Output {
        &rhs * self
    }
}

impl Mul<&Matrix> for fp {
    type Output = Matrix;
    #[inline]
    fn mul(self, rhs: &Matrix) -> Self::Output {
        rhs * self
    }
}

impl Add<&Matrix> for &Matrix {
    type Output = Matrix;

    #[inline]
    fn add(self, rhs: &Matrix) -> Self::Output {
        assert!(self.v.len() == rhs.v.len());
        assert!(self.v[0].len() == rhs.v[0].len());
        let mut result = self.clone();
        for y in 0..self.v.len() {
            for x in 0..self.v[0].len() {
                result.v[y][x] += rhs.v[y][x];
            }
        }
        result
    }
}

impl Add<Matrix> for Matrix {
    type Output = Matrix;
    #[inline]
    fn add(self, rhs: Matrix) -> Self::Output {
        &self + &rhs
    }
}

impl Sub<&Matrix> for &Matrix {
    type Output = Matrix;

    #[inline]
    fn sub(self, rhs: &Matrix) -> Self::Output {
        assert!(self.v.len() == rhs.v.len());
        assert!(self.v[0].len() == rhs.v[0].len());
        let mut result: Matrix = self.clone();
        for y in 0..self.v.len() {
            for x in 0..self.v[0].len() {
                result.v[y][x] -= rhs.v[y][x];
            }
        }
        result
    }
}

impl Sub<Matrix> for Matrix {
    type Output = Matrix;
    #[inline]
    fn sub(self, rhs: Matrix) -> Self::Output {
        &self - &rhs
    }
}
