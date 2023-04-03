use std::{
    fmt::Display,
    ops::{Add, Index, IndexMut, Mul, Sub, Neg},
};

use crate::num::*;
use crate::operator::*;

#[derive(Clone)]
pub struct Matrix {
    v: Vec<Vec<ShallowComplex>>,
}

impl Matrix {
    pub fn new(v00: f64, v01: f64, v10: f64, v11: f64) -> Matrix {
        Matrix {
            v: [
                [SC!(v00), SC!(v01)].to_vec(),
                [SC!(v10), SC!(v11)].to_vec(),
            ]
            .to_vec(),
        }
    }

    pub fn zeroed(rows: usize, columns: usize) -> Matrix {
        Matrix {
            v: vec![vec![SC!(0.0); columns]; rows],
        }
    }


    pub fn identity(size: usize) -> Matrix {
        //let mut result = Matrix::zeros(size);
        Matrix {
            v: (0..size)
                .map(|y| (0..size).map(|x| SC!(((x == y) as u32) as f64)).collect())
                .collect(),
        }
    }

    // pub fn from_partial_diagonal(diag: &[Complex]) -> Self {
    //     let size = diag.len();
    //     let mut result = Matrix::zeroed(size, size);
    //     for i in 0..size {
    //         result.v[i][i] = diag[i];
    //     }
    //     result
    // }

    pub fn vector(elements: &[f64]) -> Self {
        let size = elements.len();
        let mut result = Matrix::zeroed(size, 1);
        for i in 0..size {
            result.v[i][0] = SC!(elements[i]);
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
                        result[index] = &coeff * &rhs[(i2, j2)];

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
        let mut M = self.clone();

        if M.v.len() != Operator::SIZE && M.v[0].len() != Operator::SIZE {
            // Add extra col
            for row in M.v.iter_mut() {
                row.push(SC!(0.0));
            }

            // Add extra row
            M.v.push([SC!(0.0); Operator::SIZE].to_vec());
        }

        assert_eq!(Operator::SIZE, M.v.len());
        assert!(Operator::SIZE == M.v.len());
        assert!(Operator::SIZE == M.v[0].len());
        let mut result = Operator::zero();
        for (row_id, row) in result.elements.iter_mut().enumerate() {
            for (column, element) in row.iter_mut().enumerate() {
                *element = C!(M.v[row_id][column].real as fp, M.v[row_id][column].imag as fp);
            }
        }
        result
        //Operator::new(
        //    self.v[0].as_slice(),
        //    self.v[1].as_slice(),
        //    self.v[2].as_slice(),
        //    self.v[3].as_slice(),
        //)
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
                result.v[y][x] = SC!(result.v[y][x].imag, result.v[y][x].real);
            }
        }
        result
    }

    pub fn dagger(&self) -> Self {
        self.transpose().conjugate()
    }


    pub fn max_abs(&self) -> (f64, f64) {
        let (mut max_real, mut max_imag) = (f64::MIN, f64::MIN);
        for i in 0..Operator::SIZE {
            for j in 0..Operator::SIZE {
                if self[(i,j)].real.abs() > max_real {
                    max_real = self[(i,j)].real.abs();
                }
                if self[(i,j)].imag.abs() > max_imag {
                    max_imag = self[(i,j)].imag.abs();
                }
            }
        }
        (max_real, max_imag)
    }

    pub fn budget_matrix_exponential(&self) -> Result<Matrix, f64> {
        let mut result = Matrix::identity(Operator::SIZE);
        let mut tmp = Matrix::identity(Operator::SIZE);

        for i in 1..16 {
            tmp = Matrix::identity(Operator::SIZE);
            for n in 1..=i {
                tmp = tmp * ((1.0/(n as f64)) * self);
            }
            result = &result + &tmp;
        }

        // Convergence check
        let max = tmp.max_abs();
        let max = max.0.max(max.1);
        if max > 1.0e-30 || result[(0,0)].real.is_nan() { Err(max) } else { Ok(result) }
    }


    pub fn pow(&self, n: u32) -> Matrix {
        if n == 0 { return Matrix::identity(Operator::SIZE); }
        if n == 1 { return self.clone(); }

        let half = n/2;
        let part = self.pow(half);
        &part * &part * (0..(n-2*half)).fold(Matrix::identity(Operator::SIZE), |p, _| p*self)
    }

}

impl Display for Matrix {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        for row in self.v.iter() {
            f.write_str(" ")?;
            for element in row.iter() {
                f.write_fmt(format_args!("{}+{}i", element.real, element.imag))?;
            }
            f.write_str("\n")?;
        }
        Ok(())
    }
}

impl Index<(usize, usize)> for Matrix {
    type Output = ShallowComplex;

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
                    result.v[y][x] = result.v[y][x] + &self.v[y][id] * &rhs.v[id][x];
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

impl Mul<f64> for &Matrix {
    type Output = Matrix;
    #[inline]
    fn mul(self, rhs: f64) -> Self::Output {
        self * &SC!(rhs)
    }
}
impl Mul<&ShallowComplex> for &Matrix {
    type Output = Matrix;
    #[inline]
    fn mul(self, rhs: &ShallowComplex) -> Self::Output {
        let mut result = self.clone();
        for row in result.v.iter_mut() {
            for element in row.iter_mut() {
                *element = &*element * rhs;
            }
        }
        result
    }
}
impl Mul<ShallowComplex> for Matrix {
    type Output = Matrix;
    #[inline]
    fn mul(self, rhs: ShallowComplex) -> Self::Output {
        &self * &rhs
    }
}
impl Mul<&Matrix> for ShallowComplex {
    type Output = Matrix;
    #[inline]
    fn mul(self, rhs: &Matrix) -> Self::Output {
        rhs * &self
    }
}
impl Mul<Matrix> for ShallowComplex {
    type Output = Matrix;
    #[inline]
    fn mul(self, rhs: Matrix) -> Self::Output {
        &rhs * &self
    }
}
impl Mul<f64> for Matrix {
    type Output = Matrix;
    #[inline]
    fn mul(self, rhs: f64) -> Self::Output {
        &self * rhs
    }
}

impl Mul<Matrix> for f64 {
    type Output = Matrix;
    #[inline]
    fn mul(self, rhs: Matrix) -> Self::Output {
        &rhs * self
    }
}

impl Mul<&Matrix> for f64 {
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
                result.v[y][x] = result.v[y][x] + rhs.v[y][x];
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
                result.v[y][x] = result.v[y][x] + SC!(-rhs.v[y][x].real, -rhs.v[y][x].imag);
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

impl Neg for Matrix {
    type Output = Matrix;

    fn neg(mut self) -> Self::Output {
        for row in self.v.iter_mut() {
            for element in row.iter_mut() {
                *element = SC!(-element.real, -element.imag);
            }
        }
        self
    }
}

impl From<Operator> for Matrix {
    fn from(value: Operator) -> Self {
        let mut result = Matrix::zeroed(Operator::SIZE, Operator::SIZE);
        for i in 0..Operator::SIZE {
            for j in 0..Operator::SIZE {
                let (real, imag) = value[(i,j)].first();
                result[(i,j)] = SC!(real as f64, imag as f64);
            }
        }

        result
    }
}
