use std::{ops::{Index, IndexMut}, fmt::Display};

use crate::num::*;

pub struct Matrix {
    v: Vec<Vec<Complex>>,
}

impl Matrix {
    pub fn new(v00: f32, v01: f32, v10: f32, v11: f32) -> Matrix {
        Matrix {
            v: [[C!(v00), C!(v01)].to_vec(), [C!(v10), C!(v11)].to_vec()].to_vec()
        }
    }

    pub fn zeros(size: usize) -> Matrix {
        Matrix {
            v: vec![vec![C!(0.0); size]; size],
        }
    }

    pub fn identity(size: usize) -> Matrix {
        //let mut result = Matrix::zeros(size);
        Matrix {
            v: (0..size)
                .map(|y| (0..size).map(|x| C!(((x == y) as u32) as f32)).collect())
                .collect(),
        }
    }

    pub fn from_partial_diagonal(diag: &[Complex]) -> Self {
        let size = diag.len();
        let mut result = Matrix::zeros(size);
        for i in 0..size {
            result.v[i][i] = diag[i];
        }
        result
    }

    pub fn kronecker(&self, rhs: &Matrix) -> Matrix {
        let size = self.v.len()*rhs.v.len();
        let mut result = Matrix::zeros(size);
        let mut index = (0, 0);

        for j1 in 0..self.v[0].len() {
            for j2 in 0..rhs.v[0].len() {
                for i1 in 0..self.v.len() {
                    let coeff = self[((i1, j1))] ;

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

impl IndexMut<(usize, usize)> for Matrix  {
    fn index_mut(&mut self, index: (usize, usize)) -> &mut Self::Output {
        &mut self.v[index.0][index.1]
    }
}
