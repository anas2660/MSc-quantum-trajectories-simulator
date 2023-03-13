use std::{
    fmt::Display,
    mem::MaybeUninit,
    ops::{Add, AddAssign, Div, Index, Mul, Sub}, simd::SimdFloat,
};

use crate::{num::*, lindblad::Lindblad};


#[derive(Debug, Clone, Copy)]
pub struct Operator {
    pub elements: [[Complex; 4]; 4],
}

impl Operator {
    pub const QUBIT_COUNT: usize = 2;
    pub const SIZE: usize = 1 << Operator::QUBIT_COUNT;
    //pub const IDENTITY: Self = Operator::identity();

    #[inline(always)]
    pub fn zero() -> Self {
        unsafe { std::mem::MaybeUninit::zeroed().assume_init() }
    }

    #[inline(always)]
    fn uninitialized() -> Self {
        let mem: MaybeUninit<Operator> = std::mem::MaybeUninit::uninit();
        unsafe { mem.assume_init() }
    }

    pub fn identity() -> Self {
        let mut result = Operator::zero();
        for i in 0..Operator::SIZE {
            result.elements[i][i] = Complex::new(1.0, 0.0);
        }
        result
    }

    pub fn new(r0: &[Complex], r1: &[Complex], r2: &[Complex], r3: &[Complex]) -> Self {
        Operator {
            elements: [
                r0.try_into().unwrap(),
                r1.try_into().unwrap(),
                r2.try_into().unwrap(),
                r3.try_into().unwrap(),
            ],
        }
    }

    pub fn from_fn<F: Fn(usize, usize) -> Complex>(f: F) -> Self {
        let mut result = Operator::uninitialized();
        for y in 0..Operator::SIZE {
            for x in 0..Operator::SIZE {
                result.elements[y][x] = f(y, x);
            }
        }
        result
    }

    //pub fn from_partial_diagonal(diag: &[Complex]) -> Self {
    //    assert_eq!(diag.len(), Operator::SIZE);
    //    Operator::new(diag[0], Complex::from(0.), Complex::from(0.), diag[1])
    //}

    #[inline]
    pub fn transpose(&self) -> Self {
        let mut result = Operator::uninitialized();
        for y in 0..Operator::SIZE {
            for x in 0..Operator::SIZE {
                result.elements[y][x] = self.elements[x][y];

            }
        }
        result
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

    #[inline]
    pub fn dagger(&self) -> Self {
        let mut result = Operator::uninitialized();

        for y in 0..Operator::SIZE {
            for x in 0..Operator::SIZE {
                result.elements[y][x] = self.elements[x][y].conjugate();
            }
        }

        result
    }

    #[inline]
    pub fn adjoint(&self) -> Self {
        self.dagger()
    }

    #[inline]
    pub fn scale<'a, 'b, T>(&'a mut self, scalar: T) -> &'a mut Self
        where T: Mul<&'b Complex, Output = Complex> + Copy
    {
        for row in self.elements.iter_mut() {
            for element in row.iter_mut() {
                element.scale(scalar);
            }
        }
        self
    }

    #[inline]
    pub fn add(&mut self, op: &Operator) -> &mut Self {
        for row in 0..Operator::SIZE {
            for col in 0..Operator::SIZE {
                self.elements[row][col] += &op.elements[row][col];
            }
        }
        self
    }

    pub fn pow(&self, n: u32) -> Operator {
        (0..n).fold(Operator::identity(), |p, _| p*self)
    }

    #[inline]
    pub fn lindblad_term(ρ: &Operator, op: &Lindblad) -> Operator {
        op.c * ρ.mul_daggered(&op.c) - 0.5 * (op.c_dag_c * ρ + ρ * op.c_dag_c)
    }

    #[inline]
    pub fn lindblad(&mut self, ρ: &Operator, op: &Lindblad) -> &mut Operator {
        self.add(
            &(op.c * ρ.mul_daggered(&op.c)
                - 0.5 * (&op.c_dag_c * ρ + ρ * &op.c_dag_c))
        )
    }

    #[inline]
    pub fn lindblad2(&mut self, ρ: &Operator, op: &Lindblad) -> &mut Operator {
        //  2 L.L.ρ.Ld.Ld
        // -2 L.ρ.Ld.(L.Ld+cdc)
        // + cdc.cdc.ρ
        // + cdc.ρ.cdc
        // + complex_conjugate

        let c = &op.c;
        let cd = c.dagger();
        let cdc = &op.c_dag_c;
        let cρ = op.c * ρ;
        let cdag = op.c.dagger();
        let cdcd = cdag * cdag;
        let cdcρ = cdc * ρ;

        let part = (crate::Δt/800000.0)*(
            2.0 * c * cρ * cdcd
                - 2.0 * c * ρ * cd * (c * cd + *cdc)
                + cdc * cdcρ
                + cdcρ * cdc
        );

        self.add(&(part + part.conjugate()))
    }

    #[inline]
    pub fn trace(&self) -> Complex {
        let mut result = Complex::from(0.0);
        for i in 0..Operator::SIZE {
            result += self.elements[i][i]
        }
        result
    }

    #[inline]
    pub fn normalize(&mut self) -> &Self {
        let inverse_trace = self.trace().inverse();
        for i in 0..Operator::SIZE {
            for j in 0..Operator::SIZE {
                self.elements[i][j] *= &inverse_trace;
            }
        }
        self
    }

    #[inline]
    pub fn get_probabilites_simd(&self) -> StateProbabilitiesSimd {
        let mut result: StateProbabilitiesSimd = unsafe { std::mem::zeroed() };
        for i in 0..Operator::SIZE {
             result.v[i] = self[(i,i)].real()
        }
        result
    }

    #[inline]
    pub fn sub_identity(&self) -> Self {
        let mut result = *self;
        let one: Complex = 1.0.into();
        for i in 0..Operator::SIZE {
            result.elements[i][i] -= one;
        }
        result
    }

    #[inline]
    pub fn mul_i(&self) -> Self {
        let mut result = *self;
        for i in 0..Operator::SIZE {
            for j in 0..Operator::SIZE {
                result.elements[i][j] = result.elements[i][j].mul_i();
            }
        }
        result
    }

    #[inline]
    pub fn mul_transposed(&self, rhs: &Operator) -> Operator {
        let mut result = Operator::uninitialized();

        for y in 0..Operator::SIZE {
            for x in 0..Operator::SIZE {
                result.elements[y][x] = self.elements[y][0] * rhs.elements[x][0];
                for id in 1..Operator::SIZE {
                    result.elements[y][x] = self.elements[y][id].mul_add(rhs.elements[x][id], result.elements[y][x])
                }
            }
        }
        result
    }

    #[inline]
    pub fn mul_conjugated(&self, rhs: &Operator) -> Operator {
        let mut result = Operator::uninitialized();

        for y in 0..Operator::SIZE {
            for x in 0..Operator::SIZE {
                result.elements[y][x] = self.elements[y][0] * rhs.elements[0][x].conjugate();
            }
            for id in 1..Operator::SIZE {
                for x in 0..Operator::SIZE {
                    result.elements[y][x] = self.elements[y][id].mul_daggered_add(rhs.elements[id][x], result.elements[y][x])
                }
            }
        }
        result
    }

    // Dagger on rhs.
    #[inline]
    pub fn mul_daggered(&self, rhs: &Operator) -> Operator {
        let mut result = Operator::uninitialized();

        for y in 0..Operator::SIZE {
            for x in 0..Operator::SIZE {
                result.elements[y][x] = self.elements[y][0].mul_daggered(rhs.elements[x][0]);
                for id in 1..Operator::SIZE {
                    result.elements[y][x] = self.elements[y][id].mul_daggered_add(rhs.elements[x][id], result.elements[y][x])
                }
            }
        }
        result
    }

    // Dagger on lhs.
    #[inline]
    pub fn daggered_mul(&self, rhs: &Operator) -> Operator {
        let mut result = Operator::uninitialized();

        for y in 0..Operator::SIZE {
            for x in 0..Operator::SIZE {
                result.elements[y][x] = self.elements[0][y].daggered_mul(rhs.elements[0][x]);
            }
        }

        for id in 1..Operator::SIZE {
            for y in 0..Operator::SIZE {
                for x in 0..Operator::SIZE {
                    result.elements[y][x] = self.elements[id][y].daggered_mul_add(rhs.elements[id][x], result.elements[y][x])
                }
            }
        }
        result
    }

    pub fn max_abs(&self) -> (fp, fp) {
        let (mut max_real, mut max_imag) = (fp::MIN, fp::MIN);
        println!("self {}", self);
        for i in 0..Operator::SIZE {
            for j in 0..Operator::SIZE {
                if self[(i,j)].real[0].abs() > max_real {
                    max_real = self[(i,j)].real[0].abs();
                    println!("newmax real {}\n", max_real);
                }
                if self[(i,j)].imag[0].abs() > max_imag {
                    max_imag = self[(i,j)].imag[0].abs();
                }
            }
        }
        (max_real, max_imag)
    }

    pub fn budget_matrix_exponential(&self) -> Result<Operator, fp> {
        let mut result = Operator::identity();
        let mut tmp = Operator::identity();
        for i in 1..16 {
            tmp = Operator::identity();
            for n in 1..=i {
                tmp = tmp * ((1.0/(n as fp)) * self);
            }
            result += tmp;
        }

        // Converge check
        let max = tmp.max_abs();
        let max = max.0.max(max.1);
        println!("MAX {max}");
        if max > 1.0e-30 || result[(0,0)].first().0.is_nan() { Err(max) } else { Ok(result) }
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
        // Seems slightly slower (than reference) at 4x4.
        //self.mul_transposed(&rhs.transpose())

        let mut result = Operator::uninitialized();

        // Reference
        //for y in 0..Operator::SIZE {
        //    for x in 0..Operator::SIZE {
        //        result.elements[y][x] = self.elements[y][0] * rhs.elements[0][x];
        //        for id in 1..Operator::SIZE {
        //            result.elements[y][x] = self.elements[y][id].mul_add(rhs.elements[id][x], result.elements[y][x])
        //        }
        //    }
        //}

        // This is faster (at least at 4x4, measured 3-4% total runtime speedup).
        for y in 0..Operator::SIZE {
            for x in 0..Operator::SIZE {
                result.elements[y][x] = self.elements[y][0] * rhs.elements[0][x];
            }
            for id in 1..Operator::SIZE {
                for x in 0..Operator::SIZE {
                    result.elements[y][x] = self.elements[y][id].mul_add(rhs.elements[id][x], result.elements[y][x])
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

impl Mul<fp> for &Operator {
    type Output = Operator;
    #[inline]
    fn mul(self, rhs: fp) -> Self::Output {
        let to_mul = V::splat(rhs);
        self * &to_mul
    }
}
impl Mul<fp> for Operator {
    type Output = Operator;
    #[inline]
    fn mul(self, rhs: fp) -> Self::Output {
        &self * rhs
    }
}

impl Mul<Operator> for fp {
    type Output = Operator;
    #[inline]
    fn mul(self, rhs: Operator) -> Self::Output {
        &rhs * self
    }
}

impl Mul<&Operator> for fp {
    type Output = Operator;
    #[inline]
    fn mul(self, rhs: &Operator) -> Self::Output {
        rhs * self
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

impl Mul<Operator> for V {
    type Output = Operator;
    #[inline]
    fn mul(self, rhs: Operator) -> Self::Output {
        &rhs * &self
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
        for y in 0..Operator::SIZE {
            for x in 0..Operator::SIZE {
                self.elements[y][x] += rhs.elements[y][x];
            }
        }
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

impl Add<Operator> for &Complex {
    type Output = Operator;

    #[inline]
    fn add(self, rhs: Operator) -> Self::Output {
        let mut result = rhs;
        for i in 0..Operator::SIZE {
            result.elements[i][i] += self;
        }
        result
    }
}

impl Add<Operator> for Complex {
    type Output = Operator;

    #[inline]
    fn add(self, rhs: Operator) -> Self::Output {
        &self+rhs
    }
}

impl Sub<Operator> for &Complex {
    type Output = Operator;

    #[inline]
    fn sub(self, rhs: Operator) -> Self::Output {
        let mut result = rhs;
        for i in 0..Operator::SIZE {
            result.elements[i][i] -= self;
        }
        result
    }
}

impl Sub<Operator> for Complex {
    type Output = Operator;

    #[inline]
    fn sub(self, rhs: Operator) -> Self::Output {
        &self-rhs
    }
}

#[derive(Clone,Copy)]
pub struct StateProbabilitiesSimd {
    pub v: [Real; Operator::SIZE]
}

#[derive(Clone,Copy)]
pub struct StateProbabilities {
    pub v: [fp; Operator::SIZE]
}

impl StateProbabilitiesSimd {

    #[inline]
    pub fn zero() -> StateProbabilitiesSimd {
        StateProbabilitiesSimd { v: [Real::splat(0.); Operator::SIZE] }
    }

    #[inline]
    pub fn add(&mut self, rhs: &StateProbabilitiesSimd) {
        for (s, r) in self.v.iter_mut().zip(rhs.v.iter()) {
            *s += r;
        }
    }

    #[inline]
    pub fn divide(&mut self, rhs: fp) {
        for s in self.v.iter_mut() {
            *s /= Real::splat(rhs);
        }
    }

    pub fn average(&self) -> StateProbabilities {
        let mut result = StateProbabilities { v: [0.0; Operator::SIZE] };
        for (out, v) in result.v.iter_mut().zip(self.v.iter()) {
            *out = v.as_array().iter().sum::<fp>() / Real::LANES as fp;
        }
        result
    }

    pub fn as_array(&self) -> [StateProbabilities; Real::LANES] {
        let mut result: [StateProbabilities; Real::LANES] = unsafe { std::mem::zeroed() };

        for lane in 0..Real::LANES {
            let v = [0.0; Operator::SIZE];

            for i in 0..Operator::SIZE {
                v[i] = self.v[i][lane];
            }

            result[lane] = StateProbabilities { v };
        }

        result
    }
}

impl StateProbabilities {
    pub fn to_le_bytes(&self) -> [u8; std::mem::size_of::<fp>() * Operator::SIZE] {
        const FP_SIZE: usize = std::mem::size_of::<fp>();

        let mut buf = [0u8; FP_SIZE * Operator::SIZE];

        for (vb, v) in buf.chunks_mut(FP_SIZE).zip(self.v.iter()) {
            vb.copy_from_slice(&v.to_le_bytes());
        }

        buf
    }
}
