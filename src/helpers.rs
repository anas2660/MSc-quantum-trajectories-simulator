use crate::{operator::Operator, matrix::Matrix};

#[inline(always)]
pub fn commutator(a: &Operator, b: &Operator) -> Operator {
    a*b - b*a
}

#[inline]
pub fn anticommutator(a: Operator, b: Operator) -> Operator {
    &(a * b) + &(b * a)
}

macro_rules! alloc_zero {
    ($T: ty) => {
        unsafe {
            const layout: std::alloc::Layout = std::alloc::Layout::new::<$T>();
            let ptr = std::alloc::alloc_zeroed(layout) as *mut $T;
            Box::from_raw(ptr)
        }
    };
}
pub(crate) use alloc_zero;

pub fn anti_tensor_commutator(lhs: &Matrix, rhs: &Matrix) -> Matrix {
    lhs.kronecker(rhs) + rhs.kronecker(lhs)
}

pub fn apply_individually_parts(op: &Matrix) -> [Operator; Operator::QUBIT_COUNT] {
    let identity = Matrix::identity(2);

    // Create [[M,I], [I,M]].
    let mut parts = [[&identity; Operator::QUBIT_COUNT]; Operator::QUBIT_COUNT];
    (0..Operator::QUBIT_COUNT).for_each(|i| parts[i][i] = op );

    // Combine into [M ⊗ I, I ⊗ M].
    parts.map(|mp| {
        mp.iter().skip(1).fold(mp[0].clone(), |acc, m| acc.kronecker(m)).to_operator()
    })
}

pub fn apply_individually(op: &Matrix) -> Operator {
    apply_individually_parts(op).iter().fold(Operator::zero(), |sum, x| &sum+x)
}

pub fn apply_and_scale_individually<T>(factors: &[T], op: &Matrix) -> Operator
    where T: std::ops::Mul<Operator, Output = Operator> + Copy
{
    let mut actual_factors: [T; Operator::QUBIT_COUNT] = unsafe { std::mem::zeroed() };
    for (actual, given) in actual_factors.iter_mut().zip(factors) {
        *actual = *given;
    }

    apply_individually_parts(op).iter()
        .zip(actual_factors).fold(Operator::zero(), |sum, (part, factor)| sum + factor * (*part) )
}
