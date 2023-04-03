use crate::{operator::Operator, matrix::Matrix};

#[derive(Clone, Copy)]
pub enum InitialState {
    Probabilites([f64; Operator::SIZE-1]),
    StateVector([f64; Operator::SIZE-1]),
    DensityMatrix(Operator),
}

impl From<InitialState> for Operator {
    fn from(value: InitialState) -> Self {
        match value {
            InitialState::Probabilites(P) => {
                // Get probability sum.
                let p_sum = P.iter().fold(0.0, |acc, x| acc+x);

                // Normalize probabilities and take square root to get coefficients.
                let ψ = Matrix::vector(&P.map(|p| (p/p_sum).sqrt() ));

                // Convert to density matrix
                (&ψ * ψ.transpose()).to_operator()
            },
            InitialState::StateVector(p) => {
                // Convert to density matrix
                let ψ = Matrix::vector(&p);
                (&ψ * ψ.transpose()).to_operator()
            },
            InitialState::DensityMatrix(ρ) => ρ,
        }
    }
}
