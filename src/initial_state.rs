use crate::{operator::Operator, matrix::Matrix};

#[derive(Clone)]
pub enum InitialState {
    Probabilites(Vec<f64>),
    StateVector([f64; Operator::SIZE-1]),
    DensityMatrix(Operator),
}

impl From<InitialState> for Operator {
    fn from(value: InitialState) -> Self {
        match value {
            InitialState::Probabilites(P) => {
                let mut actual_probabilities = [0.0f64; Operator::SIZE-1];

                // Copy the given probabilites to the actual probabilites.
                for (actual, given) in actual_probabilities.iter_mut().zip(P.iter()) {
                    *actual = *given;
                }

                // Get probability sum.
                let p_sum = actual_probabilities.iter().fold(0.0, |acc, x| acc+x);

                // Normalize probabilities and take square root to get coefficients.
                let ψ = Matrix::vector(&actual_probabilities.map(|p| (p/p_sum).sqrt() ));

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
