use crate::operator::Operator;

#[derive(Clone)]
pub struct Lindblad {
    pub c: Operator,
    pub c_dag_c: Operator
}

impl Lindblad {
    pub fn new(c: Operator) -> Lindblad {
        Lindblad { c, c_dag_c: c.daggered_mul(&c) }
    }
}
