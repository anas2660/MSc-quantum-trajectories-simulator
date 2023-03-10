use crate::num::*;
use crate::operator::*;
use crate::matrix::*;

// TODO multiple qubit operations simultaneously, [QuantumGate; Operator::QUBIT_COUNT]
#[derive(Clone)]
struct QuantumGate {
    hamiltonian: Hamiltonian,
    steps: usize // Amount of time to apply the gate
}

#[derive(Clone)]
pub struct QuantumCircuit {
    operations: Vec<QuantumGate>,
}

pub struct QuantumCircuitState<'a> {
    circuit: &'a QuantumCircuit,
    next: u32 // next operation to perform
}

#[derive(Clone)]
pub struct Hamiltonian {
    H: Operator
}

fn solve_analytically(H: &Matrix) -> Operator {
    let mut current_Δt = crate::Δt as f64;

    for attempt in 1..32 {
        println!("attempt {attempt}");
        println!("current Δt {current_Δt}");
        let result = (SC!(0.0,-current_Δt)*H).budget_matrix_exponential();
        if let Ok(result) = result  {
            println!("Ok(result) = \n{}", result);
            return result.pow(1<<attempt).to_operator();
        }

        match result {
            Ok(_) => (),
            Err(max) => println!("Max was {max}"),
        }

        current_Δt /= 2.0;
    }

    panic!("Insufficient precision in matrix exponential when solving analytically!");
}


impl Hamiltonian {

    #[inline]
    pub fn apply(&self,  ρ: &Operator) -> Operator {
        self.H * ρ.mul_conjugated(&self.H)
    }

}


impl QuantumCircuit {

    // Using gate bases
    // pub fn new(gate_descriptions: &[(&[GateBase], fp)]) -> QuantumCircuit {
    //     let mut ops: Vec<QuantumGate> = Vec::new();
    //     for (bases, time) in gate_descriptions {
    //         let mut gate = QuantumGate { bases: [None; 2], time: *time };
    //         assert!(bases.len() <= 2); // TODO: Support more than two bases
    //         for (i, base) in bases.iter().enumerate() {
    //             gate.bases[i] = Some(*base);
    //         }
    //         ops.push(gate)
    //     }
    //     QuantumCircuit { operations: ops }
    // }

    pub fn new(gates: &[(Operator, fp)]) -> QuantumCircuit {
        let mut ops: Vec<QuantumGate> = Vec::new();

        let mut circuit_time = 0.0;
        for (H, t) in gates {
            let Hmat: Matrix = (*H).into();
            ops.push(QuantumGate { hamiltonian: Hamiltonian { H: solve_analytically(&Hmat) }, steps: (*t/crate::Δt) as usize });
            println!("p:\n{}", ops.last().unwrap().hamiltonian.H);
            circuit_time += t;
        }

        // NOTE: Maybe debug_assert? or disable to allow partial circuit simulations
        assert!(circuit_time < crate::STEP_COUNT as fp * crate::Δt, "Total simulation time is less than circuit time! Increase the simulation time or disable this assert.");

        QuantumCircuit { operations: ops }
    }

    pub fn make_new_state(&self) -> QuantumCircuitState {
        QuantumCircuitState { circuit: self, next: 0 }
    }

}

impl<'a> QuantumCircuitState<'a> {
    pub fn next(&mut self) -> (&Hamiltonian, usize) {
        //if self.next as usize == self.circuit.operations.len() { return None };

        let op = &self.circuit.operations[self.next as usize];
        self.next = u32::min(self.next+1, self.circuit.operations.len() as u32 - 1);

        (&op.hamiltonian, op.steps)
    }
}
