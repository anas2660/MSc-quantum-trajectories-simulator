
use crate::num::*;
use crate::operator::*;

// TODO multiple qubit operations simultaneously, [QuantumGate; Operator::QUBIT_COUNT]
#[derive(Clone)]
struct QuantumGate {
    hamiltonian: Operator,
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
            ops.push(QuantumGate { hamiltonian: *H, steps: (*t/crate::Δt) as usize });
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
    pub fn next(&mut self) -> (&Operator, usize) {
        //if self.next as usize == self.circuit.operations.len() { return None };

        let op = &self.circuit.operations[self.next as usize];
        self.next = u32::min(self.next+1, self.circuit.operations.len() as u32 - 1);

        (&op.hamiltonian, op.steps)
    }
}
