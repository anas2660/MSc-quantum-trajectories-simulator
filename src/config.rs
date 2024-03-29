use crate::operator::Operator;

use super::num::*;


#[allow(clippy::large_enum_variant)]
pub enum IdealState {
    Full(Operator), // Only works for 2x2, where qubit_count=1
    Partial((Complex,Complex,Complex,Complex)) // Row major. Only works for 4x4, where qubit_count=2
}

pub struct SimulationConfig {
    // Simulation constants
    pub step_count: u32,
    pub thread_count: u32,
    pub simulations_per_thread: u32,
    pub silent: bool, /// will silence all prints

    // The simulation_count is thread_count * simulations_per_thread. The total
    // trajectory count is simulation_count*Real::LANES (which is 4 for double
    // precision and 8 for single precision)

    pub fidelity_probe: Option<(fp, IdealState)>, // (Time at which to probe, Ideal state to compare with)
    pub measure_purity: bool,

    // Physical constants
    pub κ: fp,
    pub κ_1: fp,    /// NOTE: Max value is the value of kappa. This is for the purpose of loss between emission and measurement.
    pub β: cfp,
    pub γ_dec: f64, /// should be g^2/ω  (174) side 49
    pub η: fp,
    pub Φ: fp,      /// c_out phase shift Phi
    pub γ_φ: f64,
    pub g_0: fp,
    pub χ_0: fp,
    pub ω_r: fp,
    //const ddelta: fp,

    pub Δ_r: fp,

    pub Δ_br: fp, // ω_b - ω_r

    pub chi_offsets: Option<Vec<ChiOffset>>
}

pub struct ChiOffset {
    pub index: usize,
    pub offset: fp
}


impl SimulationConfig {
    pub fn simulation_count(&self) -> usize {
        self.simulations_per_thread as usize * self.thread_count as usize
    }
}
