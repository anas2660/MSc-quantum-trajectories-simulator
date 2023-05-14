#![feature(portable_simd)]
#![feature(array_zip)]
#![allow(non_snake_case)]
#![allow(non_upper_case_globals)]
#![allow(confusable_idents)]

mod num;
use num::*;
mod operator;
use operator::*;
mod matrix;
use matrix::*;
mod histogram;
use histogram::*;
mod hamiltonian;
use hamiltonian::*;
mod lindblad;
use lindblad::*;
mod sgenerator;
use sgenerator::*;
mod initial_state;
use initial_state::*;
mod helpers;
use helpers::*;
mod config;
use config::*;
mod multivec;
use multivec::*;


use rand_distr::StandardNormal;
use std::{io::Write, sync::Arc, simd::{SimdFloat, StdFloat}};

use rand::{thread_rng, Rng};

mod pipewriter;
// use pipewriter::*;

const ONE:     cfp = Complex::new(1.0, 0.0);
const I:       cfp = Complex::new(0.0, 1.0);
const MINUS_I: cfp = Complex::new(0.0, -1.0);
const ZERO:    cfp = Complex::new(0.0, 0.0);

// Initial state probabilities
const INITIAL_PROBABILITIES: [f64; 4] = [
    //0.0, // 00
    //0.45, // 01
    //0.55, // 10
    //0.00  // 11
    //0.30, // 00
    //0.25, // 01
    //0.15, // 10
    //0.30, // 11
    1.0, // 00
    0.0, // 01
    0.0, // 10
    0.0  // 11
];

// Simulation constants
pub const Δt: fp = 0.000005;

const DEFAULT_CONFIG: SimulationConfig = SimulationConfig {
    step_count: 628,
    thread_count: 10,
    simulations_per_thread: 100,
    silent: false,

    // Physical constants
    κ: 0.5,
    κ_1: 0.5, // NOTE:
    β: Complex::new(4.0, 0.0), // Max value is kappa
    γ_dec: 563.9773943, // should be g^2/ω    (174) side 49
    η: 0.96,
    Φ: 0.0,    // c_out phase shift Phi
    γ_φ: 0.001,
    g_0: 10.0, // sqrt(Δ_s_0 χ)
    χ_0: 0.1,
    ω_r: 2500.0,

    Δ_r: 0.0,
    Δ_br: 10.0, // ω_b - ω_r
};

const SIMULATION_COUNT: u32 = DEFAULT_CONFIG.thread_count * DEFAULT_CONFIG.simulations_per_thread;

const HIST_BIN_COUNT: usize = 32;
//const STEP_COUNT: u32 = 628;
//const THREAD_COUNT: u32 = 12;
//const SIMULATIONS_PER_THREAD: u32 = 100;

// Physical constants
// const κ: fp = 0.5;
// const κ_1: fp = 0.5; // NOTE: Max value is the value of kappa. This is for the purpose of loss between emission and measurement.
// const β: cfp = Complex::new(4.0, 0.0); // Max value is kappa
// const γ_dec: f64 = 563.9773943; // should be g^2/ω    (174) side 49
// const η: fp = 0.96;
// const Φ: fp = 0.0; // c_out phase shift Phi
// const γ_φ: f64 = 0.001;
// //const ddelta: fp = delta_r - delta_s;
// const g_0: fp = 10.0; // sqrt(Δ_s_0 χ)
// const χ_0: fp = 0.1;
// const ω_r: fp = 2500.0;
//////////////const ω_s_0: fp = ω_r + g_0 * g_0 / χ_0;
////////////////const Δ_s_0: fp = 26318.94506957162;
//////////////
//////////////const ω_b: fp = ω_r;
////////////////const Δ_s:  [fp; 2] = [Δ_s_0+0., Δ_s_0-0.]; // |ω_r - ω_s|
//////////////const Δ_b: [fp; 2] = [20.0, 20.0]; //  ω_b - ω_s
//////////////const Δ_br: fp = 10.0; // ω_b - ω_r
//////////////const Δ_r: fp = 0.0;
//////////////const χ: [fp; 2] = [χ_0 + 0.00, χ_0 - 0.00];
//////////////const g: [fp; 2] = [g_0, g_0];


const ω_not: f64 = 500.0;
const NOISE_FACTOR: fp = 1.0;

#[derive(Clone)]
struct QubitSystem {
    ρ: Operator,
    c_out_phased: Lindblad,
    c1: Lindblad,
    c2: Lindblad,
    c3: [Lindblad; Operator::QUBIT_COUNT],
    dZ: Complex,
}

impl QubitSystem {

    fn new(initial_state: InitialState, config: &SimulationConfig) -> (Self, QuantumCircuit) {
        let c = config;

        #[allow(unused_variables)]
        let σ_plus   = Matrix::new(0.0, 1.0,  0.0,  0.0);
        let σ_x      = Matrix::new(0.0, 1.0,  1.0,  0.0);
        let σ_y      = SC!(0.0,1.0) * Matrix::new(0.0, -1.0, 1.0,  0.0);
        let σ_z      = Matrix::new(1.0, 0.0,  0.0, -1.0);
        let σ_minus  = Matrix::new(0.0, 0.0,  1.0,  0.0);
        let hadamard = Matrix::new(1.0, 1.0,  1.0, -1.0);
        let identity = Operator::identity();

        // TODO: Add offsets to config
        //const Δ_s:  [fp; 2] = [Δ_s_0+0., Δ_s_0-0.]; // |ω_r - ω_s|
        let Δ_b: [fp; 2] = [20.0, 20.0]; //  ω_b - ω_s
        //let Δ_r: fp = 0.0;
        let χ: [fp; 2] = [config.χ_0 + 0.00, config.χ_0 - 0.00];
        let g: [fp; 2] = [config.g_0, config.g_0];


        // DISPERSIVE
        let a = Operator::from_fn(|r, c| {
            (r==c) as u32 as fp
                * (2.*config.κ_1).sqrt() * config.β
                / (config.Δ_r + config.κ + I*χ.iter().enumerate().fold(0.0, |acc, (n, χ_n)| {
                    acc - (((c>>n)&1) as i32 * 2 - 1) as fp * χ_n // TODO: Check this
            }))
        });

        let N = a.dagger() * a;

        // Hamiltonian
        let term2 = apply_and_scale_individually(&Δ_b.zip(χ).map(|(Δ_bs, χ_s)| /**/0.5*Δ_bs*identity + χ_s*N), &σ_z);

        let σ_m = apply_individually_parts(&σ_minus);
        let σ_p = apply_individually_parts(&σ_plus);

        let mut term3 = Operator::zero();

        for s in 0..Operator::QUBIT_COUNT {
            for q in 0..Operator::QUBIT_COUNT {
                let c = χ[s] * (g[q]/g[s]);
                term3 += c * σ_p[s] * σ_m[q] + c * σ_p[q] * σ_m[s];
            }
        }

        let H = c.Δ_br * N + term2 + term3
            + I*(2.0 * c.κ_1).sqrt() * (c.β * a.dagger() - c.β.conjugate() * a); // Detuning

        let zero = Matrix::vector(&[1., 0.]);
        let one  = Matrix::vector(&[0., 1.]);

        let ket1 = &one;
        let bra1 = &one.transpose();
        let ket0 = &zero;
        let bra0 = &zero.transpose();

        //// CNOT 0, 1
        //let ω = 1000.0;
        //let cnot = ω * (ket1*bra1).kronecker(&(ket1*bra0 + ket0*bra1)).to_operator();

        // NOT 0
        let not = apply_individually_parts(&(ω_not * &σ_x));

        // Remove any non-hermitian numerical error
        let H = 0.5*(H + H.conjugate());
        if !config.silent { println!("H:\n{H}") };

        // Lindblad terms
        let c_out = (c.κ_1 * 2.0).sqrt() * a - c.β * identity;

        let T = Δt * (c.step_count-1) as fp;

        let circuit = QuantumCircuit::new(&[
            (H + not[0], T*0.5),
            //(H, T*0.5)
        ], c);

        (
            Self {
                ρ: initial_state.into(),
                c_out_phased: Lindblad::new(c_out * ((I * c.Φ).exp())),
                c1: Lindblad::new((2.0 * c.κ).sqrt() * a),
                c2: Lindblad::new(apply_individually(&(c.γ_dec.sqrt() * &σ_minus))),
                c3: apply_individually_parts(&((c.γ_φ / 2.0).sqrt() * &σ_z)).map(Lindblad::new),
                dZ: Complex::new(0.0, 0.0),
            },
            circuit
        )
    }

    fn lindblad_terms(&self, ρ: &Operator) -> Operator {
        let mut lt = Operator::lindblad_term(ρ, &self.c1); // Photon field transmission/losses

        lt.lindblad(ρ, &self.c_out_phased);

        for c3 in self.c3.iter() {
            lt.lindblad(ρ, c3);
        }

        lt
        // .lindblad(ρ, &self.c2) // Decay to ground state
        // .lindblad2(ρ, &self.c_out_phased) // Second order
    }

    fn stochastic(&self, ρ: &Operator) -> [Operator; 2] {
        let ρ_norm = *ρ.clone().normalize();
        let r = self.c_out_phased.c;
        let rρ = r*ρ*FRAC_1_SQRT_2;
        let a1 = rρ - rρ.trace()*ρ_norm;
        let a2 = a1.dagger();
        [ a1+a2, I*(a2-a1) ]
    }

    fn srk2(&mut self, H: &Hamiltonian, S: [SV32; 2]) {
        let ΔWx = self.dZ.real;
        let ΔWy = self.dZ.imag;

        let sqrtΔt = Real::splat(Δt.sqrt());

        let S = S.map(convert_sv32_to_real);

        let offset = S.map(|s| s * sqrtΔt );

        let bt = self.stochastic(&self.ρ);
        let K1 = Δt * self.lindblad_terms(&self.ρ)
            + (ΔWx - offset[0]) * bt[0]
            + (ΔWy - offset[1]) * bt[1];

        let evolved_ρ = H.apply(&self.ρ);

        let new_ρ = evolved_ρ + K1;
        // new_ρ.normalize();

        let bt = self.stochastic(&new_ρ);
        let K2 = Δt * self.lindblad_terms(&new_ρ)
            + (ΔWx + offset[0]) * bt[0]
            + (ΔWy + offset[1]) * bt[1];

        self.ρ = evolved_ρ + 0.5 * (K1 + K2);
    }

    fn euler(&mut self, H: &Hamiltonian) {
        let a = self.lindblad_terms(&self.ρ);
        let b = self.stochastic(&self.ρ);
        self.ρ = H.apply(&self.ρ) + a * Δt + self.dZ.real*b[0] + self.dZ.imag*b[1];
    }
}




#[derive(Clone)]
struct SimulationResults {
    measurement_records: Option<MVec<Complex>>, //[[Complex; STEP_COUNT as usize]; SIMULATION_COUNT as usize]
    last_fidelities: Option<Vec<Real>>
}

fn simulate<const CUSTOM_RECORDS: bool, const RETURN_RECORDS: bool, const WRITE_FILES: bool>(initial_state: InitialState, config: &SimulationConfig, records: Option<Arc<SimulationResults>>)
     -> SimulationResults {

    let timestamp = std::time::SystemTime::UNIX_EPOCH
        .elapsed()
        .unwrap()
        .as_secs();

    // This if statement seems ridiculous, but it cannot resolve OutputFile::<WRITE_FILES>::create.
    let create_output_file = |name| {
        if WRITE_FILES {
            std::fs::File::create(format!("results/{timestamp}_{name}")).unwrap()
        } else {
            if cfg!(windows) {
                // FIXME: I haven't tested if "nul" works on windows.
                std::fs::File::create("nul").unwrap()
            } else {
                std::fs::File::create("/dev/null").unwrap()
            }
        }
    };

    let mut parameter_file   = create_output_file("parameters.txt");
    let mut data_file        = create_output_file("trajectories.dat");
    let mut hist_file        = create_output_file("hist.dat");
    let mut current_file     = create_output_file("currents.dat");
    let mut final_state_file = create_output_file("final_state.dat");
    let mut fidelity_file    = create_output_file("fidelity.dat");

    // FIXME // TODO:
//    parameter_file
//        .write_all(
//            format!(
//"let β = {β};
//let Δ_r = {Δ_r};
//let κ = {κ};
//let κ_1 = {κ_1};
//let η = {η};
//let Φ = {Φ};
//let γ_dec = {γ_dec};
//let γ_φ = {γ_φ};
//").as_bytes(),
//        ).unwrap();

    // metadata
    current_file.write_all(&(SIMULATION_COUNT * Real::LANES as u32).to_le_bytes()).unwrap();

    //current_file.write(&STEP_COUNT.to_le_bytes()).unwrap();

    #[cfg(not(feature = "double-precision"))]
    data_file.write_all(&(0u32.to_le_bytes())).unwrap();
    #[cfg(feature="double-precision")]
    data_file.write_all(&(1u32.to_le_bytes())).unwrap();

    data_file.write_all(&(SIMULATION_COUNT * Real::LANES as u32).to_le_bytes()).unwrap();
    data_file.write_all(&(Operator::SIZE as u32).to_le_bytes()).unwrap();
    data_file.write_all(&(config.step_count + 1).to_le_bytes()).unwrap();

    hist_file.write_all(&(Operator::SIZE as u32).to_le_bytes()).unwrap();
    hist_file.write_all(&(config.step_count + 1).to_le_bytes()).unwrap();
    hist_file.write_all(&(HIST_BIN_COUNT as u32 * Operator::SIZE as u32).to_le_bytes()).unwrap();

    struct ThreadResult {
        trajectory_sum: Vec<StateProbabilitiesSimd>, // [StateProbabilitiesSimd; STEP_COUNT as usize+1],
        trajectory_hist: MVec<Histogram<HIST_BIN_COUNT>>, // [[Histogram<HIST_BIN_COUNT>; Operator::SIZE]; STEP_COUNT as usize+1],
        current_sum: Vec<Complex>, //[Complex; SIMULATIONS_PER_THREAD as usize],
        final_states: Vec<StateProbabilitiesSimd>, //[StateProbabilitiesSimd; SIMULATIONS_PER_THREAD as usize],
        measurements: Option<MVec<Complex>>, // Option<Box<[[Complex; STEP_COUNT as usize]; SIMULATIONS_PER_THREAD as usize]>>,
        fidelities: Option<MVec<Real>>    // Option<Box<[[Complex; STEP_COUNT as usize]; SIMULATIONS_PER_THREAD as usize]>>
    }

    // Combination of all thread results
    let mut trajectory_average = vec![StateProbabilitiesSimd::zero(); (config.step_count+1) as usize].into_boxed_slice();
    //let mut trajectory_histograms = alloc_zero!([[Histogram::<HIST_BIN_COUNT>; Operator::SIZE]; config.step_count as usize+1]);
    let mut trajectory_histograms = unsafe { MVec::<Histogram<HIST_BIN_COUNT>>::alloc_zeroed(config.step_count as usize+1, Operator::SIZE)};// alloc_zero!([[Histogram::<HIST_BIN_COUNT>; Operator::SIZE]; config.step_count as usize+1]);
    let mut measurements = unsafe {MVec::alloc_zeroed(config.simulation_count(), config.step_count as usize)};
    let mut fidelities = Vec::new();
    let mut last_fidelities = Vec::new();

    // Communication channel to send thread results.
    let (tx, rx) = std::sync::mpsc::channel();

    std::thread::scope(|s|{
        for thread_id in 0..config.thread_count {
            let input_records_arc = records.clone();
            let initial_state = initial_state.clone();
            let tx = tx.clone();
            s.spawn(move || {

                //let mut local = alloc_zero!(ThreadResult);
                let mut local = ThreadResult {
                    trajectory_sum: Vec::with_capacity(config.step_count as usize + 1),
                    trajectory_hist: unsafe { MVec::alloc_zeroed(config.step_count as usize+1, Operator::SIZE) },
                    current_sum: Vec::with_capacity(config.simulations_per_thread as usize),
                    final_states: Vec::with_capacity(config.simulations_per_thread as usize),
                    measurements: None,
                    fidelities: None
                };
                unsafe {
                    local.trajectory_sum.set_len(config.step_count as usize + 1);
                    local.trajectory_sum.iter_mut().for_each(|s| *s = std::mem::zeroed() );
                }

                //let mut measurements =  alloc_zero!([[Complex; STEP_COUNT as usize]; SIMULATIONS_PER_THREAD as usize]);
                //let mut fidelities    = alloc_zero!([[Complex; STEP_COUNT as usize]; SIMULATIONS_PER_THREAD as usize]);
                let mut measurements = unsafe {MVec::alloc_zeroed(config.simulations_per_thread as usize, config.step_count as usize)};
                let mut fidelities = unsafe {MVec::alloc_zeroed(config.simulations_per_thread as usize, config.step_count as usize)};
                let input_records = &input_records_arc;

                // Start the timer.
                let now = std::time::Instant::now();
                let mut total_section_time = 0;

                // Calculate ideal ρ.
                let ideal_ρ = get_ideal_ρ(&initial_state);

                // Create initial system.
                let (initial_system, circuit) = QubitSystem::new(initial_state, config);

                // RNG
                let mut rng = thread_rng();

                // S generator
                let mut S = SGenerator::new(&mut rng);

                // sqrt(η/2) is from the definition of dZ.
                // sqrt(dt) is to make the variance σ = dt
                let sqrtηdt = Real::splat((config.η * 0.5).sqrt() * Δt.sqrt() * NOISE_FACTOR);
                if !config.silent {println!("sqrtηdt: {sqrtηdt:?}")};

                for simulation in 0..config.simulations_per_thread {
                    // Initialize system
                    let mut system = initial_system.clone();
                    let mut circuit_state = circuit.make_new_state();
                    let mut t = Δt;

                    let c = system.c_out_phased.c;
                    let x = c + c.dagger();
                    let y = MINUS_I*(c - c.dagger());

                    let mut J = ZERO;

                    let (mut H, mut steps_to_next_gate) = circuit_state.next();

                    let current_measurements = &mut measurements[simulation as usize];
                    let fidelities = &mut fidelities[simulation as usize];

                    // Do 2000 steps.
                    for step in 0..config.step_count as usize {

                        if steps_to_next_gate == 0 {
                            (H, steps_to_next_gate) = circuit_state.next();
                        }
                        steps_to_next_gate -= 1;

                        // Write current state.
                        let P = system.ρ.get_probabilites_simd();
                        local.trajectory_sum[step].add(&P);
                        for (i, p) in P.v.iter().enumerate() {
                            local.trajectory_hist[step][i].add_values(p);
                        }

                        // TODO: DELETE
                        // assert_eq!((system.rho[(0, 0)].imag()*system.rho[(0, 0)].imag()).simd_lt(Real::splat(0.02)).to_bitmask(), 255);

                        // Get a dZ value.
                        if CUSTOM_RECORDS {
                            let dI = input_records.as_ref().unwrap().measurement_records.as_ref().unwrap()[thread_id as usize * config.simulations_per_thread as usize + simulation as usize][step];

                            let to_sub = Complex {
                                real: (x*system.ρ).trace().real,
                                imag: (y*system.ρ).trace().real
                            };

                            const DT_OVER_SQRT2: Real = Real::from_array([Δt/SQRT_2; Real::LANES]);
                            system.dZ = DT_OVER_SQRT2 * &(dI - to_sub);
                        } else {
                            // Sample on the normal distribution.
                            for lane in 0..Real::LANES {
                                system.dZ.real[lane] = rng.sample::<fp, StandardNormal>(StandardNormal);
                                system.dZ.imag[lane] = rng.sample::<fp, StandardNormal>(StandardNormal);
                            }
                            system.dZ *= &sqrtηdt;
                        }

                        // Do the stochastic rk2 step.
                        system.srk2(H, [S.gen(&mut rng), S.gen(&mut rng)]);
                        /////system.euler(H);

                        // Normalize rho.
                        system.ρ.normalize();

                        // Compute fidelity.
                        //let fidelity_part = (system.ρ*ideal_ρ).sqrt().trace();
                        //fidelities[step] = fidelity_part*fidelity_part;
                        if Operator::SIZE == 2 {
                            fidelities[step] = system.ρ.fidelity_2x2(&ideal_ρ);
                        }

                        // Compute current.
                        const SQRT2_OVER_DT: Real = Real::from_array([SQRT_2/Δt; Real::LANES]);
                        let dI = Complex {
                            real: (x*system.ρ).trace().real + SQRT2_OVER_DT * system.dZ.real,
                            imag: (y*system.ρ).trace().real + SQRT2_OVER_DT * system.dZ.imag
                        };

                        J += dI;
                        current_measurements[step] = dI;

                        t += Δt;
                    }

                    // Write last state.
                    let P = system.ρ.get_probabilites_simd();
                    local.trajectory_sum[config.step_count as usize].add(&P);
                    for (i, p) in P.v.iter().enumerate() {
                        local.trajectory_hist[config.step_count as usize][i].add_values(p);
                    }

                    //local.current_sum[simulation as usize] = J;
                    //local.final_states[simulation as usize] = P;
                    local.current_sum.push(J);
                    local.final_states.push(P);
                }

                let total_time = now.elapsed().as_millis();


                if !config.silent {
                    println!("Thread {thread_id} finished {} simulations in {total_time} ms ({} μs/sim) (total section time {} ms)",
                             config.simulations_per_thread,
                             (total_time*1000)/(config.simulations_per_thread*Real::LANES as u32) as u128,
                             total_section_time/1000);
                }

                for s in local.trajectory_sum.iter_mut() {
                    s.divide(config.simulations_per_thread as fp);
                }

                local.measurements = if RETURN_RECORDS { Some(measurements) } else { None };
                local.fidelities = if Operator::SIZE == 2 { Some(fidelities) } else { None };

                tx.send(local).unwrap();
            });
        }

        // Combine thread results.
        for i in 0..(config.thread_count as usize) {
            //let local = tt.join().unwrap();
            let local = rx.recv().unwrap();

            for (s, ls) in trajectory_average.iter_mut().zip(local.trajectory_sum.iter()) {
                s.add(ls);
            }

            //for (i, histograms) in trajectory_histograms.iter_mut().enumerate() {
            for i in 0..trajectory_histograms.row_count() {
                let local_histograms = &local.trajectory_hist[i];
                let histograms = &mut trajectory_histograms[i];
                for (histogram, local_histogram) in histograms.iter_mut().zip(local_histograms.iter()) {
                    histogram.add_histogram(local_histogram);
                }
            }

            for the_current in local.current_sum.iter() {
                for lane in 0..Real::LANES {
                    current_file.write_all(&the_current.real.as_array()[lane].to_le_bytes()).unwrap();
                    current_file.write_all(&the_current.imag.as_array()[lane].to_le_bytes()).unwrap();
                }
            }

            for the_final_state in local.final_states.iter() {
                let final_state_array = the_final_state.as_array();
                for final_state in final_state_array {
                    final_state_file.write_all(&final_state.to_le_bytes()).unwrap();
                }
            }

            if let Some(m) = local.measurements {
                let a = i * config.simulations_per_thread as usize;
                let b = (i+1) * config.simulations_per_thread as usize;
                measurements[a..b].copy_from_slice(m.as_slice());
            }

            if let Some(f) = local.fidelities {
                fidelities.extend_from_slice(f.as_slice());
                for i in 0..f.row_count() {
                    last_fidelities.push(*f[i].last().unwrap());
                }
            }
        }
    });


    for s in trajectory_average.iter_mut() {
        s.divide(config.thread_count as fp);
        data_file.write_all(&s.average().to_le_bytes()).unwrap();
    }

    // TODO: actual time
    // TODO: fix magic number (simcount)
    for i in 0..=config.step_count {
        let t = i as fp * Δt;
        data_file.write_all(&t.to_le_bytes()).unwrap();
    }

    let mut buffer = Vec::with_capacity(std::mem::size_of::<u32>() * HIST_BIN_COUNT * Operator::SIZE * config.step_count as usize);
    //for state in trajectory_histograms.iter() {
    //    for hist in state.iter() {
    for hist in trajectory_histograms.as_slice().iter() {
        for bin in hist.bins.iter().rev() {
            buffer.extend_from_slice(&bin.to_le_bytes());
        }
    }
    hist_file.write_all(&buffer).unwrap();

    buffer.clear();
    if !fidelities.is_empty() {
        for lane in 0..V::LANES {
            for sim in fidelities.iter() {
                buffer.extend_from_slice(&sim[lane].to_le_bytes());
                //for f in sim.iter() {
                //    buffer.extend_from_slice(&f.real[lane].to_le_bytes());
                //}
            }
        }
        fidelity_file.write_all(&buffer).unwrap();
    }

    SimulationResults {
        measurement_records: if RETURN_RECORDS { Some(measurements) } else { None },
        last_fidelities: if Operator::SIZE == 2 { Some(last_fidelities) } else {None}
    }
}

fn fidelity_data() {
    let mut conf = DEFAULT_CONFIG;
    conf.simulations_per_thread = 750;
    conf.silent = true;

    let mut csv_data = Vec::new();

    for i in 0..=20 {
        for j in 0..=20 {
            conf.χ_0 = i as fp / 5.0;
            conf.β = Complex::new(j as fp, 0.0);
            let simulation_results = simulate::<false, false, false>(InitialState::Probabilites(INITIAL_PROBABILITIES.to_vec()), &conf, None);
            let fidelities = simulation_results.last_fidelities.as_ref().unwrap();
            let zero = Real::splat(0.0);
            let n = (fidelities.len() * Real::LANES) as fp;

            // Compute mean.
            let sum = fidelities.iter().fold(zero, |acc, f| acc + f).reduce_sum();
            let mean = sum / n;

            // Compute sample standard deviation.
            let real_mean = Real::splat(mean);
            let inner_sum = fidelities.iter().fold(zero, |acc, x| {
                let delta = x-real_mean;
                acc + delta*delta
            }).reduce_sum();
            let std = (inner_sum/(n - 1.0)).sqrt();

            // Save result
            csv_data.push((i, conf.χ_0, conf.β, mean, std));

            println!("Completed [i:{i}/20, j:{j}/20], χ: {}, β: {}, μ: {mean}, std: {std}", conf.χ_0, conf.β.real[0]);
        }
    }

    // Write results
    let mut csv = std::fs::File::create("fidelity_data.csv").unwrap();
    csv.write_all(b"index, chi, beta, mean, standard_deviation\n").unwrap();
    csv_data.iter().for_each(|(i,χ,β,μ,s)| csv.write_all(format!("{i},{χ},{},{μ},{s}\n", β.real[0]).as_bytes()).unwrap() );
}

fn simple() {
    simulate::<false, false, true>(InitialState::Probabilites(INITIAL_PROBABILITIES.to_vec()), &DEFAULT_CONFIG, None);
}

fn feed_current_known_state () {
    // Current of known state of |11>
    let a = simulate::<false, true, false>(InitialState::Probabilites(vec![0.0, 0.45, 0.55, 0.0]), &DEFAULT_CONFIG, None);

    // Feed current into perfect superposition.
    let b = simulate::<true, false, true>(InitialState::Probabilites(vec![0.25, 0.25, 0.25, 0.25]), &DEFAULT_CONFIG, Some(a.into()));
}

fn main() {
    println!("Starting simulation...");

    #[cfg(feature="double-precision")]
    println!("DOUBLE PRECISION");

    let start = std::time::Instant::now();
    //feed_current_known_state();
    //simple();
    fidelity_data();
    let elapsed = start.elapsed().as_millis();

    println!(
        "Simulations took {elapsed} ms ({} sim/s, {} steps/s)",
        (1000*SIMULATION_COUNT*Real::LANES as u32) as fp / elapsed as fp,
        (1000*SIMULATION_COUNT as u64 * DEFAULT_CONFIG.step_count as u64 * Real::LANES as u64) as u128 / elapsed
    );
}
