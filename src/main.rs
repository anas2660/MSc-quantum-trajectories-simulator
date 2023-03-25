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

use rand_distr::StandardNormal;
use std::{io::Write, sync::Arc, clone};

use rand::{rngs::ThreadRng, thread_rng, Rng};

mod pipewriter;
// use pipewriter::*;

const ONE:     cfp = Complex::new(1.0, 0.0);
const I:       cfp = Complex::new(0.0, 1.0);
const MINUS_I: cfp = Complex::new(0.0, -1.0);
const ZERO:    cfp = Complex::new(0.0, 0.0);

// Initial state probabilities
const INITIAL_PROBABILITIES: [fp; Operator::SIZE] = [
    //0.0, // 00
    //0.45, // 01
    //0.55, // 10
    //0.00  // 11
    0.50, // 00
    0.00, // 01
    0.00, // 10
    0.50, // 11
];

// Simulation constants
pub const Δt: fp = 0.001;
const STEP_COUNT: u32 = 5000;
const THREAD_COUNT: u32 = 14;
const HIST_BIN_COUNT: usize = 64;
const SIMULATIONS_PER_THREAD: u32 = 50;
const SIMULATION_COUNT: u32 = THREAD_COUNT * SIMULATIONS_PER_THREAD;

// Physical constants
const κ: fp = 1.2;
const κ_1: fp = 1.2; // NOTE: Max value is the value of kappa. This is for the purpose of loss between emission and measurement.
const β: cfp = Complex::new(1.0, 0.0); // Max value is kappa
const γ_dec: f64 = 563.9773943; // should be g^2/ω    (174) side 49
const η: fp = 0.95;
const Φ: fp = 0.0; // c_out phase shift Phi
const γ_φ: f64 = 0.001;
//const ddelta: fp = delta_r - delta_s;
const χ_0: fp = 0.6;
const g_0: fp = 24.5; // sqrt(Δ_s_0 χ)
const ω_r: fp = 2500.0;
const ω_s_0: fp = 3500.0;
//const Δ_s_0: fp = 26318.94506957162;

const ω_b: fp = ω_r;
//const Δ_s:  [fp; 2] = [Δ_s_0+0., Δ_s_0-0.]; // |ω_r - ω_s|
const Δ_b: [fp; 2] = [1000.0, 1000.0]; //  ω_b - ω_s
const Δ_br: fp = 0.0; // ω_b - ω_r
const Δ_r: fp = 0.0;
const χ: [fp; 2] = [χ_0 + 0.00, χ_0 - 0.00];
const g: [fp; 2] = [g_0, g_0];

const NOISE_FACTOR: fp = 1.0;

#[derive(Clone)]
struct QubitSystem {
    ρ: Operator,
    c_out_phased: Lindblad,
    c1: Lindblad,
    c2: Lindblad,
    c3: [Lindblad; 2],
    dZ: Complex,
}

#[inline(always)]
fn commutator(a: &Operator, b: &Operator) -> Operator {
    a*b - b*a
}

#[inline]
fn anticommutator(a: Operator, b: Operator) -> Operator {
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

fn anti_tensor_commutator(lhs: &Matrix, rhs: &Matrix) -> Matrix {
    lhs.kronecker(rhs) + rhs.kronecker(lhs)
}

fn apply_individually_parts(op: &Matrix) -> [Operator; Operator::QUBIT_COUNT] {
    let identity = Matrix::identity(2);

    // Create [[M,I], [I,M]].
    let mut parts = [[&identity; Operator::QUBIT_COUNT]; Operator::QUBIT_COUNT];
    (0..Operator::QUBIT_COUNT).for_each(|i| parts[i][i] = op );

    // Combine into [M ⊗ I, I ⊗ M].
    parts.map(|mp| {
        mp.iter().skip(1).fold(mp[0].clone(), |acc, m| acc.kronecker(m)).to_operator()
    })
}

fn apply_individually(op: &Matrix) -> Operator {
    apply_individually_parts(op).iter().fold(Operator::zero(), |sum, x| &sum+x)
}

fn apply_and_scale_individually<T>(factors: [T; Operator::QUBIT_COUNT], op: &Matrix) -> Operator
    where T: std::ops::Mul<Operator, Output = Operator>
{
    apply_individually_parts(op).iter()
        .zip(factors).fold(Operator::zero(), |sum, (part, factor)| sum + factor * (*part) )
}

impl QubitSystem {

    fn new(initial_probabilities: [fp; Operator::SIZE]) -> (Self, QuantumCircuit) {
        #[allow(unused_variables)]
        let σ_plus   = Matrix::new(0.0, 1.0, 0.0, 0.0);
        let σ_z      = Matrix::new(1.0, 0.0, 0.0, -1.0);
        let σ_minus  = Matrix::new(0.0, 0.0, 1.0, 0.0);
        let hadamard = Matrix::new(1.0, 1.0, 1.0, -1.0);
        let identity = Operator::identity();

        let χσ_z = apply_and_scale_individually(χ, &σ_z);

        let sqrt2κ1 = fp::sqrt(2.0*κ_1);

        // DISPERSIVE
        let a = Operator::from_fn(|r, c| {
            (r==c) as u32 as fp
                * (2.*κ_1).sqrt() * β
                / (Δ_r + κ + I*χ.iter().enumerate().fold(0.0, |acc, (n, χ_n)| {
                    acc - (((c>>n)&1) as i32 * 2 - 1) as fp * χ_n // TODO: Check this
            }))
        });

        let N = a.dagger() * a;

        // Hamiltonian
        let term2 = apply_and_scale_individually(Δ_b.zip(χ).map(|(Δ_bs, χ_s)| 0.5*Δ_bs*identity + χ_s*N ), &σ_z);

        let mut term3 = Operator::zero();

        let σ_m = apply_individually_parts(&σ_minus);
        let σ_p = apply_individually_parts(&σ_plus);

        for s in 0..Operator::QUBIT_COUNT {
            for q in 0..Operator::QUBIT_COUNT {
                let c = χ[s] * (g[q]/g[s]);
                term3 += c * σ_p[s] * σ_m[q] + c * σ_p[q] * σ_m[s];
            }
        }

        let H = Δ_br * N + term2 + term3
            + I*(2.0 * κ_1).sqrt() * (β * a.dagger() - β.conjugate() * a); // Detuning

        // CNOT 0, 1
        //let hamiltonian = hamiltonian
        //    + omega * (ket(&one)*bra(&one)).kronecker(&(ket(&one)*bra(&zero) + ket(&zero)*bra(&one))).to_operator();

        // Remove any non-hermitian numerical error
        let H = 0.5*(H + H.conjugate());

        // Construct initial state
        let p_sum = initial_probabilities.iter().fold(0.0, |acc, x| acc+x);
        let ψ = Matrix::vector(&initial_probabilities.map(|p| (p/p_sum).sqrt() ));
        let ρ = (&ψ * ψ.transpose()).to_operator();
        println!("Initial ρ:\n{ρ}");

        let c_out = (κ_1 * 2.0).sqrt() * a - β * identity;
        let c_out_phased = Lindblad::new(c_out * ((I * Φ).exp()));
        let c1 = Lindblad::new((2.0 * κ).sqrt() * a);
        let c2 = Lindblad::new(apply_individually(&(γ_dec.sqrt() * &σ_minus)));
        let c3 = apply_individually_parts(&((γ_φ / 2.0).sqrt() * &σ_z)).map(Lindblad::new);

        let sqrt_η = η.sqrt();
        let dχρ = sqrt_η * (c_out_phased.c + c_out_phased.c.adjoint());

        println!("H:\n{H}");

        let omega = 1.0;
        let hadamard_parts = apply_individually_parts(&((1.0/SQRT_2 as f64)*hadamard));
        let H1 = H + omega * hadamard_parts[0];
        let H2 = H + omega * hadamard_parts[1];

        let circuit = QuantumCircuit::new(&[
            //(H1, 0.000001/SQRT_2),
            (H, Δt * (STEP_COUNT-1) as fp)
        ]);

        (
            Self {
                ρ,
                c_out_phased,
                c1,
                c2,
                c3,
                dZ: Complex::new(0.0, 0.0),
            },
            circuit
        )
    }

    fn lindblad_terms(&self, ρ: &Operator) -> Operator {
        *Operator::lindblad_term(ρ, &self.c1) // Photon field transmission/losses
            // .lindblad(ρ, &self.c2) // Decay to ground state
            .lindblad(ρ, &self.c3[0])
            .lindblad(ρ, &self.c3[1])
            .lindblad(ρ, &self.c_out_phased)
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

#[derive(Clone, Copy)]
struct MeasurementRecords {
    measurements: [[Complex; STEP_COUNT as usize]; SIMULATION_COUNT as usize]
}

fn simulate<const CUSTOM_RECORDS: bool, const RETURN_RECORDS: bool, const WRITE_FILES: bool>
    (initial_probabilities: [fp; Operator::SIZE], records: Option<Arc<MeasurementRecords>>) -> Option<Box<MeasurementRecords>> {

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

    // FIXME
    parameter_file
        .write_all(
            format!(
"let β = {β};
let Δ_r = {Δ_r};
let κ = {κ};
let κ_1 = {κ_1};
let η = {η};
let Φ = {Φ};
let γ_dec = {γ_dec};
let γ_φ = {γ_φ};
").as_bytes(),
        ).unwrap();

    // metadata
    current_file.write_all(&(SIMULATION_COUNT * Real::LANES as u32).to_le_bytes()).unwrap();

    //current_file.write(&STEP_COUNT.to_le_bytes()).unwrap();

    #[cfg(not(feature = "double-precision"))]
    data_file.write_all(&(0u32.to_le_bytes())).unwrap();
    #[cfg(feature="double-precision")]
    data_file.write_all(&(1u32.to_le_bytes())).unwrap();

    data_file.write_all(&(SIMULATION_COUNT * Real::LANES as u32).to_le_bytes()).unwrap();
    data_file.write_all(&(Operator::SIZE as u32).to_le_bytes()).unwrap();
    data_file.write_all(&(STEP_COUNT + 1).to_le_bytes()).unwrap();

    hist_file.write_all(&(Operator::SIZE as u32).to_le_bytes()).unwrap();
    hist_file.write_all(&(STEP_COUNT + 1).to_le_bytes()).unwrap();
    hist_file.write_all(&(HIST_BIN_COUNT as u32 * Operator::SIZE as u32).to_le_bytes()).unwrap();

    struct ThreadResult {
        trajectory_sum:  [StateProbabilitiesSimd; STEP_COUNT as usize+1],
        trajectory_hist: [[Histogram<HIST_BIN_COUNT>; Operator::SIZE]; STEP_COUNT as usize+1],
        current_sum: [Complex; SIMULATIONS_PER_THREAD as usize],
        final_states: [StateProbabilitiesSimd; SIMULATIONS_PER_THREAD as usize],
        measurements: Option<Box<[[Complex; STEP_COUNT as usize]; SIMULATIONS_PER_THREAD as usize]>>
    }

    let threads: Vec<_> = (0..THREAD_COUNT).map(|thread_id| {
        let input_records_arc = records.clone();
        std::thread::spawn( move || {

        let mut local = alloc_zero!(ThreadResult);
        let mut measurements = alloc_zero!([[Complex; STEP_COUNT as usize]; SIMULATIONS_PER_THREAD as usize]);
        let input_records = &input_records_arc;

        // Start the timer.
        let now = std::time::Instant::now();
        let mut total_section_time = 0;

        // Create initial system.
        let (initial_system, circuit) = QubitSystem::new(initial_probabilities);

        // RNG
        let mut rng = thread_rng();

        // S generator
        let mut S = SGenerator::new(&mut rng);

        // sqrt(η/2) is from the definition of dZ.
        // sqrt(dt) is to make the variance σ = dt
        let sqrtηdt = Real::splat((η * 0.5).sqrt() * Δt.sqrt() * NOISE_FACTOR);
        println!("sqrtηdt: {sqrtηdt:?}");



        for simulation in 0..SIMULATIONS_PER_THREAD {
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

            // Do 2000 steps.
            for step in 0..STEP_COUNT as usize {

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
                    let dI = input_records.as_ref().unwrap().measurements[thread_id as usize * SIMULATIONS_PER_THREAD as usize + simulation as usize][step];

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

                // Normalize rho.
                system.ρ.normalize();

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
            local.trajectory_sum[STEP_COUNT as usize].add(&P);
            for (i, p) in P.v.iter().enumerate() {
                local.trajectory_hist[STEP_COUNT as usize][i].add_values(p);
            }

            local.current_sum[simulation as usize] = J;
            local.final_states[simulation as usize] = P;
        }

        let total_time = now.elapsed().as_millis();

        println!("Thread {thread_id} finished {SIMULATIONS_PER_THREAD} simulations in {total_time} ms ({} μs/sim) (total section time {} ms)",
                 (total_time*1000)/(SIMULATIONS_PER_THREAD*Real::LANES as u32) as u128,
                 total_section_time/1000
        );

        for s in local.trajectory_sum.iter_mut() {
            s.divide(SIMULATIONS_PER_THREAD as fp);
        }

        local.measurements = if RETURN_RECORDS { Some(measurements) } else { None };

        local
    })}).collect();

    let mut trajectory_average = vec![StateProbabilitiesSimd::zero(); (STEP_COUNT+1) as usize].into_boxed_slice();
    let mut trajectory_histograms = alloc_zero!([[Histogram::<HIST_BIN_COUNT>; Operator::SIZE]; STEP_COUNT as usize+1]);
    let mut measurements = alloc_zero!(MeasurementRecords);

    // Wait for threads

    for (i, tt) in threads.into_iter().enumerate() {
        let local = tt.join().unwrap();
        for (s, ls) in trajectory_average.iter_mut().zip(local.trajectory_sum.iter()) {
            s.add(ls);
        }

        for (histograms, local_histograms) in trajectory_histograms.iter_mut().zip(local.trajectory_hist.iter()) {
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
            let a = i * SIMULATIONS_PER_THREAD as usize;
            let b = (i+1) * SIMULATIONS_PER_THREAD as usize;
            measurements.measurements[a..b].copy_from_slice(&m[..]);
        }
    }

    for s in trajectory_average.iter_mut() {
        s.divide(THREAD_COUNT as fp);
        data_file.write_all(&s.average().to_le_bytes()).unwrap();
    }

    // TODO: actual time
    // TODO: fix magic number (simcount)
    for i in 0..=STEP_COUNT {
        let t = i as fp * Δt;
        data_file.write_all(&t.to_le_bytes()).unwrap();
    }

    let mut buffer = Vec::with_capacity(std::mem::size_of::<u32>() * HIST_BIN_COUNT * Operator::SIZE * STEP_COUNT as usize);
    for state in trajectory_histograms.iter() {
        for hist in state.iter() {
            for bin in hist.bins.iter().rev() {
                buffer.extend_from_slice(&bin.to_le_bytes());
            }
        }
    }
    hist_file.write_all(&buffer).unwrap();

    if RETURN_RECORDS { Some(measurements) } else { None }
}

fn bloch_vector(rho: &Operator) -> [fp; 3] {
    const OFFSET: usize = 0;
    [
        rho[(OFFSET, 1 + OFFSET)].real()[0] + rho[(1 + OFFSET, 0 + OFFSET)].real()[0],
        rho[(OFFSET, 1 + OFFSET)].imag()[0] - rho[(1 + OFFSET, 0 + OFFSET)].imag()[0],
        rho[(OFFSET, 0 + OFFSET)].real()[0] - rho[(1 + OFFSET, 1 + OFFSET)].real()[0],
    ]
}


fn simple() {
    simulate::<false, false, true>(INITIAL_PROBABILITIES, None);
}

fn feed_current_known_state () {
    // Current of known state of |11>
    let a = simulate::<false, true, false>([0.0,0.0,0.0,1.0], None);

    // Feed current into perfect superposition.
    let b = simulate::<true, false, true>([0.25,0.25,0.25,0.25], Some(a.unwrap().into()));
}

fn main() {
    println!("Starting simulation...");

    #[cfg(feature="double-precision")]
    println!("DOUBLE PRECISION");

    let start = std::time::Instant::now();
    feed_current_known_state();
    let elapsed = start.elapsed().as_millis();

    println!(
        "Simulations took {elapsed} ms ({} sim/s, {} steps/s)",
        (1000*SIMULATION_COUNT*Real::LANES as u32) as fp / elapsed as fp,
        (1000*SIMULATION_COUNT as u64 * STEP_COUNT as u64 * Real::LANES as u64) as u128 / elapsed
    );
}
