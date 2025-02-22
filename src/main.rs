#![feature(portable_simd)]
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
mod system;
use system::*;
mod simulation;
use simulation::*;

use rand_distr::StandardNormal;
use std::{io::Write, sync::Arc, simd::{num::SimdFloat, StdFloat}};

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
pub const Δt: fp = 0.000000025;

const DEFAULT_CONFIG: SimulationConfig = SimulationConfig {
    step_count: 628,
    thread_count: 10,
    simulations_per_thread: 100,
    silent: false,

    fidelity_probe: None,
    measure_purity: false,

    // Physical constants
    κ: 0.5,
    κ_1: 0.5, // NOTE:
    β: Complex::new(4.0, 0.0), // Max value is kappa
    γ_dec: 563.9773943, // should be g^2/ω    (174) side 49
    η: 0.96,
    Φ: 0.0,    // c_out phase shift Phi
    γ_φ: 0.001,
    g_0: 10.0, // sqrt(Δ_s_0 χ)
    χ_0: 0.6,
    ω_r: 2500.0,

    Δ_r: 0.0,
    Δ_br: 10.0, // ω_b - ω_r

    chi_offsets: None
};

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

fn fidelity_data() {

    pub fn get_ideal_ρ(initial_state: &InitialState) -> Matrix {
        let initial_state: Operator = initial_state.clone().into();

        let σ_x = Matrix::new(0.0, 1.0, 1.0, 0.0);
        let applied = apply_individually_parts(&σ_x)[0];
        // let applied = σ_x.kronecker(&Matrix::identity(2)).to_operator();
        //let applied = &σ_x;
        let m = applied*initial_state*applied;
        let (m1,m2,m3,m4) = m.partial_trace();
        Matrix::new(
            m1.first().0 as f64,
            m2.first().0 as f64,
            m3.first().0 as f64,
            m4.first().0 as f64
        )
    }

    let initial_state = InitialState::Probabilites(INITIAL_PROBABILITIES.to_vec());
    //let ideal_ρ = get_ideal_ρ(&initial_state);
    let m = get_ideal_ρ(&initial_state);
    let ideal_ρ = (
        m[(0,0)].to_complex(),
        m[(0,1)].to_complex(),
        m[(1,0)].to_complex(),
        m[(1,1)].to_complex()
    );
    let ideal_ρ = m.to_operator();

    let mut conf = DEFAULT_CONFIG;
    conf.simulations_per_thread = 500;
    conf.thread_count = 16;
    //conf.step_count = 4000;
    conf.step_count = (3.45e-5 / Δt) as u32; //pi/(2×500×0.0000025)
    conf.silent = true;
    conf.fidelity_probe = Some((0.00314, IdealState::Full(ideal_ρ))); // 2x2
    conf.χ_0 = 0.6;
    conf.κ = 0.2;
    conf.κ_1 = 0.2;

    let mut csv_data = Vec::new();

    for i in 0..=200 {
        for j in 8..=8 {
            //    conf.χ_0 = i as fp / 5.0;
            conf.β = Complex::new(30.0*j as fp, 0.0);
            let offset = i as fp / 100.0 - conf.χ_0 + 0.0075;
            let chi = conf.χ_0 + offset;
            conf.chi_offsets = Some(vec![ChiOffset { index: 0, offset }]);

            let simulation_results = simulate::<false, false, false>(initial_state.clone(), &conf, None);
            let fidelities = simulation_results.fidelity_probe_results.as_ref().unwrap();
            let zero = Real::splat(0.0);
            let n = (fidelities.len() * Real::LEN) as fp;

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
            csv_data.push((i, chi, conf.β, mean, std));

            println!("Completed [i:{i}/20, j:{j}/20], χ: {chi}, β: {}, μ: {mean}, std: {std}", conf.β.real[0]);
            //panic!();
        }
    }

    // Write results
    let mut csv = std::fs::File::create("fidelity_data.csv").unwrap();
    csv.write_all(b"index, chi, beta, mean, standard_deviation\n").unwrap();
    csv_data.iter().for_each(|(i,χ,β,μ,s)| csv.write_all(format!("{i},{χ},{},{μ},{s}\n", β.real[0]).as_bytes()).unwrap() );
}

fn purity_means_and_std(offsets: Option<Vec<ChiOffset>>) -> (Vec<fp>, Vec<fp>) {
    let mut conf = DEFAULT_CONFIG;
    conf.silent = true;
    conf.simulations_per_thread = 30;
    conf.thread_count = 10;
    conf.measure_purity = true;
    conf.step_count = 1000;
    conf.χ_0 = 0.6;
    conf.β = C!(125.0);
    conf.η = 0.95;
    conf.chi_offsets = offsets;

    let ρ_initial = Operator::from_fn(|r,c| C!((r==c) as i32 as fp / Operator::SIZE as fp));
    //let ρ_initial = Operator::from_fn(|r,c| C!((r==0 && c==0) as i32 as fp));
    let mut results = simulate::<false, false, false>(InitialState::DensityMatrix(ρ_initial), &conf, None);
    let purities = results.purity_results.as_mut().unwrap();
    purities.transpose();
    let z = Real::splat(0.0);
    let n = (Real::LEN * conf.simulation_count()) as fp;

    // Compute averages.
    let mean: Vec<fp> = (0..conf.step_count as usize)
        .map(|i| purities[i].iter()
             .fold(z, |acc, x| { acc+x })
             .reduce_sum() / n)
        .collect();

    // Compute standard deviations
    let std: Vec<fp> = (0..conf.step_count as usize)
        .map(|step| purities[step].iter()
             .fold(z, |acc, p| {
                 let delta = p - Real::splat(mean[step]);
                 acc + delta*delta
             }))
        .map(|r| r.reduce_sum())
        .map(|s| s / (n-1.0))
        .map(|x| x.sqrt())
        .collect();

    (mean, std)
}


fn purity_data() {
    let (mean, std) = purity_means_and_std(None);

    let mut csv = std::fs::File::create("purity_data.csv").unwrap();
    csv.write_all(b"t, mean, standard_deviation\n").unwrap();
    for i in 0..mean.len() {
        let t = i as fp * Δt;
        csv.write_all(format!("{t}, {}, {}\n", mean[i], std[i]).as_bytes()).unwrap();
    }
}


fn purity_data_varying_chi() {
    let mut csv_data = Vec::new();

    for i in 0..250 {
        let offset = i as fp * 0.0008/5.0;
        let offsets = Some(vec![
            ChiOffset { index: 0, offset: -offset/2.0 },
            ChiOffset { index: 1, offset:  offset/2.0 }
        ]);
        let (mean, std) = purity_means_and_std(offsets);
        let (max_mean, max_std) = mean.into_iter().zip(std).max_by(|(m1,_), (m2,_)| m1.partial_cmp(m2).unwrap()).unwrap();

        csv_data.push((offset, max_mean, max_std));
        println!("Completed [{i}/250]");
    }

    let mut csv = std::fs::File::create("purity_data_varying_chi.csv").unwrap();
    csv.write_all(b"offset, mean, standard_deviation\n").unwrap();
    csv_data.iter().for_each(|(δ,μ,s)| csv.write_all(format!("{δ},{μ},{s}\n").as_bytes()).unwrap() );
}


fn simple() {
    let mut conf = DEFAULT_CONFIG;
    let initial_state = InitialState::Probabilites(vec![
        0.25,0.375,0.125,0.25
    ]);
    //conf.Φ = 1.0;

    let offsets = Some(vec![
        ChiOffset { index: 0, offset: -0.1 },
        ChiOffset { index: 1, offset:  0.1 }
    ]);

    conf.step_count = 2500;
    conf.simulations_per_thread = 100;
    conf.chi_offsets = offsets;

    let start = std::time::Instant::now();
    simulate::<false, false, true>(initial_state, &conf, None);
    let elapsed = start.elapsed().as_millis();
    println!(
        "Simulations took {elapsed} ms ({} sim/s, {} steps/s)",
        (1000*DEFAULT_CONFIG.simulation_count() as u32 * Real::LEN as u32) as fp / elapsed as fp,
        (1000*DEFAULT_CONFIG.simulation_count() as u64 * DEFAULT_CONFIG.step_count as u64 * Real::LEN as u64) as u128 / elapsed
    );
}

fn feed_current_known_state () {
    let mut config = DEFAULT_CONFIG;
    config.step_count = 10000;
    config.simulations_per_thread = 1;
    config.thread_count = 1;
    config.β = C!(10.0);

    // Current of known state of |11>
    let a = simulate::<false, true, true>(InitialState::Probabilites(vec![0.25, 0.15, 0.35, 0.25]), &config, None);

    // Sleep 1 second to not overwrite last files
    std::thread::sleep(std::time::Duration::from_secs(1));

    let initial_ρ = *Operator::identity().normalize();

    // Feed current into perfect superposition.
    let b = simulate::<true, false, true>(InitialState::DensityMatrix(initial_ρ), &config, Some(&a));
}

fn main() {
    println!("Starting simulation...");

    #[cfg(feature="double-precision")]
    println!("DOUBLE PRECISION");

    //feed_current_known_state();
    simple();
    //fidelity_data();
    //purity_data();
    //purity_data_varying_chi()


}
