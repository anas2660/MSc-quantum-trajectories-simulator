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
mod system;
use system::*;
mod simulation;
use simulation::*;

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
    χ_0: 0.1,
    ω_r: 2500.0,

    Δ_r: 0.0,
    Δ_br: 10.0, // ω_b - ω_r
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
    let mut conf = DEFAULT_CONFIG;
    conf.simulations_per_thread = 750;
    conf.silent = true;
    conf.fidelity_probe = Some(0.00314);

    let mut csv_data = Vec::new();

    for i in 0..=20 {
        for j in 0..=20 {
            conf.χ_0 = i as fp / 5.0;
            conf.β = Complex::new(j as fp, 0.0);
            let simulation_results = simulate::<false, false, false>(InitialState::Probabilites(INITIAL_PROBABILITIES.to_vec()), &conf, None);
            let fidelities = simulation_results.fidelity_probe_results.as_ref().unwrap();
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


fn purity_data() {
    let mut conf = DEFAULT_CONFIG;
    // conf.silent = true;
    conf.simulations_per_thread = 50;
    conf.thread_count = 10;
    conf.measure_purity = true;
    conf.step_count = 2000;
    conf.χ_0 = 0.6;
    conf.β = C!(100.0);
    conf.η = 0.95;


    let ρ_initial = Operator::from_fn(|r,c| C!((r==c) as i32 as fp / Operator::SIZE as fp));
    //let ρ_initial = Operator::from_fn(|r,c| C!((r==0 && c==0) as i32 as fp));

    let mut results = simulate::<false, false, true>(InitialState::DensityMatrix(ρ_initial), &conf, None);
    let purities = results.purity_results.as_mut().unwrap();
    purities.transpose();
    let z = Real::splat(0.0);
    let n = (Real::LANES * conf.simulation_count()) as fp;

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


    let mut csv = std::fs::File::create("purity_data.csv").unwrap();
    csv.write_all(b"t, mean, standard_deviation\n").unwrap();
    for i in 0..conf.step_count as usize {
        let t = i as fp * Δt;
        csv.write_all(format!("{t}, {}, {}\n", mean[i], std[i]).as_bytes()).unwrap();
    }
    //csv_data.iter().for_each(|(i,χ,β,μ,s)| csv.write_all(format!("{i},{χ},{},{μ},{s}\n", β.real[0]).as_bytes()).unwrap() );
}


fn simple() {
    let start = std::time::Instant::now();
    simulate::<false, false, true>(InitialState::Probabilites(INITIAL_PROBABILITIES.to_vec()), &DEFAULT_CONFIG, None);
    let elapsed = start.elapsed().as_millis();
    println!(
        "Simulations took {elapsed} ms ({} sim/s, {} steps/s)",
        (1000*DEFAULT_CONFIG.simulation_count() as u32 * Real::LANES as u32) as fp / elapsed as fp,
        (1000*DEFAULT_CONFIG.simulation_count() as u64 * DEFAULT_CONFIG.step_count as u64 * Real::LANES as u64) as u128 / elapsed
    );
}

fn feed_current_known_state () {
    // Current of known state of |11>
    let a = simulate::<false, true, false>(InitialState::Probabilites(vec![0.0, 0.45, 0.55, 0.0]), &DEFAULT_CONFIG, None);

    // Feed current into perfect superposition.
    let b = simulate::<true, false, true>(InitialState::Probabilites(vec![0.25, 0.25, 0.25, 0.25]), &DEFAULT_CONFIG, Some(&a));
}

fn main() {
    println!("Starting simulation...");

    #[cfg(feature="double-precision")]
    println!("DOUBLE PRECISION");

    //feed_current_known_state();
    //simple();
    //fidelity_data();
    purity_data();


}
