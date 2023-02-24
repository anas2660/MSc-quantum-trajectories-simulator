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

use rand_distr::StandardNormal;
use std::io::Write;

use rand::{rngs::ThreadRng, thread_rng, Rng};

mod pipewriter;
// use pipewriter::*;

const ONE:     cfp = Complex::new(1.0, 0.0);
const I:       cfp = Complex::new(0.0, 1.0);
const MINUS_I: cfp = Complex::new(0.0, -1.0);
const ZERO:    cfp = Complex::new(0.0, 0.0);

// Initial state probabilities
const initial_probabilities: [fp; Operator::SIZE] = [
    //0.0, // 00
    //0.45, // 01
    //0.55, // 10
    //0.00  // 11
    0.0, // 00
    0.00, // 01
    0.99, // 10
    0.01  // 11
];

// Simulation constants
const Δt: fp = 0.1;
const STEP_COUNT: u32 = 2000;
const THREAD_COUNT: u32 = 8;
const HIST_BIN_COUNT: usize = 32;
const SIMULATIONS_PER_THREAD: u32 = 40;
const SIMULATION_COUNT: u32 = THREAD_COUNT * SIMULATIONS_PER_THREAD;

// Physical constants
const κ:     fp = 1.2;
const κ_1:   fp = 1.2; // NOTE: Max value is the value of kappa. This is for the purpose of loss between emission and measurement.
const β:    cfp = Complex::new(1.50, 0.0); // Max value is kappa
const γ_dec: fp = 1.0;
const η:     fp = 1.500000;
const Φ:     fp = 0.0; // c_out phase shift Phi
const γ_φ:   fp = 0.001;
//const ddelta: fp = delta_r - delta_s;
const χ_0:   fp = 0.6;
//const g:     fp = 125.6637061435917 / 1.0;
const ω_r:   fp = 28368.582-2049.0;
const ω_s_0: fp = 2049.6365-2049.0;
//const Δ_s_0: fp = 26318.94506957162 / 1000000.0;

const ω_b:       fp = 0.1;
//const Δ_s:  [fp; 2] = [Δ_s_0+0., Δ_s_0-0.]; // |ω_r - ω_s|
const Δ_b:  [fp; 2] = [ω_s_0+0., ω_s_0-0.]; //  ω_b - ω_s
const Δ_br:      fp = ω_r; // ω_b - ω_r
const Δ_r:       fp = ω_r;
const χ:    [fp; 2] = [χ_0; 2];

#[derive(Clone)]
struct QubitSystem {
    ρ: Operator,
    Y: cfp,
    //t: fp
    sqrt_η: fp,
    c_out_phased: Operator,
    dχρ: Operator,
    rng: ThreadRng,
    measurement: Operator,
    c1: Operator,
    c2: Operator,
    c3: [Operator; 2],
    dZ: Complex,
}


struct SGenerator {
    v: u64,
    index: u32
}

impl SGenerator {
    fn new(rng: &mut ThreadRng) -> Self {
        Self { v: rng.gen(), index: 0 }
    }

    fn gen(&mut self, rng: &mut ThreadRng) -> i32 {
        let result = (((self.v >> self.index) & 1) << 1) as i32 - 1;
        self.index += 1;
        if self.index > 64 {
            *self = SGenerator::new(rng);
        }
        result
    }
}

#[inline]
fn commutator(a: &Operator, b: &Operator) -> Operator {
    &(a * b) - &(b * a)
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
    fn new() -> (Self, QuantumCircuit) {
        #[allow(unused_variables)]
        let σ_plus   = Matrix::new(0.0, 1.0, 0.0, 0.0);
        let σ_z      = Matrix::new(1.0, 0.0, 0.0, -1.0);
        let σ_minus  = Matrix::new(0.0, 0.0, 1.0, 0.0);
        let hadamard = Matrix::new(1.0, 1.0, 1.0, -1.0);
        let identity = Operator::identity();

        let χσ_z = apply_and_scale_individually(χ, &σ_z);

        let a = Operator::from_fn(|r, c| {
            (r==c) as u32 as fp
                * (2.*κ_1).sqrt() * β
                / (Δ_r + κ + I*χ.iter().enumerate().fold(0.0, |acc, (n, χ_n)| {
                    acc - (((c>>n)&1) as i32 * 2 - 1) as fp * χ_n // TODO: Check this
            }))
        });

        let N = a.dagger() * a;

        // Hamiltonian
        let factors: [Operator; Operator::QUBIT_COUNT] = Δ_b.zip(χ).map(|(Δ_bs, χ_s)| {
            0.5*Δ_bs*identity + χ_s*N
        });

        let sum_σ_minus = apply_individually(&σ_minus);
        let term3 = apply_individually_parts(&σ_plus).iter().zip(χ).fold(Operator::zero(), |sum, (σ_ps, χ_s)| {
            sum + χ_s * σ_ps * sum_σ_minus
        });

        let H = Δ_br * N
            + apply_and_scale_individually(factors, &σ_z)
            + term3
            + I*(2.0 * κ_1).sqrt() * (β * a.dagger() - β.conjugate() * a); // Detuning
            //+ 0.5 * apply_and_scale_individually(Δ_b, &σ_z)
            //+ (N + 0.5*identity) * χσ_z
            //+ 0.5 * omega * apply_individually(&(&sigma_plus + &sigma_minus));

        //let hamiltonian = omega_r * a.dagger() * a + (omega_s + 2.0 * (g*g/delta_s) * (a.dagger()*a + 0.5*Operator::identity()));

        // CNOT 0, 1
        //let hamiltonian = hamiltonian
        //    + omega * (ket(&one)*bra(&one)).kronecker(&(ket(&one)*bra(&zero) + ket(&zero)*bra(&one))).to_operator();

        // Remove any non-hermitian numerical error
        let H = &H+(H.conjugate()-H).scale(0.5);

        // Construct initial state
        let ψ = Matrix::vector(&initial_probabilities.map(|p| p.sqrt() ));
        let ρ = (&ψ * ψ.transpose()).to_operator();

        let c_out = (κ_1 * 2.0).sqrt() * a - β * identity;
        let c1 = (2.0 * κ).sqrt() * a;
        let c2 = γ_dec.sqrt() * &σ_minus;//sigma_minus;
        let c3 = (γ_φ / 2.0).sqrt() * &σ_z;

        let identity = Matrix::identity(2);

        //let c_out = (c_out.kronecker(&identity) + identity.kronecker(&c_out)).to_operator();
        //let c1 = (c1.kronecker(&identity) + identity.kronecker(&c1)).to_operator();
        let c2 = (c2.kronecker(&identity) + identity.kronecker(&c2)).to_operator();
        let c3 = apply_individually_parts(&c3);

        let sqrt_η = η.sqrt();
        let c_out_phased = c_out * ((I * Φ).exp());
        let dχρ = sqrt_η * (c_out_phased + c_out_phased.adjoint());

        println!("H:\n{H}");

        let omega = 1.0;
        let hadamard_parts = apply_individually_parts(&((1.0/SQRT_2)*hadamard));
        let H1 = H + omega * hadamard_parts[0];
        let H2 = H + omega * hadamard_parts[1];

        let circuit = QuantumCircuit::new(&[
            //(H1, 0.000001/SQRT_2),
            (H, 0.0001)
        ]);

        (
            Self {
                ρ,
                Y: ZERO,
                sqrt_η,
                c_out_phased,
                dχρ,
                rng: thread_rng(),
                measurement: Operator::from_fn(|r, c| ONE * (r * c) as fp),
                c1,
                c2,
                c3,
                dZ: Complex::new(0.0, 0.0),
            },
            circuit
        )
    }

    fn deterministic(&self, H: &Operator, ρ: &Operator) -> Operator {
        *commutator(H, ρ)
                .scale(&MINUS_I)
                ////+ self.lindblad(a)
                // NOTE: Are these not missing a ρ factor?
                .lindblad(ρ, &self.c1) // Photon field transmission/losses
                //////+ self.lindblad(&self.c2) // Decay to ground state
                .lindblad(ρ, &self.c3[0]).lindblad(ρ, &self.c3[1])
                .lindblad(ρ, &self.c_out_phased) // c_out
    }

    fn stochastic(&self, H: &Operator, ρ: &Operator) -> ([Operator; 2], [Complex; 2]) {
        let cop = &self.c_out_phased;
        let cop_dagger = cop.dagger();

        let u = [ρ * cop_dagger, cop*ρ];
        let v = [-u[0].trace(), u[1].trace()];
        let t = [v[0]*ρ + u[0], v[1]*ρ + u[1]];

        let tp = [
            v[0] - cop_dagger*(ρ.sub_identity()),
            v[1] + (&u[1] - cop)
        ];

        (
            [t[0]-t[1],   (t[0]+t[1]).mul_i()],
            [
                -((cop+&cop_dagger)*ρ).trace(),
                I*((cop-&cop_dagger)*ρ).trace()
            ]
        )

        //(
        //    [t[0]-t[1],   (t[0]+t[1]).mul_i()],
        //    [tp[0]-tp[1], (tp[0]+tp[1]).mul_i()]
        //)
    }

    fn millstein(&mut self, H: &Operator) {
        //let a = self.deterministic(H, &self.ρ);
        let adt = self.runge_kutta(H);
        let (b, b_prime) = self.stochastic(H, &self.ρ);
        let ΔWx = self.dZ.real;
        let ΔWy = self.dZ.imag;
        let Δtv = V::splat(Δt);
        self.ρ += adt //a * Δt
            + b[0]*ΔWx + 0.5*b[0]*b_prime[0]*(ΔWx*ΔWx - Δtv)
            + b[1]*ΔWy + 0.5*b[1]*b_prime[1]*(ΔWy*ΔWy - Δtv)
    }


    fn runge_kutta(&mut self, H: &Operator) -> Operator {
        let k0 = self.deterministic(H, &self.ρ);
        let k1 = self.deterministic(H, (self.ρ + 0.5 * Δt * k0).normalize());
        let k2 = self.deterministic(H, (self.ρ + 0.5 * Δt * k1).normalize());
        let k3 = self.deterministic(H, (self.ρ + Δt * k2).normalize());
        (Δt / 3.0) * (k1 + k2 + 0.5 * (k0 + k3))
        //self.Y += t3 * (dY1 + dY2 + 0.5 * (dY0 + dY3));
    }


    fn stochastic2(&self, H: &Operator, ρ: &Operator) -> [Operator; 2] {
        #[inline]
        fn Hcalρ(r: &Operator, ρ: &Operator) -> Operator {
            let c = r*ρ + ρ*r.dagger();
            c - c.trace()*ρ
        }
        [
            FRAC_1_SQRT_2*Hcalρ(&self.c_out_phased, ρ),
            FRAC_1_SQRT_2*Hcalρ(&(MINUS_I*self.c_out_phased), ρ)
        ]
    }

    fn srk2(&mut self, H: &Operator) {
        let δ: fp = Δt;
        let sqrtδ = δ.sqrt();
        let term1 = self.runge_kutta(H);
        //let term1 = self.deterministic(H,ρ)*δ;
        let ρ = &self.ρ;
        let b = self.stochastic2(H, &self.ρ);
        let Ycal = ρ + &(term1) + b[0]*sqrtδ + b[1]*sqrtδ;
        let bcal = self.stochastic2(H, &Ycal);

        let ΔWx = self.dZ.real;
        let ΔWy = self.dZ.imag;
        //let one_over_2sqrtδ = 1.0/(2.0*sqrtδ);
        let one_over_2sqrtδ = 1.0/(2.0*sqrtδ);
        let δ = V::splat(δ);
        self.ρ += term1
            + (b[0]*ΔWx + b[1]*ΔWy)
            + one_over_2sqrtδ*(bcal[0]-b[0])*(ΔWx*ΔWx - δ)
            + one_over_2sqrtδ*(bcal[1]-b[1])*(ΔWy*ΔWy - δ)
    }

    //fn euler(&mut self, H: &Operator) {
    //    let a = self.deterministic(H, &self.ρ);
    //    let (b, b_prime) = self.stochastic(H, &self.ρ);
    //    self.ρ += a * Δt + b;
    //}

    //fn runge_kutta(&mut self, H: &Operator) {
    //    let (k0, dY0) = self.dv(H, &self.ρ);
    //    let (k1, dY1) = self.dv(H, (self.ρ + 0.5 * Δt * k0).normalize());
    //    let (k2, dY2) = self.dv(H, (self.ρ + 0.5 * Δt * k1).normalize());
    //    let (k3, dY3) = self.dv(H, (self.ρ + Δt * k2).normalize());
    //    self.ρ += (Δt / 3.0) * (k1 + k2 + 0.5 * (k0 + k3));
    //    //self.Y += t3 * (dY1 + dY2 + 0.5 * (dY0 + dY3));
    //}

    //fn euler(&mut self, H: &Operator) {
    //    self.ρ += self.dv(H, &self.ρ).0 * Δt;
    //}


}

fn simulate() {
    let timestamp = std::time::SystemTime::UNIX_EPOCH
        .elapsed()
        .unwrap()
        .as_secs();

    let create_output_file = |name| std::fs::File::create(format!("results/{timestamp}_{name}")).unwrap();
    let mut parameter_file = create_output_file("parameters.txt");
    let mut data_file      = create_output_file("trajectories.dat");
    let mut hist_file      = create_output_file("hist.dat");
    let mut current_file   = create_output_file("currents.dat");

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
    data_file.write_all(&(SIMULATION_COUNT * Real::LANES as u32).to_le_bytes()).unwrap();
    data_file.write_all(&(Operator::SIZE as u32).to_le_bytes()).unwrap();
    data_file.write_all(&(STEP_COUNT + 1).to_le_bytes()).unwrap();

    hist_file.write_all(&(Operator::SIZE as u32).to_le_bytes()).unwrap();
    hist_file.write_all(&(STEP_COUNT + 1).to_le_bytes()).unwrap();
    hist_file.write_all(&(HIST_BIN_COUNT as u32 * Operator::SIZE as u32).to_le_bytes()).unwrap();

    struct ThreadResult {
        trajectory_sum:  [StateProbabilitiesSimd; STEP_COUNT as usize+1],
        trajectory_hist: [[Histogram<HIST_BIN_COUNT>; Operator::SIZE]; STEP_COUNT as usize+1],
        current_sum: [Complex; SIMULATIONS_PER_THREAD as usize]
    }

    let threads: Vec<_> = (0..THREAD_COUNT).map(|thread_id| std::thread::spawn(move || {
        let mut local = alloc_zero!(ThreadResult);

        // Start the timer.
        let now = std::time::Instant::now();
        let mut total_section_time = 0;

        // Create initial system.
        let (initial_system, circuit) = QubitSystem::new();

        // sqrt(η) is from the definition of dZ.
        // 1/sqrt(dt) is to make the variance σ = dt
        let sqrtηdt = Real::splat(η.sqrt() * Δt.sqrt());
        println!("sqrtηdt: {sqrtηdt:?}");

        for simulation in 0..SIMULATIONS_PER_THREAD {
            //// let mut trajectory = [StateProbabilitiesSimd::zero(); STEP_COUNT as usize+1 ];

            // Initialize system
            //let before = now.elapsed().as_millis();
            let mut system = initial_system.clone();
            let mut circuit_state = circuit.make_new_state();

            //let after = now.elapsed().as_millis();
            //total_section_time += after - before;

            let mut t = Δt;

            let c = system.c_out_phased;
            let x = c + c.dagger();
            let y = MINUS_I*(c - c.dagger());

            let mut J = ZERO;

            let (mut H, mut steps_to_next_gate) = circuit_state.next();

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

                // Sample on the normal distribution.
                {
                    for lane in 0..Real::LANES {
                        system.dZ.real[lane] =
                            system.rng.sample::<fp, StandardNormal>(StandardNormal);
                        system.dZ.imag[lane] =
                            system.rng.sample::<fp, StandardNormal>(StandardNormal);
                    }

                    system.dZ *= &sqrtηdt;
                }

                // Do the runge-kutta4 step.
                //system.runge_kutta(H);
                //system.euler(H);
                system.millstein(H);

                // Check for NANs
                // if system.rho[(0,0)].first().0.is_nan() { panic!("step {}", step) }

                // Normalize rho.
                ////println!("p1{}", system.ρ);
                system.ρ.normalize();
                ////println!("p2{}", system.ρ);
                //assert!(system.ρ.trace().real[0] < 2.01);
                //assert!(system.ρ.trace().real[0] > 0.1);

                //println!("trace(rho) = {}", system.ρ.trace());

                // Compute current.
                // Original dW implementation reference
                // const SQRT2_OVER_DT: Real = Real::from_array([SQRT_2/Δt; Real::LANES]);
                // J += Complex {
                //     real: (x*system.ρ).trace().real + SQRT2_OVER_DT * system.dW[0],
                //     imag: (y*system.ρ).trace().real + SQRT2_OVER_DT * system.dW[1]
                // };
                // FIXME: Fix this, missing factor i think.
                const SQRT2_OVER_DT: Real = Real::from_array([SQRT_2/Δt; Real::LANES]);
                J += Complex {
                    real: (x*system.ρ).trace().real + SQRT2_OVER_DT * system.dZ.real,
                    imag: (y*system.ρ).trace().real + SQRT2_OVER_DT * system.dZ.imag
                };


                // Calculate integrated current
                //let zeta = system.Y * (1.0 / t.sqrt());

                t += Δt;
            }

            // Write last state.
            let P = system.ρ.get_probabilites_simd();
            local.trajectory_sum[STEP_COUNT as usize].add(&P);
            for (i, p) in P.v.iter().enumerate() {
                local.trajectory_hist[STEP_COUNT as usize][i].add_values(p);
            }

            local.current_sum[simulation as usize] = J;

        }
        let total_time = now.elapsed().as_millis();

        println!("Thread {thread_id} finished {SIMULATIONS_PER_THREAD} simulations in {total_time} ms ({} μs/sim) (total section time {} ms)",
                 (total_time*1000)/(SIMULATIONS_PER_THREAD*Real::LANES as u32) as u128,
                 total_section_time/1000
        );

        for s in local.trajectory_sum.iter_mut() {
            s.divide(SIMULATIONS_PER_THREAD as fp);
        }

        local
    })).collect();

    let mut trajectory_average = vec![StateProbabilitiesSimd::zero(); (STEP_COUNT+1) as usize].into_boxed_slice();
    let mut trajectory_histograms = alloc_zero!([[Histogram::<HIST_BIN_COUNT>; Operator::SIZE]; STEP_COUNT as usize+1]);

    // Wait for threads
    for tt in threads {
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

}

fn bloch_vector(rho: &Operator) -> [fp; 3] {
    const OFFSET: usize = 0;
    [
        rho[(OFFSET, 1 + OFFSET)].real()[0] + rho[(1 + OFFSET, 0 + OFFSET)].real()[0],
        rho[(OFFSET, 1 + OFFSET)].imag()[0] - rho[(1 + OFFSET, 0 + OFFSET)].imag()[0],
        rho[(OFFSET, 0 + OFFSET)].real()[0] - rho[(1 + OFFSET, 1 + OFFSET)].real()[0],
    ]
}

fn main() {
    println!("Starting simulation...");

    let start = std::time::Instant::now();
    simulate();
    let elapsed = start.elapsed().as_millis();

    println!(
        "Simulations took {elapsed} ms ({} sim/s, {} steps/s)",
        (1000*SIMULATION_COUNT*Real::LANES as u32) as fp / elapsed as fp,
        (1000*SIMULATION_COUNT as u64 * STEP_COUNT as u64 * Real::LANES as u64) as u128 / elapsed
    );

}
