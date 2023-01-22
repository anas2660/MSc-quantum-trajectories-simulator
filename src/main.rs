#![feature(portable_simd)]
#![allow(non_snake_case)]
#![allow(non_upper_case_globals)]

mod num;
use num::*;
mod operator;
use operator::*;
mod matrix;
use matrix::*;
mod histogram;
use histogram::*;
// mod hamiltonian;
// use hamiltonian::*;



use rand_distr::StandardNormal;
use std::{
    io::Write,
    simd::{self, SimdPartialOrd, ToBitMask},
};

use rand::{rngs::ThreadRng, thread_rng, Rng};

mod pipewriter;
use pipewriter::*;

// type Operator = SMatrix<Complex<f32>, 2, 2>;
//type Operator = SMatrix<Complex<f32>, 2, 2>;
#[allow(non_camel_case_types)]
type cf32 = Complex;

const ONE: cf32 = Complex::new(1.0, 0.0);
const I: cf32 = Complex::new(0.0, 1.0);
const MINUS_I: cf32 = Complex::new(0.0, -1.0);
const ZERO: cf32 = Complex::new(0.0, 0.0);

const dt: f32 = 0.01;
const STEP_COUNT: u32 = 600;
const THREAD_COUNT: u32 = 10;
const HIST_BIN_COUNT: usize = 128;
const SIMULATIONS_PER_THREAD: u32 = 100;
const SIMULATION_COUNT: u32 = THREAD_COUNT * SIMULATIONS_PER_THREAD;

// From qutip implementation
//macro_rules! lowering {
//    ($D:expr) => {
//        SMatrix::<f32, $D, $D>::from_fn(|r, c| if r + 1 == c { (c as f32).sqrt() } else { 0.0 })
//    };
//}

//#[allow(unsued)]
struct QubitSystem {
    hamiltonian: Operator,
    rho: Operator,
    Y: cf32,
    //t: f32
    sqrt_eta: f32,
    c_out_phased: Operator,
    d_chi_rho: Operator,
    rng: ThreadRng,
    measurement: Operator,
    c1: Operator,
    c2: Operator,
    c3: [Operator; 2],
    dW: [Real; 2],
}

#[inline]
fn commutator(a: Operator, b: Operator) -> Operator {
    &(a * b) - &(b * a)
}

#[inline]
fn anticommutator(a: Operator, b: Operator) -> Operator {
    &(a * b) + &(b * a)
}


macro_rules! alloc_zero {
    ($T: ty) => {
        unsafe {
            let layout = std::alloc::Layout::new::<$T>();
            let ptr = std::alloc::alloc_zeroed(layout) as *mut $T;
            Box::from_raw(ptr)
        }
    };
}


fn anti_tensor_commutator(lhs: &Matrix, rhs: &Matrix) -> Matrix {
    lhs.kronecker(rhs) + rhs.kronecker(lhs)
}

fn apply_individually(op: &Matrix) -> Operator {
    let mut sum = Operator::zero();
    let identity = Matrix::identity(2);
    for i in 0..Operator::QUBIT_COUNT {
        let mut tmp = if i == 0 { op.clone() } else { identity.clone() };

        for j in 1..Operator::QUBIT_COUNT {
            tmp = if i == j { tmp.kronecker(op) } else { tmp.kronecker(&identity) }
        }

        sum += tmp.to_operator();
    }
    sum
}

fn apply_and_scale_individually(factors: [f32; Operator::QUBIT_COUNT], op: &Matrix) -> Operator {
    let mut sum = Operator::zero();
    let identity = Matrix::identity(2);
    for i in 0..Operator::QUBIT_COUNT {
        let mut tmp = if i == 0 { op.clone() } else { identity.clone() };

        for j in 1..Operator::QUBIT_COUNT {
            tmp = if i == j { tmp.kronecker(op) } else { tmp.kronecker(&identity) }
        }

        sum += factors[i] * tmp.to_operator();
    }
    sum
}




impl QubitSystem {
    fn new(
        A: Operator,
        delta_s: f32,
        g: f32,
        kappa_1: f32,
        kappa: f32,
        beta: cf32,
        delta_r: f32,
        eta: f32,
        Phi: f32,
        gamma_dec: f32,
        gamma_phi: f32,
    ) -> Self {
        let ddelta = delta_r - delta_s;
        let omega = 1.0;
        // let g = chi * ddelta.abs();

        let chi     = 0.6 / 10.0;
        let g       = 125.6637061435917 / 100.0;
        let omega_r = 28368.582 / 100.0;
        let omega_s = 2049.6365 / 100.0;
        let delta_s = 26318.94506957162 / 100.0;


        let omega_b = 0.1;
        let delta_s = [delta_s; 2]; // |ω_r - ω_s|
        let delta_bs = [omega_s; 2]; // ω_b - ω_s
        let delta_br = omega_r; // ω_b - ω_r
        let chi = [chi; 2];
        //let chi = [0.6, 0.6];

        #[allow(unused_variables)]
        let sigma_plus: Matrix = Matrix::new(0.0, 1.0, 0.0, 0.0);
        let sigma_z: Matrix = Matrix::new(1.0, 0.0, 0.0, -1.0);
        let sigma_minus: Matrix = Matrix::new(0.0, 0.0, 1.0, 0.0);
        let identity = Operator::identity();

        let χσ_z = apply_and_scale_individually(chi, &sigma_z);
        let a = (2.0*kappa_1).sqrt() * beta / (delta_r*identity + I*χσ_z + kappa*identity);

        let N = a.dagger() * a;

        let zero = Matrix::vector(&[1., 0.]);
        let one = Matrix::vector(&[0., 1.]);

        let bra = |m: &Matrix| m.dagger();
        let ket = |m: &Matrix| m.clone();

        /////let hamiltonian =
        /////    //+ g * (a * sigma_plus + a.dagger() * sigma_minus)
        /////    //////+ chi*(&sigma_z + &identity)
        /////    //+ omega * (ket(&one)*bra(&zero) + ket(&zero)*bra(&one)) // ω(|1X0| + |0X1|)
        /////    ;

        /// REFERENCE
        /// let sigma_z_4x4 = apply_individually(&sigma_z);
        /// let hamiltonian =
        ///     //(hamiltonian.kronecker(&identity) + identity.kronecker(&hamiltonian)).to_operator()
        ///     0.5 * delta_s * sigma_z_4x4
        ///     + I * (2.0 * kappa_1).sqrt() * (beta * a.dagger() - beta.conjugate() * a) // Detuning
        ///     + delta_r * N
        ///     + chi * N * sigma_z_4x4;
        ///     //0.5 * omega * apply_individually(&(&sigma_plus + &sigma_minus));


        let hamiltonian =
            //(hamiltonian.kronecker(&identity) + identity.kronecker(&hamiltonian)).to_operator()
            0.5 * apply_and_scale_individually(delta_bs, &sigma_z)
            + I * (2.0 * kappa_1).sqrt() * (beta * a.dagger() - beta.conjugate() * a) // Detuning
            + delta_br * N
            + (N + 0.5*identity) * χσ_z;
            //+ 0.5 * omega * apply_individually(&(&sigma_plus + &sigma_minus));



        //let hamiltonian = omega_r * a.dagger() * a + (omega_s + 2.0 * (g*g/delta_s) * (a.dagger()*a + 0.5*Operator::identity()));


        // CNOT 0, 1
        //let hamiltonian = hamiltonian
        //    + omega * (ket(&one)*bra(&one)).kronecker(&(ket(&one)*bra(&zero) + ket(&zero)*bra(&one))).to_operator();

        // let gamma_p = 2.0 * g * g * kappa / (kappa * kappa + ddelta * ddelta);

        // Initial state probabilities
        let mut p = [
            0.4, // 00
            0.1, // 01
            0.4, // 10
            0.1  // 11
        ];

        // Transform into coefficients
        for x in p.iter_mut() { *x = f32::sqrt(*x) }

        // Construct initial state
        let psi = Matrix::vector(&p);
        let rho = (&psi * &psi.transpose()).to_operator();

        // println!("{rho}");

        //let c_out =
        //    (kappa_1 * 2.0).sqrt() * a.kronecker(&a).to_operator() - beta * Operator::identity();
        //let c1 = (2.0 * kappa).sqrt() * a.kronecker(&a).to_operator();
        //let c2 = gamma_dec.sqrt() * sigma_minus.kronecker(&sigma_minus).to_operator();
        //let c3 = (gamma_phi / 2.0).sqrt() * sigma_z.kronecker(&sigma_z).to_operator();

        let c_out = (kappa_1 * 2.0).sqrt() * a - beta * Operator::identity();
        let c1 = (2.0 * kappa).sqrt() * a;
        let c2 = gamma_dec.sqrt() * &sigma_minus;//sigma_minus;
        let c3 = (gamma_phi / 2.0).sqrt() * sigma_z;

        let identity = Matrix::identity(2);

        //let c_out = (c_out.kronecker(&identity) + identity.kronecker(&c_out)).to_operator();
        //let c1 = (c1.kronecker(&identity) + identity.kronecker(&c1)).to_operator();
        let c2 = (c2.kronecker(&identity) + identity.kronecker(&c2)).to_operator();
        let c3 = [c3.kronecker(&identity).to_operator(), identity.kronecker(&c3).to_operator()];

        let sqrt_eta = eta.sqrt();
        let c_out_phased = c_out * ((I * Phi).exp());
        let d_chi_rho = sqrt_eta * (c_out_phased + c_out_phased.adjoint());

        Self {
            hamiltonian,
            rho,
            Y: ZERO,
            sqrt_eta,
            c_out_phased,
            d_chi_rho,
            rng: thread_rng(),
            measurement: A,
            c1,
            c2,
            c3,
            dW: [Real::splat(0.0); 2],
        }
    }

    fn lindblad(&self, operator: &Operator) -> Operator {
        let op_dag = operator.dagger();

        operator * self.rho * op_dag
            - 0.5 * anticommutator(op_dag * operator, self.rho)
    }

    fn dv(&mut self, rho: Operator) -> (Operator, ()/*cf32*/) {
        let cop = self.c_out_phased;
        let copa = self.c_out_phased.adjoint();
        let cop_rho = cop * self.rho;
        let rho_copa = self.rho * copa;
        let last_term = (cop_rho + copa * self.rho).trace() * self.rho;

        let h_cal = cop_rho + rho_copa - last_term;
        let h_cal_neg = MINUS_I * cop_rho + I * rho_copa - last_term;

        // let a = self.measurement;
        (
            MINUS_I * commutator(self.hamiltonian, rho)
                //+ self.lindblad(a)
                + self.lindblad(&self.c1) // Photon field transmission/losses
                //+ self.lindblad(&self.c2) // Decay to ground state
                + self.lindblad(&self.c3[0]) + self.lindblad(&self.c3[1])
                //+ self.lindblad(&cop) // c_out
                + (h_cal * self.dW[0] + h_cal_neg * self.dW[1]) * self.sqrt_eta * self.rho,
            ()//ZERO,
        )
        // + chi_rho * self.d_chi_rho.scale((self.dW * self.dW * dt - 1.0) * 0.5) // Milstein
    }

    fn runge_kutta(&mut self, time_step: f32) {
        let (k_0, dY_0) = self.dv(self.rho);
        let (k_1, dY_1) = self.dv(self.rho + 0.5 * time_step * k_0);
        let (k_2, dY_2) = self.dv(self.rho + 0.5 * time_step * k_1);
        let (k_3, dY_3) = self.dv(self.rho + time_step * k_2);
        let t_3 = time_step / 3.0;
        self.rho += t_3 * (k_1 + k_2 + 0.5 * (k_0 + k_3));
        //self.Y += t_3 * (dY_1 + dY_2 + 0.5 * (dY_0 + dY_3));
    }
}

//fn tensor_dot(m1: Operator, m2: Operator) -> SMatrix<cf32, 4, 4> {
//    m1.kronecker(&m2.transpose())
//}

fn simulate() {
    let timestamp = std::time::SystemTime::UNIX_EPOCH
        .elapsed()
        .unwrap()
        .as_secs();

    let mut parameter_file =
        std::fs::File::create(format!("results/{timestamp}_parameters.txt")).unwrap();
    let mut data_file =
        std::fs::File::create(format!("results/{timestamp}_trajectories.dat")).unwrap();
    let mut hist_file =
        std::fs::File::create(format!("results/{timestamp}_hist.dat")).unwrap();
    let mut current_file =
        std::fs::File::create(format!("results/{timestamp}_currents.dat")).unwrap();

    //let gamma = SMatrix::<cf32, 4, 4>::from_diagonal_element(ONE);

    let A = Operator::from_fn(|r, c| ONE * (r * c) as f32);
    let delta_s = 2.0;
    let g = 2.0;
    let kappa = 1.2;
    let kappa_1 = 1.2; // NOTE: Max value is the value of kappa. This is for the purpose of loss between emission and measurement.
    let beta = 0.5*ONE;
    let delta_r = 0.5;
    let eta = 0.1;
    let Phi = 0.0;
    let gamma_dec = 1.0;
    let gamma_phi = 1.0;

    parameter_file
        .write_all(
            format!(
                "let A = {A};
let beta = {beta};
let delta_s = {delta_s};
let delta_r = {delta_r};
let g = {g};
let kappa = {kappa};
let kappa_1 = {kappa_1};
let eta = {eta};
let Phi = {Phi};
let gamma_dec = {gamma_dec};
let gamma_phi = {gamma_phi};
"
            )
                .as_bytes(),
        )
        .unwrap();

    let mut pipe = PipeWriter::open("/tmp/blochrender_pipe");
    pipe.write_u32(STEP_COUNT);

    ////////let mut signal = Vec::new();

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

        for simulation in 0..SIMULATIONS_PER_THREAD {
            //// let mut trajectory = [StateProbabilitiesSimd::zero(); STEP_COUNT as usize+1 ];

            // Initialize system
            let before = now.elapsed().as_millis();
            let mut system = QubitSystem::new(
                A, delta_s, g, kappa_1, kappa, beta, delta_r, eta, Phi, gamma_dec,
                gamma_phi,
            );
            let after = now.elapsed().as_millis();
            total_section_time += after - before;


            let mut t = dt;

            //current_file.write(b"0.0, 0.0").unwrap();
            //current_file.write(&[0u8; 8]).unwrap();

            let c = system.c_out_phased;
            let x = c + c.dagger();
            let y = MINUS_I*(c - c.dagger());

            let mut J = ZERO;

            // Do 2000 steps.
            for step in 0..STEP_COUNT as usize {
                // Write current state.
                //data_file
                //    .write(format!("{}, ", system.rho[(0, 0)].real()).as_bytes())
                //    .unwrap();
                //// trajectory[step] = system.rho.get_probabilites_simd();

                //let section_start = std::time::Instant::now();
                let P = system.rho.get_probabilites_simd();
                local.trajectory_sum[step].add(&P);
                for (i, p) in P.v.iter().enumerate() {
                    local.trajectory_hist[step][i].add_values(p);
                }
                //total_section_time += section_start.elapsed().as_micros();


                // TODO: DELETE
                //assert_eq!((system.rho[(0, 0)].imag()*system.rho[(0, 0)].imag()).simd_lt(Real::splat(0.02)).to_bitmask(), 255);

                // Sample on the normal distribution.
                {
                    for lane in 0..Real::LANES {
                        system.dW[0][lane] =
                            system.rng.sample::<f32, StandardNormal>(StandardNormal);
                        system.dW[1][lane] =
                            system.rng.sample::<f32, StandardNormal>(StandardNormal);
                    }
                    let c = Real::splat(1.0 / dt.sqrt());
                    system.dW[0] *= c;
                    system.dW[1] *= c;
                }

                // Do the runge-kutta4 step.
                system.runge_kutta(dt);

                // if system.rho[(0,0)].first().0.is_nan() { panic!("step {}", step) }


                // Normalize rho.
                //println!("Trace: {}", system.rho.trace());
                system.rho = system.rho / system.rho.trace();

                // Compute current.
                const SQRT2_OVER_DT: Real = Real::from_array([std::f32::consts::SQRT_2/dt; Real::LANES]);
                J += Complex {
                    real: (x*system.rho).trace().real + SQRT2_OVER_DT * system.dW[0],
                    imag: (y*system.rho).trace().real + SQRT2_OVER_DT * system.dW[1]
                };

                // println!("[{}, {}]", system.rho[(0,0)], system.rho[(1,1)]);

                ////////if pipe.is_opened() {
                ////////    let bv = bloch_vector(&system.rho);
                ////////    //let f = |c: Complex| {
                ////////    //    let (r, i) = c.first();
                ////////    //    (r*r + i*i).sqrt()
                ////////    //};
                ////////    //pipe.write_vec3([f(system.rho[(0,0)]), f(system.rho[(1,1)]), f(system.rho[(1,1)])]);
                ////////    pipe.write_vec3(bv);
                ////////    //dbg!(&bv);
                ////////}

                // Calculate integrated current
                let zeta = system.Y * (1.0 / t.sqrt());

                //current_file
                //    .write(format!(", {}, {}", zeta.real(), zeta.imag()).as_bytes())
                //    .unwrap();
                ////////signal.push(zeta);

                t += dt;
            }

            // Write last state.
            //// trajectory[STEP_COUNT as usize] = system.rho.get_probabilites_simd();
            let P = system.rho.get_probabilites_simd();
            local.trajectory_sum[STEP_COUNT as usize].add(&P);
            for (i, p) in P.v.iter().enumerate() {
                local.trajectory_hist[STEP_COUNT as usize][i].add_values(p);
            }

            local.current_sum[simulation as usize] = J;

            //current_file
            //    .write(unsafe {
            //        std::slice::from_raw_parts(
            //            signal.as_ptr() as *const u8,
            //            signal.len() * std::mem::size_of::<cf32>(),
            //        )
            //    })
            //    .unwrap();



            ////////signal.clear();

            ////////if pipe.is_opened() {
            ////////    break;
            ////////}
        }
        let total_time = now.elapsed().as_millis();

        println!("Thread {thread_id} finished {SIMULATIONS_PER_THREAD} simulations in {total_time} ms ({} μs/sim) (total section time {} ms)",
                 (total_time*1000)/(SIMULATIONS_PER_THREAD*Real::LANES as u32) as u128,
                 total_section_time/1000
        );

        for s in local.trajectory_sum.iter_mut() {
            s.divide(SIMULATIONS_PER_THREAD as f32);
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
        s.divide(THREAD_COUNT as f32);
        data_file.write_all(&s.average().to_le_bytes()).unwrap();
    }

    // TODO: actual time
    // TODO: fix magic number (simcount)
    for i in 0..=STEP_COUNT {
        let t = i as f32 * dt;
        data_file.write_all(&t.to_le_bytes()).unwrap();
    }

    for state in trajectory_histograms.iter() {
        for hist in state.iter() {
            for bin in hist.bins.iter().rev() {
                hist_file.write_all(&bin.to_le_bytes()).unwrap();
            }
        }
    }
}

fn bloch_vector(rho: &Operator) -> [f32; 3] {
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
        (1000*SIMULATION_COUNT*Real::LANES as u32) as f32 / elapsed as f32,
        (1000*SIMULATION_COUNT as u64 * STEP_COUNT as u64 * Real::LANES as u64) as u128 / elapsed
    );

}
