#![feature(portable_simd)]
#![allow(non_snake_case)]
#![allow(non_upper_case_globals)]

mod num;
use num::*;
mod operator;
use operator::*;
mod matrix;
use matrix::*;

use rand_distr::StandardNormal;
use std::{io::Write, simd::{self, SimdPartialOrd, ToBitMask}};

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

const dt: f32 = 0.03;
const STEP_COUNT: u32 = 300;
const THREAD_COUNT: u32 = 10;
const SIMULATIONS_PER_THREAD: u32 = 100;
const SIMULATION_COUNT: u32 = THREAD_COUNT*SIMULATIONS_PER_THREAD;

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
    c3: Operator,
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
        let chi = 0.6;
        let omega = 3.0;

        #[allow(unused_variables)]
        let sigma_plus: Matrix = Matrix::new(0.0, 1.0, 0.0, 0.0);
        let sigma_z: Matrix = Matrix::new(1.0, 0.0, 0.0, -1.0);
        let sigma_minus: Matrix = Matrix::new(0.0, 0.0, 1.0, 0.0);

        let a = Matrix::from_partial_diagonal(&[
            beta / (0.5 * kappa - I * (delta_r - Complex::from(chi))),
            beta / (0.5 * kappa - I * (delta_r + Complex::from(chi))),
        ]);

        let N = a.dagger() * &a;

        let zero = Matrix::vector(&[1., 0.]);
        let one  = Matrix::vector(&[0., 1.]);

        let bra = |m: &Matrix| { m.dagger() };
        let ket = |m: &Matrix| { m.clone() };

        let identity = Matrix::identity(2);
        let hamiltonian =
            0.5 * delta_s * &sigma_z
            //+ g * (a * sigma_plus + a.dagger() * sigma_minus)
            + delta_r * &N
            + I * (2.0 * kappa_1).sqrt() * beta * a.dagger() - beta.conjugate() * &a // Detuning
            + chi * &N * &sigma_z
            + chi*(&sigma_z + &identity)
            //+ omega * (ket(&one)*bra(&zero) + ket(&zero)*bra(&one)) // ω(|1X0| + |0X1|)
            ;


        let hamiltonian = (hamiltonian.kronecker(&identity) + identity.kronecker(&hamiltonian)).to_operator();

        // CNOT 0, 1
        let hamiltonian = hamiltonian
            +
            omega*(&(ket(&one)*bra(&one))).kronecker(&(ket(&one)*bra(&zero) + ket(&zero)*bra(&one))).to_operator()
            //omega*(&(ket(&zero)*bra(&zero))).kronecker(&identity).to_operator()
            ;

        // let gamma_p = 2.0 * g * g * kappa / (kappa * kappa + ddelta * ddelta);

        // 00  0.5
        // 01  0
        // 10  0.5
        // 11  0
        let psi = Matrix::vector(&[f32::sqrt(0.5), 0.0, f32::sqrt(0.5), 0.0]);
        let rho = (&psi * &psi.transpose()).to_operator();

        // println!("{rho}");

        let c_out =
            (kappa_1 * 2.0).sqrt() * a.kronecker(&a).to_operator() - beta * Operator::identity();
        let c1 = (2.0 * kappa).sqrt() * a.kronecker(&a).to_operator();
        let c2 = gamma_dec.sqrt() * sigma_minus.kronecker(&sigma_minus).to_operator();
        let c3 = (gamma_phi / 2.0).sqrt() * sigma_z.kronecker(&sigma_z).to_operator();

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

    fn lindblad(&self, operator: Operator) -> Operator {
        operator * self.rho * operator.dagger()
            - 0.5 * anticommutator(operator.dagger() * operator, self.rho)
    }

    fn dv(&mut self, rho: Operator) -> (Operator, cf32) {
        let h_cal = self.c_out_phased * self.rho + self.rho * self.c_out_phased.adjoint()
            - ((self.c_out_phased + self.c_out_phased.adjoint()) * self.rho).trace() * self.rho;

        let h_cal_neg = MINUS_I * self.c_out_phased * self.rho
            + I * self.rho * self.c_out_phased.adjoint()
            - ((self.c_out_phased + self.c_out_phased.adjoint()) * self.rho).trace() * self.rho;

        // let a = self.measurement;
        (
            MINUS_I * commutator(self.hamiltonian, rho)
                //+ self.lindblad(a)
                + self.lindblad(self.c1) // Photon field transmission/losses
                //+ self.lindblad(self.c2) // Decay to ground state
                + self.lindblad(self.c3)
                + (h_cal * self.dW[0] + h_cal_neg * self.dW[1]) * self.sqrt_eta * self.rho,
            ZERO,
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
        self.Y += t_3 * (dY_1 + dY_2 + 0.5 * (dY_0 + dY_3));
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
        std::fs::File::create(format!("results/{}_parameters.txt", timestamp)).unwrap();
    let mut data_file =
        std::fs::File::create(format!("results/{}_trajectories.dat", timestamp)).unwrap();
    //let mut current_file =
    //    std::fs::File::create(format!("results/{}_currents.dat", timestamp)).unwrap();

    //let gamma = SMatrix::<cf32, 4, 4>::from_diagonal_element(ONE);


    let A = Operator::from_fn(|r, c| ONE * (r * c) as f32);
    let delta_s = 1.0;
    let g = 2.0;
    let kappa_1 = 10.0;
    let kappa = 10.0;
    let beta = ONE;
    let delta_r = 0.0;
    let eta = 0.5;
    let Phi = 0.0;
    let gamma_dec = 1.0;
    let gamma_phi = 1.0;
    let aaa = 3.3/2.0;

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
    //current_file.write(&SIMULATION_COUNT.to_le_bytes()).unwrap();
    //current_file.write(&STEP_COUNT.to_le_bytes()).unwrap();
    data_file.write(&(SIMULATION_COUNT * Real::LANES as u32).to_le_bytes()).unwrap();
    data_file.write(&(Operator::SIZE as u32).to_le_bytes()).unwrap();
    data_file.write(&(STEP_COUNT + 1).to_le_bytes()).unwrap();

    let threads: Vec<_> = (0..THREAD_COUNT).map(|thread_id| std::thread::spawn(move || {
        let mut local_trajectories = Vec::new();

        for simulation in 0..SIMULATIONS_PER_THREAD {
            let mut trajectory = Vec::new();
            // Initialize system
            let mut system = QubitSystem::new(
                A, delta_s, g, kappa_1, kappa, beta, delta_r, eta, Phi, gamma_dec, gamma_phi,
            );

            // Start the timer.
            let now = std::time::Instant::now();
            let mut t = dt;

            //current_file.write(b"0.0, 0.0").unwrap();
            //current_file.write(&[0u8; 8]).unwrap();

            // Do 2000 steps.
            for _ in 0..STEP_COUNT {
                // Write current state.
                //data_file
                //    .write(format!("{}, ", system.rho[(0, 0)].real()).as_bytes())
                //    .unwrap();
                trajectory.push([system.rho[(0, 0)].real(),system.rho[(1, 1)].real(),system.rho[(2, 2)].real(),system.rho[(3, 3)].real()]);

                // TODO: DELETE
                //assert_eq!((system.rho[(0, 0)].imag()*system.rho[(0, 0)].imag()).simd_lt(Real::splat(0.02)).to_bitmask(), 255);


                // Sample on the normal distribution.
                {
                    for lane in 0..Real::LANES {
                        system.dW[0][lane] = system.rng.sample::<f32, StandardNormal>(StandardNormal);
                        system.dW[1][lane] = system.rng.sample::<f32, StandardNormal>(StandardNormal);
                    }
                    let c = Real::splat(1.0 / dt.sqrt());
                    system.dW[0] *= c;
                    system.dW[1] *= c;
                }

                // Do the runge-kutta4 step.
                system.runge_kutta(dt);

                // Normalize rho.
                //println!("Trace: {}", system.rho.trace());
                system.rho = system.rho / system.rho.trace();

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
            //trajectory.push(system.rho[(0, 0)].real()[0]);
            trajectory.push([system.rho[(0, 0)].real(),system.rho[(1, 1)].real(),system.rho[(2, 2)].real(),system.rho[(3, 3)].real()]);


            //data_file
            //    .write(format!("{}\n", system.rho[(0, 0)].real()).as_bytes())
            //    .unwrap();
            //current_file.write(b"\n").unwrap();

            println!("Sim ({}) time: {} us", thread_id*SIMULATIONS_PER_THREAD + simulation, now.elapsed().as_micros());

            //data_file
            //    .write(unsafe {
            //        std::slice::from_raw_parts(
            //            trajectory.as_ptr() as *const u8,
            //            trajectory.len() * std::mem::size_of::<f32>(),
            //        )
            //    })
            //    .unwrap();


            //current_file
            //    .write(unsafe {
            //        std::slice::from_raw_parts(
            //            signal.as_ptr() as *const u8,
            //            signal.len() * std::mem::size_of::<cf32>(),
            //        )
            //    })
            //    .unwrap();

            ////////trajectory.clear();
            local_trajectories.push(trajectory);
            ////////signal.clear();

            ////////if pipe.is_opened() {
            ////////    break;
            ////////}
        }
        local_trajectories
    })).collect();

    let mut trajectories = [[Real::splat(0.0); Operator::SIZE]; (STEP_COUNT+1) as usize];

    // Wait for threads
    for tt in threads {
        for trajectory in tt.join().unwrap().iter() {
            for (out, cur) in trajectories.iter_mut().zip(trajectory.iter()) {
                for i in 0..Operator::SIZE {
                    out[i] += cur[i];
                }
            }
        }
    }

    let mut trajectory = [[0f32; Operator::SIZE]; (STEP_COUNT+1) as usize];
    for (result, s) in trajectory.iter_mut().zip(trajectories.iter()) {
        *result = s.map(|sv| sv.as_array().iter().sum::<f32>()/(Real::LANES as f32 * SIMULATION_COUNT as f32) );
    }


    // write average trajectory
    for state in trajectory.iter() {
        let mut buf = [0u8; std::mem::size_of::<f32>() * Operator::SIZE];
        for (vb, v) in buf.chunks_mut(std::mem::size_of::<f32>()).zip(state.iter()) {
            vb.copy_from_slice(&v.to_le_bytes());
        }

        data_file.write(&buf).unwrap();
    }

    // TODO: actual time
    // TODO: fix magic number (simcount)
    for i in 0..=STEP_COUNT {
        let t = i as f32 * dt;
        data_file.write(&t.to_le_bytes()).unwrap();
    }

}

fn bloch_vector(rho: &Operator) -> [f32; 3] {
    const OFFSET: usize = 0;
    [
        rho[(0 + OFFSET, 1 + OFFSET)].real()[0] + rho[(1 + OFFSET, 0 + OFFSET)].real()[0],
        rho[(0 + OFFSET, 1 + OFFSET)].imag()[0] - rho[(1 + OFFSET, 0 + OFFSET)].imag()[0],
        rho[(0 + OFFSET, 0 + OFFSET)].real()[0] - rho[(1 + OFFSET, 1 + OFFSET)].real()[0],
    ]
}

fn main() {
    println!("Starting simulation...");
    simulate();
}
