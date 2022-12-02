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
use std::io::Write;

use rand::{rngs::ThreadRng, thread_rng, Rng};

mod pipewriter;
use pipewriter::*;

// type Operator = SMatrix<Complex<f32>, 2, 2>;
//type Operator = SMatrix<Complex<f32>, 2, 2>;
#[allow(non_camel_case_types)]
type cf32 = Complex;

const HALF: cf32 = Complex::new(0.5, 0.0);
const ONE: cf32 = Complex::new(1.0, 0.0);
const MINUS_ONE: cf32 = Complex::new(-1.0, 0.0);
const I: cf32 = Complex::new(0.0, 1.0);
const MINUS_I: cf32 = Complex::new(0.0, -1.0);
const ZERO: cf32 = Complex::new(0.0, 0.0);

const dt: f32 = 0.01;
const STEP_COUNT: u32 = 1000;
const SIMULATION_COUNT: u32 = 1000;

// From qutip implementation
macro_rules! lowering {
    ($D:expr) => {
        SMatrix::<f32, $D, $D>::from_fn(|r, c| if r + 1 == c { (c as f32).sqrt() } else { 0.0 })
    };
}

//#[inline]
//fn cscale(c: cf32, m: Operator) -> Operator {
//    m.component_mul(&Operator::from_element(c))
//}

#[inline]
fn cscale(c: cf32, m: Operator) -> Operator {
    c * &m
    //m.component_mul(&Operator::from_element(c))
}

// fn cscale4(c: cf32, m: SMatrix<cf32, 4, 4>) -> SMatrix<cf32, 4, 4> {
//     m.component_mul(&SMatrix::<cf32, 4, 4>::from_element(c))
// }

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

#[inline]
fn half(operator: Operator) -> Operator {
    0.5 * operator
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
        let alpha = (2.0 * kappa_1).sqrt() / (kappa + I * delta_r) * beta;
        let ddelta = delta_r - delta_s;
        let epsilon_s = g * g * ddelta / (kappa * kappa + ddelta * ddelta);
        //let a = cscale((2.0 * kappa_1).sqrt() / (kappa + I * delta_r), beta)
        //    - cscale(I * g / (kappa + I * ddelta), sigma_minus);
        let delta_e = 1.0;
        let chi = 0.6;

        let sigma_z: Matrix = Matrix::new(1.0, 0.0, 0.0, -1.0);
        let sigma_plus: Matrix = Matrix::new(0.0, 1.0, 0.0, 0.0);
        let sigma_minus: Matrix = Matrix::new(0.0, 0.0, 1.0, 0.0);

        let a = Matrix::from_partial_diagonal(&[
            beta / (0.5 * kappa - I * (delta_r - Complex::from(chi))),
            beta / (0.5 * kappa - I * (delta_r + Complex::from(chi))),
        ]);

        let N = a.dagger() * &a;

        let hamiltonian = 0.5 * delta_s * &sigma_z
            //+ g * (a * sigma_plus + a.dagger() * sigma_minus)
            + delta_r * &N
            + I * (2.0 * kappa_1).sqrt() * beta * a.dagger() - beta.conjugate() * &a // Detuning
            + chi * &N * &sigma_z
            + chi*(&sigma_z + &Matrix::identity(2));

        let hamiltonian = hamiltonian.kronecker(&hamiltonian).to_operator();

        let gamma_p = 2.0 * g * g * kappa / (kappa * kappa + ddelta * ddelta);
        // let c_1 = cscale(gamma_p.sqrt(), sigma_minus);
        // let c_2 = cscale(gamma_p.sqrt(), sigma_plus);
        // let hamiltonian = hamilt17onian;
        //    - cscale(0.5*I, c_1.ad_mul(&c_1));
        //    - cscale(0.5*I, c_2.ad_mul(&c_2));

        //let psi = Vector2::<cf32>::new(ONE, ZERO);
        //let rho = psi.mul(&psi.transpose());

        let small_rho = Matrix::new(0.5, 0.5, 0.5, 0.5);
        let rho = small_rho.kronecker(&small_rho).to_operator();

        //let rho = Vector4::<cf32>::new(
        //    *rho.index((0, 0)),
        //    *rho.index((0, 1)),
        //    *rho.index((1, 0)),
        //    *rho.index((1, 1)),
        //);

        //let hamiltonian = cscale(MINUS_I, hamiltonian);
        let c_out =
            (kappa_1 * 2.0).sqrt() * a.kronecker(&a).to_operator() - beta * Operator::identity();

        //let Ls = tensor_dot(A, A.adjoint())
        //    - tensor_dot(Operator::identity().scale(0.5), A * (A.adjoint()))
        //    - tensor_dot((A * A.adjoint()).scale(0.5), Operator::identity());

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

        let a = self.measurement;
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

    let mut trajectory = Vec::new();
    let mut signal = Vec::new();

    // metadata
    //current_file.write(&SIMULATION_COUNT.to_le_bytes()).unwrap();
    //current_file.write(&STEP_COUNT.to_le_bytes()).unwrap();
    data_file.write(&SIMULATION_COUNT.to_le_bytes()).unwrap();
    data_file.write(&(STEP_COUNT + 1).to_le_bytes()).unwrap();

    for simulation in 0..1000 {
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
            trajectory.push(system.rho[(0, 0)].real()[0]);

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
            system.rho = system.rho / system.rho.trace();

            //println!("[{}, {}]", system.rho[(0,0)], system.rho[(1,1)]);

            if pipe.is_opened() {
                let bv = bloch_vector(&system.rho);
                pipe.write_vec3(bv);
                //dbg!(&bv);
            }

            // Calculate integrated current
            let zeta = system.Y * (1.0 / t.sqrt());

            //current_file
            //    .write(format!(", {}, {}", zeta.real(), zeta.imag()).as_bytes())
            //    .unwrap();
            signal.push(zeta);

            t += dt;
        }

        // Write last state.
        trajectory.push(system.rho[(0, 0)].real()[0]);

        //data_file
        //    .write(format!("{}\n", system.rho[(0, 0)].real()).as_bytes())
        //    .unwrap();
        //current_file.write(b"\n").unwrap();

        println!("Sim ({simulation}) time: {} us", now.elapsed().as_micros());

        data_file
            .write(unsafe {
                std::slice::from_raw_parts(
                    trajectory.as_ptr() as *const u8,
                    trajectory.len() * std::mem::size_of::<f32>(),
                )
            })
            .unwrap();

        //current_file
        //    .write(unsafe {
        //        std::slice::from_raw_parts(
        //            signal.as_ptr() as *const u8,
        //            signal.len() * std::mem::size_of::<cf32>(),
        //        )
        //    })
        //    .unwrap();

        trajectory.clear();
        signal.clear();

        if pipe.is_opened() {
            break;
        }
    }
}

fn bloch_vector(rho: &Operator) -> [f32; 3] {
    [
        rho[(0, 1)].real()[0] + rho[(1, 0)].real()[0],
        rho[(0, 1)].imag()[0] - rho[(1, 0)].imag()[0],
        rho[(0, 0)].real()[0] - rho[(1, 1)].real()[0],
    ]
}

fn main() {
    //let other_sigma_z: Matrix = Matrix::new(1.0, 0.0, 0.0, -1.0);
    //println!("other_sigma_z: \n{other_sigma_z}");
    //let mut ident = Matrix::identity(3);
    //ident[(0,1)] = ONE;
    //println!("ident: \n{ident}");
    //let prod = ident.kronecker(&other_sigma_z);
    //println!("prod: \n{prod}");

    println!("Starting simulation...");
    simulate();
}
