#![feature(portable_simd)]
#![allow(non_snake_case)]
#![allow(non_upper_case_globals)]

mod num;
use num::*;

use rand_distr::StandardNormal;
use rayon::prelude::IntoParallelIterator;
use std::io::Write;

// use nalgebra::*;
use nalgebra::Vector3;
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

const sigma_z: Operator = Operator::new(ONE, ZERO, ZERO, MINUS_ONE);
const sigma_plus: Operator = Operator::new(ZERO, ONE, ZERO, ZERO);
const sigma_minus: Operator = Operator::new(ZERO, ZERO, ONE, ZERO);

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
    cscale(HALF, operator)
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

        let a = Operator::from_partial_diagonal(&[
            beta / (0.5 * kappa - I * (delta_r - g * g / delta_e)),
            beta / (0.5 * kappa - I * (delta_r + g * g / delta_e)),
        ]);

        let N = a.adjoint() * a;

        let hamiltonian = sigma_z.scale(delta_s * 0.5)
            + (a * sigma_plus + a.adjoint() * sigma_minus).scale(g)
            + N.scale(delta_r)
            + cscale( // Detuning
                I * (2.0 * kappa_1).sqrt(),
                beta * a.adjoint() - beta.conjugate() * a,
            )
            + (N * sigma_z).scale(chi);


        let gamma_p = 2.0 * g * g * kappa / (kappa * kappa + ddelta * ddelta);
        // let c_1 = cscale(gamma_p.sqrt(), sigma_minus);
        // let c_2 = cscale(gamma_p.sqrt(), sigma_plus);
        // let hamiltonian = hamilt17onian;
        //    - cscale(0.5*I, c_1.ad_mul(&c_1));
        //    - cscale(0.5*I, c_2.ad_mul(&c_2));

        //let psi = Vector2::<cf32>::new(ONE, ZERO);
        //let rho = psi.mul(&psi.transpose());

        let rho = Operator::new(ONE,ZERO,ZERO,ZERO);

        //let rho = Vector4::<cf32>::new(
        //    *rho.index((0, 0)),
        //    *rho.index((0, 1)),
        //    *rho.index((1, 0)),
        //    *rho.index((1, 1)),
        //);

        //let hamiltonian = cscale(MINUS_I, hamiltonian);
        let c_out = a.scale((kappa_1 * 2.0).sqrt()) - beta*Operator::identity();

        //let Ls = tensor_dot(A, A.adjoint())
        //    - tensor_dot(Operator::identity().scale(0.5), A * (A.adjoint()))
        //    - tensor_dot((A * A.adjoint()).scale(0.5), Operator::identity());

        let c1 = cscale(ONE * (2.0 * kappa).sqrt(), a);
        let c2 = cscale(ONE * gamma_dec.sqrt(), sigma_minus);
        let c3 = cscale(ONE * (gamma_phi / 2.0).sqrt(), sigma_z);

        let sqrt_eta = eta.sqrt();
        let c_out_phased = c_out * ((I * Phi).exp());
        let d_chi_rho = (c_out_phased + c_out_phased.adjoint()).scale(sqrt_eta);

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
        operator * self.rho * operator.adjoint()
            - half(anticommutator(operator.adjoint() * operator, self.rho))
    }

    fn dv(&mut self, rho: Operator) -> (Operator, cf32) {
        let h_cal = self.c_out_phased * self.rho + self.rho * self.c_out_phased.adjoint()
            - cscale(
                ((self.c_out_phased + self.c_out_phased.adjoint()) * self.rho).trace(),
                self.rho,
            );

        let h_cal_neg = cscale(MINUS_I, self.c_out_phased * self.rho)
            + cscale(I, self.rho * self.c_out_phased.adjoint())
            - cscale(
                ((self.c_out_phased + self.c_out_phased.adjoint()) * self.rho).trace(),
                self.rho,
            );

        let a = self.measurement;
        //println!("{}", rho);
        (
            cscale(MINUS_I, commutator(self.hamiltonian, rho))
                + self.lindblad(a)
                + self.lindblad(self.c1)
                + self.lindblad(self.c2)
                + self.lindblad(self.c3)
                + (h_cal*self.dW[0] + h_cal_neg*self.dW[1])
                    * self.rho.scale(self.sqrt_eta),
            ZERO,
        )

        // + chi_rho * self.d_chi_rho.scale((self.dW * self.dW * dt - 1.0) * 0.5) // Milstein
    }

    fn runge_kutta(&mut self, time_step: f32) {
        let (k_0, dY_0) = self.dv(self.rho);
        let (k_1, dY_1) = self.dv(self.rho + k_0.scale(0.5 * time_step));
        let (k_2, dY_2) = self.dv(self.rho + k_1.scale(0.5 * time_step));
        let (k_3, dY_3) = self.dv(self.rho + k_2.scale(time_step));
        self.rho += (k_1 + k_2 + (k_0 + k_3).scale(0.5)).scale(time_step / 3.0);
        self.Y += (dY_1 + dY_2 + (dY_0 + dY_3) * 0.5) * (time_step / 3.0);
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
            //for dw in system.dW.iter_mut() {
            //    *dw = system.rng.sample::<f32, StandardNormal>(StandardNormal) / dt.sqrt();
            //}

            //system
            //    .dW
            //    .fill_with(|| );
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
            system.rho = cscale(1.0 / system.rho.trace(), system.rho);

            if pipe.is_opened() {
                let bv = bloch_vector(&system.rho);
                pipe.write_vec3(bv);
                dbg!(&bv);
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



fn bloch_vector(rho: &Operator) -> Vector3<f32> {
    Vector3::new(
        rho[(0, 1)].real()[1] + rho[(1, 0)].real()[1],
        rho[(0, 1)].imag()[1] - rho[(1, 0)].imag()[1],
        rho[(0, 0)].real()[1] - rho[(1, 1)].real()[1],
    )
}

fn main() {
    println!("Starting simulation...");
    simulate();

    /*
    return;
    {
        let op = Operator::identity();
        let op2 = Operator::identity();
        let mut op3 = Operator::identity();

        let timer = std::time::Instant::now();
        for lane in 0..num::Real::LANES {
            for i in 0..10000000 {
                //let factor = 100000.0/(i as f32);
                unsafe { write_volatile(&mut op3, op * op2 * op3.scale(0.4)) };
            }
        }
        let time_in_ms = timer.elapsed().as_millis();
        println!("nalgebra took: {} ms", time_in_ms);
        //// println!("nalgebra was\n{}", op3);
    }

    {
        let op = num::Operator::ident();
        let op2 = num::Operator::ident();
        let mut op3 = num::Operator::ident();

        let timer = std::time::Instant::now();

        for _ in 0..10000000 {
            //let factors = num::Real::splat(100000.0)/ivals;

            unsafe { write_volatile(&mut op3, op * op2 * op3 * 0.4) };
            //ivals = seq + num::Real::splat((i*8) as f32);
        }
        let time_in_ms = timer.elapsed().as_millis();

        println!("ours took: {} ms", time_in_ms);
        //// println!("Ours was\n{}", op3);
        //println!("ivals {:?}", ivals);
        //println!("op1*op2 {}", op*op2);
        //println!("1*1={}", Complex::from(1.0)*Complex::from(1.0));
}
*/
}
