#![allow(non_snake_case)]
#![allow(non_upper_case_globals)]

use rand_distr::StandardNormal;
use std::{io::Write, ops::Mul};

use nalgebra::*;
use rand::{rngs::ThreadRng, thread_rng, Rng};

//type Operator = SMatrix::<f32, 2, 2>;
type Operator = SMatrix<Complex<f32>, 2, 2>;
//type Dimensions = usize;

#[allow(non_camel_case_types)]
type cf32 = Complex<f32>;

const HALF: Complex<f32> = Complex::new(0.5, 0.0);
const ONE: Complex<f32> = Complex::new(1.0, 0.0);
const MINUS_ONE: Complex<f32> = Complex::new(-1.0, 0.0);
const I: Complex<f32> = Complex::new(0.0, 1.0);
const MINUS_I: Complex<f32> = Complex::new(0.0, -1.0);
const ZERO: Complex<f32> = Complex::new(0.0, 0.0);

const dt: f32 = 0.0005;
const STEP_COUNT: u32 = 20000;

const sigma_z: Operator = Operator::new(ONE, ZERO, ZERO, MINUS_ONE);
const sigma_plus: Operator = Operator::new(ZERO, ONE, ZERO, ZERO);
const sigma_minus: Operator = Operator::new(ZERO, ZERO, ONE, ZERO);

// From qutip implementation
macro_rules! lowering {
    ($D:expr) => {
        SMatrix::<f32, $D, $D>::from_fn(|r, c| if r + 1 == c { (c as f32).sqrt() } else { 0.0 })
    };
}

macro_rules! identity {
    ($D:expr) => {
        SMatrix::<f32, $D, $D>::identity()
    };
}

macro_rules! eval {
    ($x:expr) => {
        println!("{} = {}", stringify!($x), $x);
    };
}

fn cscale(c: Complex<f32>, m: Operator) -> Operator {
    m.component_mul(&Operator::from_element(c))
}

fn cscale4(c: Complex<f32>, m: SMatrix<Complex<f32>, 4, 4>) -> SMatrix<Complex<f32>, 4, 4> {
    m.component_mul(&SMatrix::<Complex<f32>, 4, 4>::from_element(c))
}

struct QubitSystemPsi {
    hamiltonian: Operator,
    psi: Vector2<Complex<f32>>,
    //t: f32
}

struct QubitSystem {
    hamiltonian: Operator,
    rho: Operator,
    //t: f32
    sqrt_eta: Complex<f32>,
    c_out_phased: Operator,
    rng: ThreadRng,
    measurement: Operator,
    c1: Operator,
    c2: Operator,
    c3: Operator,
    dW: f32,
}

#[inline]
fn commutator(a: Operator, b: Operator) -> Operator {
    a * b - b * a
}

#[inline]
fn anticommutator(a: Operator, b: Operator) -> Operator {
    a * b + b * a
}

#[inline]
fn half(operator: Operator) -> Operator {
    cscale(HALF, operator)
}

impl QubitSystem {
    fn new(A: Operator, delta_s: cf32, g: cf32, kappa_1: f32,
           kappa: f32, beta: Operator, delta_r: f32, eta: cf32,
           Phi: f32, gamma_dec: f32, gamma_phi: f32) -> Self {
        let alpha = cscale((2.0 * kappa_1).sqrt() / (kappa + I * delta_r), beta);
        let ddelta = delta_r - delta_s;
        let epsilon_s = g * g * ddelta / (kappa * kappa + ddelta * ddelta);
        let hamiltonian = cscale(delta_s * 0.5, sigma_z)
            + cscale(g, alpha * sigma_plus + alpha.conjugate() * sigma_minus)
            - cscale(epsilon_s, sigma_plus * sigma_minus);

        let gamma_p = 2.0 * g * g * kappa / (kappa * kappa + ddelta * ddelta);
        // let c_1 = cscale(gamma_p.sqrt(), sigma_minus);
        // let c_2 = cscale(gamma_p.sqrt(), sigma_plus);
        // let hamiltonian = hamilt17onian;
        //    - cscale(0.5*I, c_1.ad_mul(&c_1));
        //    - cscale(0.5*I, c_2.ad_mul(&c_2));

        let psi = Vector2::<Complex<f32>>::new(ONE, ZERO);

        let rho = psi.mul(&psi.transpose());
        //let rho = Vector4::<Complex<f32>>::new(
        //    *rho.index((0, 0)),
        //    *rho.index((0, 1)),
        //    *rho.index((1, 0)),
        //    *rho.index((1, 1)),
        //);

        //let hamiltonian = cscale(MINUS_I, hamiltonian);
        let c_out = cscale(
            (kappa_1 * 2.0).sqrt() * ONE,
            alpha - cscale(I * g / (kappa + I * ddelta), sigma_minus),
        );

        //let Ls = tensor_dot(A, A.adjoint())
        //    - tensor_dot(Operator::identity().scale(0.5), A * (A.adjoint()))
        //    - tensor_dot((A * A.adjoint()).scale(0.5), Operator::identity());

        let a = cscale((2.0 * kappa_1).sqrt() / (kappa + I * delta_r), beta)
            - cscale(I * g / (kappa + I * ddelta), sigma_minus);
        let c1 = cscale(ONE * (2.0 * kappa).sqrt(), a);
        let c2 = cscale(ONE * gamma_dec.sqrt(), sigma_minus);
        let c3 = cscale(ONE * (gamma_phi / 2.0).sqrt(), sigma_z);

        Self {
            hamiltonian,
            rho,
            sqrt_eta: eta.sqrt(),
            c_out_phased: c_out * ((I * Phi).exp()),
            rng: thread_rng(),
            measurement: A,
            c1,
            c2,
            c3,
            dW: 0.0,
        }
    }

    fn lindblad(&self, operator: Operator) -> Operator {
        operator * self.rho * operator.adjoint()
            - half(anticommutator(operator.adjoint() * operator, self.rho))
    }

    fn dv(&mut self, rho: Operator) -> Operator {
        let chi_rho = cscale(
            self.sqrt_eta,
            self.c_out_phased * self.rho + self.rho * self.c_out_phased.adjoint(),
        );
        let dY = chi_rho.trace() + self.dW;

        let a = self.measurement;
        cscale(MINUS_I, commutator(self.hamiltonian, rho))
            + self.lindblad(a)
            + self.lindblad(self.c1)
            + self.lindblad(self.c2)
            + self.lindblad(self.c3)
            + chi_rho * dY
    }

    fn runge_kutta(&mut self, time_step: f32) {
        let k_0 = self.dv(self.rho).scale(time_step);
        let k_1 = self.dv(self.rho + k_0.scale(0.5)).scale(time_step);
        let k_2 = self.dv(self.rho + k_1.scale(0.5)).scale(time_step);
        let k_3 = self.dv(self.rho + k_2).scale(time_step);
        self.rho += (k_1 + k_2 + (k_0 + k_3).scale(0.5)).scale(1.0 / 3.0);
    }
}

fn tensor_dot(m1: Operator, m2: Operator) -> SMatrix<Complex<f32>, 4, 4> {
    m1.kronecker(&m2.transpose())
}

enum PipeWriter {
    Closed,
    Opened(std::fs::File),
}

impl PipeWriter {
    fn open(path: &str) -> Self {
        if std::fs::metadata(path).is_ok() {
            Self::Opened(
                std::fs::File::options()
                    .write(true)
                    .read(false)
                    .open(path)
                    .unwrap(),
            )
        } else {
            Self::Closed
        }
    }

    fn write_vec3(&mut self, vec: Vector3<f32>) {
        match self {
            PipeWriter::Opened(pipe) => {
                let buf_data = vec.data.as_slice().as_ptr() as *const u8;
                let buf = unsafe {
                    std::slice::from_raw_parts(buf_data, std::mem::size_of::<Vector3<f32>>())
                };
                pipe.write(buf).unwrap();
            }
            PipeWriter::Closed => (),
        }
    }

    fn write_u32(&mut self, number: u32) {
        match self {
            PipeWriter::Opened(pipe) => {
                pipe.write(&number.to_le_bytes()).unwrap();
            }
            PipeWriter::Closed => (),
        }
    }

    fn is_opened(&self) -> bool {
        match self {
            PipeWriter::Closed => false,
            PipeWriter::Opened(_) => true,
        }
    }
}

fn simulate() {
    let timestamp = std::time::SystemTime::UNIX_EPOCH
        .elapsed()
        .unwrap()
        .as_secs();


    let mut parameter_file =
        std::fs::File::create(format!("results/{}_parameters.txt", timestamp)).unwrap();
    let mut data_file =
        std::fs::File::create(format!("results/{}_trajectories.csv", timestamp)).unwrap();

    let gamma = SMatrix::<Complex<f32>, 4, 4>::from_diagonal_element(ONE);

    let A = Operator::from_fn(|r, c| ONE * (r * c) as f32);
    let delta_s = ONE;
    let g = ONE * 2.0;
    let kappa_1 = 10.0;
    let kappa = 10.0;
    let beta = Operator::identity();
    let delta_r = 0.0;
    let eta = 0.5 * ONE;
    let Phi = 0.0;
    let gamma_dec = 1.0;
    let gamma_phi = 1.0;

    parameter_file.write_all(
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
    ).unwrap();

    let mut pipe = PipeWriter::open("/tmp/blochrender_pipe");
    pipe.write_u32(STEP_COUNT);

    for simulation in 0..1000 {
        // Initialize system
        let mut system = QubitSystem::new(
            A, delta_s, g, kappa_1, kappa, beta,
            delta_r, eta, Phi, gamma_dec, gamma_phi
        );

        // Start the timer.
        let now = std::time::Instant::now();

        // Do 2000 steps.
        for _ in 0..STEP_COUNT {

            // Write current state.
            data_file.write(format!("{}, ", system.rho[(0,0)].re).as_bytes()).unwrap();

            // Sample on the normal distribution.
            system.dW = system.rng.sample(StandardNormal);

            // Do the runge-kutta4 step.
            system.runge_kutta(dt);

            // Normalize rho.
            system.rho = cscale(1.0 / system.rho.trace(), system.rho);

            if pipe.is_opened() {
                pipe.write_vec3(bloch_vector(system.rho));
            }



            // Bloch v-O1ector magnitude test
            //let mag = bloch_vector(system.rho).magnitude();
            //let eigenvalues = system.rho.eigenvalues().unwrap();
            //assert!(((2.0 * eigenvalues[0].re - 1.0) - mag).abs() < 0.01, "Bloch vector magnitude was wrong.");

            //if  i % (1 << 12) == 0 {
            //  println!("Trace: {}", system.rho.trace());
            //  println!("rho: {}", system.rho);
            //}
        }

        println!("Sim ({simulation}) time: {} ms", now.elapsed().as_millis());

        // Write last state.
        data_file.write(format!("{}\n", system.rho[(0,0)].re).as_bytes()).unwrap();

        if pipe.is_opened() { break; }
    }
}

fn bloch_vector(rho: Operator) -> Vector3<f32> {
    Vector3::new(
        rho.index((0, 1)).re + rho.index((1, 0)).re,
        rho.index((0, 1)).im - rho.index((1, 0)).im,
        rho.index((0, 0)).re - rho.index((1, 1)).re,
    )
}

fn main() {
    println!("Starting simulation...");
    simulate();
}
