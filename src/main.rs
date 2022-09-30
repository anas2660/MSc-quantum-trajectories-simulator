#![allow(non_snake_case)]
#![allow(non_upper_case_globals)]

use rand_distr::StandardNormal;
use std::ops::Mul;

use nalgebra::*;
use rand::{thread_rng, Rng, rngs::ThreadRng};

//type Operator = SMatrix::<f32, 2, 2>;
type Operator = SMatrix<Complex<f32>, 2, 2>;
//type Dimensions = usize;

const HALF: Complex<f32> = Complex::new(0.5, 0.0);
const ONE: Complex<f32> = Complex::new(1.0, 0.0);
const MINUS_ONE: Complex<f32> = Complex::new(-1.0, 0.0);
const I: Complex<f32> = Complex::new(0.0, 1.0);
const MINUS_I: Complex<f32> = Complex::new(0.0, -1.0);
const ZERO: Complex<f32> = Complex::new(0.0, 0.0);

const dt: f32 = 0.0001;


/*
#[allow(non_upper_case_globals)]
const sigma_x: Operator = Operator::new(
    0.0, 1.0,
    1.0, 0.0
);

#[allow(non_upper_case_globals)]
const sigma_y: Operator = Operator::new(
    ZERO, MINUS_I,
       I, ZERO
);
 */


// #[allow(non_upper_case_globals)]
// #[allow(non_upper_case_globals)]
// #[allow(non_upper_case_globals)]
const sigma_z: Operator = Operator::new(ONE, ZERO, ZERO, MINUS_ONE);
const sigma_plus: Operator = Operator::new(ZERO, ONE, ZERO, ZERO);
const sigma_minus: Operator = Operator::new(ZERO, ZERO, ONE, ZERO);


// From qutip implementation
macro_rules! lowering {
    ($D:expr) => {
        SMatrix::<f32, $D, $D>::from_fn(|r, c| if r + 1 == c { (c as f32).sqrt() } else { 0.0 })
    }; // qutip implementation
       //data = np.sqrt(np.arange(offset+1, N+offset, dtype=complex))
       //ind = np.arange(1,N, dtype=np.int32)
       //ptr = np.arange(N+1, dtype=np.int32)
       //ptr[-1] = N-1
       //return Qobj(fast_csr_matrix((data,ind,ptr),shape=(N,N)), isherm=False)
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

fn reference() {
    //  const N: Dimensions = 10;
    //let omega_a = 1.0;
    //let omega_c = 1.25;
    //let g = 0.05;
    //
    //    let a  = identity!(2).kronecker(&lowering!(N));
    //    let sm = lowering!(2).kronecker(&identity!(N));
    //    let sz = sigma_z.kronecker(&identity!(N));
    //
    //    let hamiltonian =
    //        0.5 * omega_a * sz +
    //        omega_c * a.adjoint() * a +
    //        g * (a.adjoint()*sm + a*sm.adjoint());
    //
    //
    //    eval!(hamiltonian);
    //    eval!(sigma_z);
    //    eval!(hamiltonian.eigenvalues().unwrap());
    //    eval!(a);
    //    eval!(sm);
    //    eval!(sz);
    //    eval!(hamiltonian);
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
    measurement: Operator,
    sqrt_eta: Complex<f32>,
    c_out_phased: Operator,
    rng: ThreadRng,
    dW: f32
}

#[inline]
fn commutator(a: Operator, b: Operator) -> Operator {
    a*b - b*a
}

#[inline]
fn anticommutator(a: Operator, b: Operator) -> Operator {
    a*b + b*a
}

impl QubitSystem {
    fn dv(&mut self, rho: Operator) -> Operator {
        let chi_rho = cscale(self.sqrt_eta, self.c_out_phased*self.rho + self.rho*self.c_out_phased.adjoint());
        let dY = chi_rho.trace() + self.dW;

        let a = self.measurement;
        cscale(MINUS_I, commutator(self.hamiltonian, rho))
            + (a*rho*a.adjoint() - cscale(HALF, anticommutator(a.adjoint()*a, rho)))
            + chi_rho*dY
    }

    fn runge_kutta(&mut self, time_step: f32) {
        let k_0 = self.dv(self.rho).scale(time_step);
        let k_1 = self.dv(self.rho + k_0.scale(0.5)).scale(time_step);
        let k_2 = self.dv(self.rho + k_1.scale(0.5)).scale(time_step);
        let k_3 = self.dv(self.rho + k_2).scale(time_step);
        self.rho += (k_1 + k_2 + (k_0 + k_3).scale(0.5)).scale(1.0 / 3.0);
    }
}

//fn euler_integration(system: QubitSystem, dt: f32) -> QubitSystem {
//    let mut result = system;
//    result.psi += result.dv(0.0, result.psi).scale(dt);
//    result
//}

fn ours() {
    /*
    let delta_s = ONE;
    let g = ONE;////ONE*0.65;
    let kappa_1 = 1.0;
    let kappa = 0.1;
    let beta = Operator::identity();
    let delta_r = 1.0;

    let alpha =
        cscale((2.0*kappa_1).sqrt()/(kappa+I*delta_r), beta);

    let ddelta = delta_r - delta_s;
    let epsilon_s = g*g * ddelta/(kappa*kappa + ddelta*ddelta);


    let hamiltonian =
        cscale(delta_s*0.5, sigma_z)
        + cscale(g, alpha*sigma_plus + alpha.conjugate() * sigma_minus)
        - cscale(epsilon_s, sigma_plus*sigma_minus);

    let gamma_p = 2.0*g*g*kappa/(kappa*kappa + ddelta*ddelta);

    // let c_1 = cscale(gamma_p.sqrt(), sigma_minus);
    // let c_2 = cscale(gamma_p.sqrt(), sigma_plus);
    // let hamiltonian = hamiltonian;
    //    - cscale(0.5*I, c_1.ad_mul(&c_1));
    //    - cscale(0.5*I, c_2.ad_mul(&c_2));

    let psi = Vector2::<Complex<f32>>::new(ONE, ZERO);

    let mut system = QubitSystem { hamiltonian, psi };

    let mut file = std::fs::File::create("data.csv").unwrap();
    file.write(b"index, x_real, x_imag, y_real, y_imag, magnitude, magnitude_squared\n").unwrap();

    for i in 0..50000 {
        let psi = &system.psi;
        ///////////file.write(format!(
        ///////////    "{i}, {}, {}, {}, {}, {}, {}\n",
        ///////////    psi.x.real(), psi.x.imaginary(),
        ///////////    psi.y.real(), psi.y.imaginary(),
        ///////////    psi.magnitude(), psi.magnitude_squared()
        ///////////).as_bytes()).unwrap();
        //println!("Step {i}, Psi: [{}, {}]", system.psi.x, system.psi.y);

        system = runge_kutta(system, 0.001);
    }

    */
}

fn tensor_dot(m1: Operator, m2: Operator) -> SMatrix<Complex<f32>, 4, 4> {
    m1.kronecker(&m2.transpose())
}

fn ours2() {
    let A = Operator::from_fn(|r, c| ONE * (r * c) as f32);
    let gamma = SMatrix::<Complex<f32>, 4, 4>::from_diagonal_element(ONE);

    let delta_s = ONE;
    let g = ONE * 0.05;
    let kappa_1 = 1.0;
    let kappa = 0.1;
    let beta = Operator::identity();
    let delta_r = 1.0;

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
    let c_out = cscale((kappa_1*2.0).sqrt()*ONE, alpha - cscale(I * g / (kappa + I*ddelta), sigma_minus));


    //let Ls = tensor_dot(A, A.adjoint())
    //    - tensor_dot(Operator::identity().scale(0.5), A * (A.adjoint()))
    //    - tensor_dot((A * A.adjoint()).scale(0.5), Operator::identity());

    let eta = 0.5*ONE;
    let Phi = 0.0;

    let mut system = QubitSystem {
        hamiltonian,
        rho,
        measurement: A,
        sqrt_eta: eta.sqrt(),
        c_out_phased: c_out*((I*Phi).exp()),
        rng: thread_rng(),
        dW: 0.0
    };


    for i in 1..1000000 {


        system.dW = system.rng.sample(StandardNormal); //(self.rng.gen::<f32>()*2.0-1.0)*dt;

        system.runge_kutta(dt);
        //system.rho += cscale(dt*ONE, chi_rho*dY);

        // Normalize
        let trace = system.rho.trace();
        system.rho = cscale(1.0/trace, system.rho);

        if system.rho[0].real().is_nan() {
            println!("{}", system.rho[0]);
            panic!();
        }

        if i % (1 << 16) == 0 {
            println!("Trace: {}", system.rho.trace());
            println!("rho: {}", system.rho);
        }
    }
}

fn bloch_sphere(rho: Vector4<Complex<f32>>) -> Vector3<f32> {
    Vector3::new(2.0 * rho.y.re, 2.0 * rho.y.im, 2.0 * rho.x.re - 1.0)
}

fn main() {
    let _v = Vector2::from_element(1.0f32);

    println!("Reference (Qutip) --------");
    reference();

    println!("Ours ---------------------",);
    ours2();
}
