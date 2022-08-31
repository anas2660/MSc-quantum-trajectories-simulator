use std::{io::Write, ops::Mul, arch};

use nalgebra::*;



//type Operator = SMatrix::<f32, 2, 2>;
type Operator = SMatrix<Complex<f32>, 2, 2>;
//type Dimensions = usize;

const ONE: Complex<f32> = Complex::new(1.0, 0.0);
const MINUS_ONE: Complex<f32> = Complex::new(-1.0, 0.0);
const I: Complex<f32> = Complex::new(0.0, 1.0);
const MINUS_I: Complex<f32> = Complex::new(0.0, -1.0);
const ZERO: Complex<f32> = Complex::new(0.0, 0.0);

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
#[allow(non_upper_case_globals)]
const sigma_z: Operator = Operator::new(
    ONE,  ZERO,
    ZERO, MINUS_ONE
);

#[allow(non_upper_case_globals)]
const sigma_plus: Operator = Operator::new(
    ZERO, ONE,
    ZERO, ZERO
);

#[allow(non_upper_case_globals)]
const sigma_minus: Operator = Operator::new(
    ZERO, ZERO,
    ONE,  ZERO
);



// From qutip implementation
macro_rules! lowering {
    ($D:expr) => (SMatrix::<f32,$D,$D>::from_fn(|r,c| if r+1 == c { (c as f32).sqrt() } else { 0.0 } ));

    // qutip implementation
    //data = np.sqrt(np.arange(offset+1, N+offset, dtype=complex))
    //ind = np.arange(1,N, dtype=np.int32)
    //ptr = np.arange(N+1, dtype=np.int32)
    //ptr[-1] = N-1
    //return Qobj(fast_csr_matrix((data,ind,ptr),shape=(N,N)), isherm=False)
}

macro_rules! identity {
    ($D:expr) => (SMatrix::<f32,$D,$D>::identity());
}

macro_rules! eval {
    ($x:expr) => { println!("{} = {}", stringify!($x), $x); };
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

struct QubitSystemPsi {
    hamiltonian: Operator,
    psi: Vector2<Complex<f32>>,
    //t: f32
}

struct QubitSystem {
    hamiltonian: Operator,
    rho: Operator,
    //t: f32
}



impl QubitSystem {

    fn dv(&self, rho: Operator) -> Operator {
        cscale(MINUS_I, self.hamiltonian*rho - rho*self.hamiltonian)
    }

    fn runge_kutta(&mut self, dt: f32) {
        let k_0 = self.dv(self.rho).scale(dt);
        let k_1 = self.dv(self.rho + k_0.scale(0.5)).scale(dt);
        let k_2 = self.dv(self.rho + k_1.scale(0.5)).scale(dt);
        let k_3 = self.dv(self.rho + k_2).scale(dt);
        self.rho += (k_1 + k_2 + (k_0 + k_3).scale(0.5)).scale(1.0/3.0);
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



fn ours2() {
    let delta_s = ONE;
    let g = ONE*0.05;
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

    let rho = psi.mul(&psi.transpose());

    let mut system = QubitSystem { hamiltonian, rho };

    for i in 1..1000000 {
        system.runge_kutta(0.0001);
        if i % (1<<16) == 0 {
            println!("Trace: {}", system.rho.trace());
        }
    }


}




fn main() {
    let _v = Vector2::from_element(1.0f32);

    println!("Reference (Qutip) --------");
    reference();

    println!("Ours ---------------------", );
    ours2();
}
