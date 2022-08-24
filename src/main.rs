use nalgebra::*;

#[allow(non_upper_case_globals)]
const sigma_z: Operator = Operator::new(
    1.0, 0.0,
    0.0,-1.0
);



type Operator = SMatrix::<f32, 2, 2>;
type Dimensions = usize;

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

fn main() {
    let v = SVector::<f32, 2>::from_element(1.0f32);

    const N: Dimensions = 10;
    let omega_a = 1.0;
    let omega_c = 1.25;
    let g = 0.05;

    let a  = identity!(2).kronecker(&lowering!(N));
    let sm = lowering!(2).kronecker(&identity!(N));
    let sz = sigma_z.kronecker(&identity!(N));
    
    let hamiltonian =
        0.5 * omega_a * sz +
        omega_c * a.adjoint() * a +
        g * (a.adjoint()*sm + a*sm.adjoint());

    //*m.index_mut((10,10)) = 2.0;
    
    macro_rules! eval {
        ($x:expr) => { println!("{} = {}", stringify!($x), $x); };
    }

    eval!(hamiltonian);
    eval!(sigma_z);
    
    //eval!(m*v);


    
    //eval!(m.index((2, 1..4)));

    eval!(hamiltonian.eigenvalues().unwrap());
    eval!(a);
    eval!(sm);
    eval!(sz);
    eval!(hamiltonian);
    //eval!(hamiltonian.eigenvalues().unwrap());
    //eval!(lowering::<3>())
}
