use crate::{*,
    operator::Operator,
    lindblad::Lindblad,
    num::*,
    matrix::Matrix,
    initial_state::InitialState,
    config::SimulationConfig,
    hamiltonian::QuantumCircuit,
    helpers::*
};

const ω_not: f64 = 4000.0;

#[derive(Clone)]
pub struct QubitSystem {
    pub ρ: Operator,
    pub c_out_phased: Lindblad,
    c1: Lindblad,
    c2: Lindblad,
    c3: [Lindblad; Operator::QUBIT_COUNT],
    pub dZ: Complex,
}


impl QubitSystem {

    pub fn new(initial_state: InitialState, config: &SimulationConfig) -> (Self, QuantumCircuit) {
        let c = config;

        #[allow(unused_variables)]
        let σ_plus   = Matrix::new(0.0, 1.0,  0.0,  0.0);
        let σ_x      = Matrix::new(0.0, 1.0,  1.0,  0.0);
        let σ_y      = SC!(0.0,1.0) * Matrix::new(0.0, -1.0, 1.0,  0.0);
        let σ_z      = Matrix::new(1.0, 0.0,  0.0, -1.0);
        let σ_minus  = Matrix::new(0.0, 0.0,  1.0,  0.0);
        let hadamard = Matrix::new(1.0, 1.0,  1.0, -1.0);
        let identity = Operator::identity();

        // TODO: Add offsets to config
        //const Δ_s:  [fp; 2] = [Δ_s_0+0., Δ_s_0-0.]; // |ω_r - ω_s|
        //let Δ_b: [fp; 2] = [20.0, 20.0]; //  ω_b - ω_s
        //let Δ_r: fp = 0.0;
        //let χ: [fp; 2] = [config.χ_0 + 0.00, config.χ_0 - 0.00];
        //let g: [fp; 2] = [config.g_0, config.g_0];

        //let Δ_b: [fp; Operator::QUBIT_COUNT] = [20.0; Operator::QUBIT_COUNT]; //  ω_b - ω_s

        //let Δ_r: fp = 0.0;
        let mut χ: [fp; Operator::QUBIT_COUNT] = [config.χ_0; Operator::QUBIT_COUNT];
        let g: [fp; Operator::QUBIT_COUNT] = [config.g_0; Operator::QUBIT_COUNT];

        //χ[0] -= 0.3;
        //χ[1] += 0.3;
        if let Some(offsets) = config.chi_offsets.as_ref() {
            for offset in offsets.iter() {
                if let Some(chi) = χ.get_mut(offset.index) {
                    *chi += offset.offset;
                }
            }
        }

        //let Δ_b = g.zip(χ).map(|(g,χ)| g*g/χ);
        let Δ_s = g.zip(χ).map(|(g,χ)| g*g/χ);
        let Δ_b = Δ_s.map(|Δ_s| config.Δ_br+Δ_s );

        // DISPERSIVE
        let a = Operator::from_fn(|r, c| {
            (r==c) as u32 as fp
                * (2.*config.κ_1).sqrt() * config.β
                / (config.Δ_r + config.κ + I*χ.iter().enumerate().fold(0.0, |acc, (n, χ_n)| {
                    acc - (((c>>n)&1) as i32 * 2 - 1) as fp * χ_n // TODO: Check this
            }))
        });

        let N = a.dagger() * a;

        // Hamiltonian
        let term2 = apply_and_scale_individually(&Δ_b.zip(χ).map(|(Δ_bs, χ_s)| /**/0.5*Δ_bs*identity + χ_s*N), &σ_z);

        let σ_m = apply_individually_parts(&σ_minus);
        let σ_p = apply_individually_parts(&σ_plus);

        let mut term3 = Operator::zero();

        for s in 0..Operator::QUBIT_COUNT {
            for q in 0..Operator::QUBIT_COUNT {
                let c = χ[s] * (g[q]/g[s]);
                term3 += c * σ_p[s] * σ_m[q] + c * σ_p[q] * σ_m[s];
            }
        }

        let H = c.Δ_br * N + term2 + term3
            + I*(2.0 * c.κ_1).sqrt() * (c.β * a.dagger() - c.β.conjugate() * a); // Detuning

        let zero = Matrix::vector(&[1., 0.]);
        let one  = Matrix::vector(&[0., 1.]);

        let ket1 = &one;
        let bra1 = &one.transpose();
        let ket0 = &zero;
        let bra0 = &zero.transpose();

        //// CNOT 0, 1
        //let ω = 1000.0;
        //let cnot = ω * (ket1*bra1).kronecker(&(ket1*bra0 + ket0*bra1)).to_operator();

        // NOT 0
        let not = apply_individually_parts(&(ω_not * &σ_x));

        // Remove any non-hermitian numerical error
        let H = 0.5*(H + H.conjugate());
        if !config.silent { println!("H:\n{H}") };

        // Lindblad terms
        let c_out = (c.κ_1 * 2.0).sqrt() * a - c.β * identity;

        let T = Δt * (c.step_count-1) as fp;

        let circuit = QuantumCircuit::new(&[
            (H + not[0], T*0.5),
            //(H, T*0.5)
        ], c);

        (
            Self {
                ρ: initial_state.into(),
                c_out_phased: Lindblad::new(c_out * ((I * c.Φ).exp())),
                c1: Lindblad::new((2.0 * c.κ).sqrt() * a),
                c2: Lindblad::new(apply_individually(&(c.γ_dec.sqrt() * &σ_minus))),
                c3: apply_individually_parts(&((c.γ_φ / 2.0).sqrt() * &σ_z)).map(Lindblad::new),
                dZ: Complex::new(0.0, 0.0),
            },
            circuit
        )
    }

    fn lindblad_terms(&self, ρ: &Operator) -> Operator {
        let mut lt = Operator::lindblad_term(ρ, &self.c1); // Photon field transmission/losses

        lt.lindblad(ρ, &self.c_out_phased);

        for c3 in self.c3.iter() {
            lt.lindblad(ρ, c3);
        }

        lt
        // .lindblad(ρ, &self.c2) // Decay to ground state
        // .lindblad2(ρ, &self.c_out_phased) // Second order
    }

    fn stochastic(&self, ρ: &Operator) -> [Operator; 2] {
        let ρ_norm = *ρ.clone().normalize();
        let r = self.c_out_phased.c;
        let rρ = r*ρ*FRAC_1_SQRT_2;
        let a1 = rρ - rρ.trace()*ρ_norm;
        let a2 = a1.dagger();
        [ a1+a2, I*(a2-a1) ]
    }

    pub fn srk2(&mut self, H: &Hamiltonian, S: [SV32; 2]) {
        let ΔWx = self.dZ.real;
        let ΔWy = self.dZ.imag;

        let sqrtΔt = Real::splat(Δt.sqrt());

        let S = S.map(convert_sv32_to_real);

        let offset = S.map(|s| s * sqrtΔt );

        let bt = self.stochastic(&self.ρ);
        let K1 = Δt * self.lindblad_terms(&self.ρ)
            + (ΔWx - offset[0]) * bt[0]
            + (ΔWy - offset[1]) * bt[1];

        let evolved_ρ = H.apply(&self.ρ);

        let new_ρ = evolved_ρ + K1;
        // new_ρ.normalize();

        let bt = self.stochastic(&new_ρ);
        let K2 = Δt * self.lindblad_terms(&new_ρ)
            + (ΔWx + offset[0]) * bt[0]
            + (ΔWy + offset[1]) * bt[1];

        self.ρ = evolved_ρ + 0.5 * (K1 + K2);
    }

    pub fn euler(&mut self, H: &Hamiltonian) {
        let a = self.lindblad_terms(&self.ρ);
        let b = self.stochastic(&self.ρ);
        self.ρ = H.apply(&self.ρ) + a * Δt + self.dZ.real*b[0] + self.dZ.imag*b[1];
    }
}
