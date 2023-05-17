use std::io::Write;
use crate::*;

const NOISE_FACTOR: fp = 1.0;

#[derive(Clone)]
pub struct SimulationResults {
    pub measurement_records: Option<MVec<Complex>>, //[[Complex; STEP_COUNT as usize]; SIMULATION_COUNT as usize]
    pub fidelity_probe_results: Option<Vec<Real>>,
    pub purity_results: Option<MVec<Real>>
}

pub fn simulate<const CUSTOM_RECORDS: bool, const RETURN_RECORDS: bool, const WRITE_FILES: bool>(initial_state: InitialState, config: &SimulationConfig, records: Option<&SimulationResults>)
     -> SimulationResults {

    let timestamp = std::time::SystemTime::UNIX_EPOCH
        .elapsed()
        .unwrap()
        .as_secs();

    // This if statement seems ridiculous, but it cannot resolve OutputFile::<WRITE_FILES>::create.
    let create_output_file = |name| {
        if WRITE_FILES {
            std::fs::File::create(format!("results/{timestamp}_{name}")).unwrap()
        } else if cfg!(windows) {
            // FIXME: I haven't tested if "nul" works on windows.
            std::fs::File::create("nul").unwrap()
        } else {
            std::fs::File::create("/dev/null").unwrap()
        }
    };

    // let mut parameter_file   = create_output_file("parameters.txt");
    let mut data_file        = create_output_file("trajectories.dat");
    let mut hist_file        = create_output_file("hist.dat");
    let mut current_file     = create_output_file("currents.dat");
    let mut final_state_file = create_output_file("final_state.dat");
    //let mut fidelity_file    = create_output_file("fidelity.dat");

    // FIXME // TODO:
//    parameter_file
//        .write_all(
//            format!(
//"let β = {β};
//let Δ_r = {Δ_r};
//let κ = {κ};
//let κ_1 = {κ_1};
//let η = {η};
//let Φ = {Φ};
//let γ_dec = {γ_dec};
//let γ_φ = {γ_φ};
//").as_bytes(),
//        ).unwrap();

    // metadata
    current_file.write_all(&(config.simulation_count() as u32 * Real::LANES as u32).to_le_bytes()).unwrap();

    //current_file.write(&STEP_COUNT.to_le_bytes()).unwrap();

    #[cfg(not(feature = "double-precision"))]
    data_file.write_all(&(0u32.to_le_bytes())).unwrap();
    #[cfg(feature="double-precision")]
    data_file.write_all(&(1u32.to_le_bytes())).unwrap();

    data_file.write_all(&(config.simulation_count() as u32 * Real::LANES as u32).to_le_bytes()).unwrap();
    data_file.write_all(&(Operator::SIZE as u32).to_le_bytes()).unwrap();
    data_file.write_all(&(config.step_count + 1).to_le_bytes()).unwrap();

    hist_file.write_all(&(Operator::SIZE as u32).to_le_bytes()).unwrap();
    hist_file.write_all(&(config.step_count + 1).to_le_bytes()).unwrap();
    hist_file.write_all(&(HIST_BIN_COUNT as u32 * Operator::SIZE as u32).to_le_bytes()).unwrap();

    struct ThreadResult {
        trajectory_sum: Vec<StateProbabilitiesSimd>, // [StateProbabilitiesSimd; STEP_COUNT as usize+1],
        trajectory_hist: MVec<Histogram<HIST_BIN_COUNT>>, // [[Histogram<HIST_BIN_COUNT>; Operator::SIZE]; STEP_COUNT as usize+1],
        current_sum: Vec<Complex>, //[Complex; SIMULATIONS_PER_THREAD as usize],
        final_states: Vec<StateProbabilitiesSimd>, //[StateProbabilitiesSimd; SIMULATIONS_PER_THREAD as usize],
        measurements: Option<MVec<Complex>>, // Option<Box<[[Complex; STEP_COUNT as usize]; SIMULATIONS_PER_THREAD as usize]>>,
        fidelities: Option<Vec<Real>>,    // Option<Box<[[Complex; STEP_COUNT as usize]; SIMULATIONS_PER_THREAD as usize]>>
        purities: Option<MVec<Real>>
    }

    // Combination of all thread results
    let mut trajectory_average = vec![StateProbabilitiesSimd::zero(); (config.step_count+1) as usize].into_boxed_slice();
    //let mut trajectory_histograms = alloc_zero!([[Histogram::<HIST_BIN_COUNT>; Operator::SIZE]; config.step_count as usize+1]);
    let mut trajectory_histograms = unsafe { MVec::<Histogram<HIST_BIN_COUNT>>::alloc_zeroed(config.step_count as usize+1, Operator::SIZE)};// alloc_zero!([[Histogram::<HIST_BIN_COUNT>; Operator::SIZE]; config.step_count as usize+1]);
    let mut measurements = unsafe {MVec::alloc_zeroed(config.simulation_count(), config.step_count as usize)};
    let mut fidelities = Vec::new();
    let mut purities = if config.measure_purity { Some(MVec::with_capacity(config.simulation_count(), config.step_count as usize)) } else { None };



    // let mut last_fidelities = Vec::new();

    // Communication channel to send thread results.
    let (tx, rx) = std::sync::mpsc::channel();

    std::thread::scope(|s|{
        for thread_id in 0..config.thread_count {
            let input_records_arc = records;
            let initial_state = initial_state.clone();
            let tx = tx.clone();
            std::thread::Builder::new().stack_size(10_000_000).name("baby bear".to_string()).spawn_scoped(s, move || {
                //s.spawn(move || {



                //let mut local = alloc_zero!(ThreadResult);
                let mut local = ThreadResult {
                    trajectory_sum: Vec::with_capacity(config.step_count as usize + 1),
                    trajectory_hist: unsafe { MVec::alloc_zeroed(config.step_count as usize+1, Operator::SIZE) },
                    current_sum: Vec::with_capacity(config.simulations_per_thread as usize),
                    final_states: Vec::with_capacity(config.simulations_per_thread as usize),
                    measurements: None,
                    fidelities: None,
                    purities: if config.measure_purity {
                        Some(MVec::with_capacity(config.simulations_per_thread as usize, config.step_count as usize))
                    } else {
                        None
                    }
                };
                unsafe {
                    local.trajectory_sum.set_len(config.step_count as usize + 1);
                    local.trajectory_sum.iter_mut().for_each(|s| *s = std::mem::zeroed() );
                }

                //let mut measurements =  alloc_zero!([[Complex; STEP_COUNT as usize]; SIMULATIONS_PER_THREAD as usize]);
                //let mut fidelities    = alloc_zero!([[Complex; STEP_COUNT as usize]; SIMULATIONS_PER_THREAD as usize]);
                let mut measurements = unsafe {MVec::alloc_zeroed(config.simulations_per_thread as usize, config.step_count as usize)};
                //let mut fidelities = unsafe {MVec::alloc_zeroed(config.simulations_per_thread as usize, config.step_count as usize)};
                let input_records = &input_records_arc;
                let mut fidelities = Vec::with_capacity(config.simulations_per_thread as usize);

                // Start the timer.
                let now = std::time::Instant::now();
                let mut total_section_time = 0;

                // Create initial system.
                let (initial_system, circuit) = QubitSystem::new(initial_state, config);

                // RNG
                let mut rng = thread_rng();
                let mut S = SGenerator::new(&mut rng);

                // Calculate ideal ρ.
                //let ideal_ρ = get_ideal_ρ(&initial_state);
                let ideal_ρ_operator = config.fidelity_probe.as_ref().map(|(_,s)| if let IdealState::Full(op) = s { Some(op) } else { None }).flatten();
                let ideal_ρ_2x2 = config.fidelity_probe.as_ref().map(|(_,s)| if let IdealState::Partial(op) = s { Some(op) } else { None }).flatten();

                // Calculate simulation step at which to probe the fidelity.
                let fidelity_probe_step = config.fidelity_probe.as_ref().map(|(t,_)| ((t/Δt).round() as usize).min(config.step_count as usize)-1);

                // sqrt(η/2) is from the definition of dZ.
                // sqrt(dt) is to make the variance σ = dt
                let sqrtηdt = Real::splat((config.η * 0.5).sqrt() * Δt.sqrt() * NOISE_FACTOR);
                if !config.silent {println!("sqrtηdt: {sqrtηdt:?}")};

                for simulation in 0..config.simulations_per_thread {
                    // Initialize system
                    let mut system = initial_system.clone();
                    let mut circuit_state = circuit.make_new_state();
                    let mut t = Δt;

                    let c = system.c_out_phased.c;
                    let x = c + c.dagger();
                    let y = MINUS_I*(c - c.dagger());

                    let mut J = ZERO;

                    let (mut H, mut steps_to_next_gate) = circuit_state.next();

                    let current_measurements = &mut measurements[simulation as usize];
                    //let fidelities = &mut fidelities[simulation as usize];

                    // Do 2000 steps.
                    for step in 0..config.step_count as usize {

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

                        // Get a dZ value.
                        if CUSTOM_RECORDS {
                            let dI = input_records.as_ref().unwrap().measurement_records.as_ref().unwrap()[thread_id as usize * config.simulations_per_thread as usize + simulation as usize][step];

                            let to_sub = Complex {
                                real: (x*system.ρ).trace().real,
                                imag: (y*system.ρ).trace().real
                            };

                            const DT_OVER_SQRT2: Real = Real::from_array([Δt/SQRT_2; Real::LANES]);
                            system.dZ = DT_OVER_SQRT2 * &(dI - to_sub);
                        } else {
                            // Sample on the normal distribution.
                            for lane in 0..Real::LANES {
                                system.dZ.real[lane] = rng.sample::<fp, StandardNormal>(StandardNormal);
                                system.dZ.imag[lane] = rng.sample::<fp, StandardNormal>(StandardNormal);
                            }
                            system.dZ *= &sqrtηdt;
                        }

                        // Do the stochastic rk2 step.
                        system.srk2(H, [S.gen(&mut rng), S.gen(&mut rng)]);
                        /////system.euler(H);

                        // Normalize rho.
                        system.ρ.normalize();

                        // Compute fidelity.
                        if Operator::SIZE == 2 {
                            if let Some(probe_step) = fidelity_probe_step {
                                if step == probe_step {
                                    let ideal = ideal_ρ_operator.as_ref().expect("Fidelity: For QUBIT_SIZE = 1, IdealState has to be given IdealState::Full.");
                                    fidelities.push(system.ρ.fidelity_2x2(ideal));
                                }
                            }
                        } else if Operator::SIZE == 4 {
                            if let Some(probe_step) = fidelity_probe_step {
                                if step == probe_step {
                                    let ideal = ideal_ρ_2x2.as_ref().expect("Fidelity: For QUBIT_SIZE = 2, IdealState has to be given IdealState::Partial.");
                                    fidelities.push(system.ρ.fidelity_4x4_partial_traced(ideal));
                                }
                            }
                        }

                        // Compute purity
                        if let Some(purities) = local.purities.as_mut() {
                            let ρ = &system.ρ;
                            purities.push((ρ*ρ).trace().real);
                        }

                        // Compute current.
                        const SQRT2_OVER_DT: Real = Real::from_array([SQRT_2/Δt; Real::LANES]);
                        let dI = Complex {
                            real: (x*system.ρ).trace().real + SQRT2_OVER_DT * system.dZ.real,
                            imag: (y*system.ρ).trace().real + SQRT2_OVER_DT * system.dZ.imag
                        };

                        J += dI;
                        current_measurements[step] = dI;

                        t += Δt;
                    }

                    // Write last state.
                    let P = system.ρ.get_probabilites_simd();
                    local.trajectory_sum[config.step_count as usize].add(&P);
                    for (i, p) in P.v.iter().enumerate() {
                        local.trajectory_hist[config.step_count as usize][i].add_values(p);
                    }

                    //local.current_sum[simulation as usize] = J;
                    //local.final_states[simulation as usize] = P;
                    local.current_sum.push(J);
                    local.final_states.push(P);
                }

                let total_time = now.elapsed().as_millis();

                if !config.silent {
                    println!("Thread {thread_id} finished {} simulations in {total_time} ms ({} μs/sim) (total section time {} ms)",
                             config.simulations_per_thread,
                             (total_time*1000)/(config.simulations_per_thread*Real::LANES as u32) as u128,
                             total_section_time/1000);
                }

                for s in local.trajectory_sum.iter_mut() {
                    s.divide(config.simulations_per_thread as fp);
                }

                local.measurements = if RETURN_RECORDS { Some(measurements) } else { None };
                local.fidelities = if Operator::SIZE == 2 && config.fidelity_probe.is_some() { Some(fidelities) } else { None };

                tx.send(local).unwrap();
            }).unwrap();
        }

        // Combine thread results.
        for i in 0..(config.thread_count as usize) {
            //let local = tt.join().unwrap();
            let local = rx.recv().unwrap();

            for (s, ls) in trajectory_average.iter_mut().zip(local.trajectory_sum.iter()) {
                s.add(ls);
            }

            //for (i, histograms) in trajectory_histograms.iter_mut().enumerate() {
            for i in 0..trajectory_histograms.row_count() {
                let local_histograms = &local.trajectory_hist[i];
                let histograms = &mut trajectory_histograms[i];
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

            for the_final_state in local.final_states.iter() {
                let final_state_array = the_final_state.as_array();
                for final_state in final_state_array {
                    final_state_file.write_all(&final_state.to_le_bytes()).unwrap();
                }
            }

            if let Some(m) = local.measurements {
                let a = i * config.simulations_per_thread as usize;
                let b = (i+1) * config.simulations_per_thread as usize;
                measurements[a..b].copy_from_slice(m.as_slice());
            }

            if let Some(f) = local.fidelities {
                fidelities.extend_from_slice(f.as_slice());
            }

            if let Some(p) = local.purities {
                purities.as_mut().unwrap().extend_from_slice(p.as_slice());
            }
        }
    });


    for s in trajectory_average.iter_mut() {
        s.divide(config.thread_count as fp);
        data_file.write_all(&s.average().to_le_bytes()).unwrap();
    }

    // TODO: actual time
    // TODO: fix magic number (simcount)
    for i in 0..=config.step_count {
        let t = i as fp * Δt;
        data_file.write_all(&t.to_le_bytes()).unwrap();
    }

    let mut buffer = Vec::with_capacity(std::mem::size_of::<u32>() * HIST_BIN_COUNT * Operator::SIZE * config.step_count as usize);
    //for state in trajectory_histograms.iter() {
    //    for hist in state.iter() {
    for hist in trajectory_histograms.as_slice().iter() {
        for bin in hist.bins.iter().rev() {
            buffer.extend_from_slice(&bin.to_le_bytes());
        }
    }
    hist_file.write_all(&buffer).unwrap();

    // buffer.clear();
    // if !fidelities.is_empty() {
    //     for lane in 0..V::LANES {
    //         for sim in fidelities.iter() {
    //             buffer.extend_from_slice(&sim[lane].to_le_bytes());
    //             //for f in sim.iter() {
    //             //    buffer.extend_from_slice(&f.real[lane].to_le_bytes());
    //             //}
    //         }
    //     }
    //     fidelity_file.write_all(&buffer).unwrap();
    // }

    SimulationResults {
        measurement_records: if RETURN_RECORDS { Some(measurements) } else { None },
        fidelity_probe_results: if Operator::SIZE == 2 && config.fidelity_probe.is_some() { Some(fidelities) } else {None},
        purity_results: purities
    }
}
