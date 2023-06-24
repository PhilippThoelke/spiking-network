use std::cmp::Reverse;
use std::collections::BinaryHeap;
use std::sync::mpsc;
use std::thread;
use std::time::{Duration, Instant};

pub fn update_neuron(state: &mut NeuronState, incoming: Option<f32>) {
    let now = Instant::now();
    let time_delta = now - state.last_update;

    // reset firing state
    state.firing = false;

    if time_delta < crate::HARD_REFRACTORY_DURATION {
        // hard refractory period
        return;
    }

    // update last_update time
    state.last_update = now;

    // update membrane potential
    if state.membrane_potential > 0.0 {
        // linear decay of the membrane potential
        state.membrane_potential -=
            (time_delta.as_secs_f32() * crate::MEMBRANE_DECAY_RATE).min(state.membrane_potential);
    } else {
        // linear decay of the refractory overshoot
        state.membrane_potential += (time_delta.as_secs_f32() * crate::REFRACTORY_DECAY_RATE)
            .min(-state.membrane_potential);
    }

    // add incoming potential
    if let Some(incoming) = incoming {
        state.membrane_potential += incoming;

        if state.membrane_potential >= crate::ACTION_POTENTIAL_THRESHOLD {
            // fire an action potential
            state.firing = true;
            state.membrane_potential = crate::REFRACTORY_POTENTIAL;
        }
    }
}

#[derive(PartialEq, Eq, PartialOrd, Ord, Clone)]
pub struct ActionPotential {
    arrival: Instant,
    target_idx: usize,
}

#[derive(Clone)]
pub struct NeuronState {
    pub idx: usize,
    pub firing: bool,
    pub membrane_potential: f32,
    pub last_update: Instant,
    pub pending_action_potentials: BinaryHeap<Reverse<ActionPotential>>,
}

pub struct Neuron {
    pub dendrite: mpsc::Sender<usize>,
    pub thread: thread::JoinHandle<()>,
}

impl Neuron {
    pub fn new(
        idx: usize,
        dendrite_handles: (mpsc::Sender<usize>, mpsc::Receiver<usize>),
        dendrite_weights: Vec<f32>,
        axon_handles: Vec<mpsc::Sender<usize>>,
        axon_durations: Vec<Duration>,
        system_handle: mpsc::Sender<NeuronState>,
    ) -> Neuron {
        let thread = thread::spawn(move || {
            let mut state = NeuronState {
                idx,
                firing: false,
                membrane_potential: 0.0,
                last_update: Instant::now() - crate::HARD_REFRACTORY_DURATION,
                pending_action_potentials: BinaryHeap::new(),
            };

            loop {
                // wait until we receive an action potential or a pending one arrives
                match dendrite_handles.1.recv_timeout(
                    state
                        .pending_action_potentials
                        .peek()
                        .map_or(Duration::from_secs(std::u64::MAX), |ap| {
                            ap.0.arrival - Instant::now()
                        }),
                ) {
                    Ok(target_idx) => {
                        /////////////////////////////////////////////////////
                        // we received an action potential (other -> self) //
                        /////////////////////////////////////////////////////

                        // get the weight of the incoming signal, simply fire if no weight is set
                        let weight = dendrite_weights
                            .get(target_idx)
                            .copied()
                            .unwrap_or(crate::ACTION_POTENTIAL_THRESHOLD);
                        // update own state with the incoming signal
                        update_neuron(&mut state, Some(weight));

                        // check if we are firing
                        if state.firing {
                            // schedule action potentials for all axonal connections
                            for i in 0..axon_handles.len() {
                                let ap = ActionPotential {
                                    arrival: Instant::now() + axon_durations[i],
                                    target_idx: i,
                                };
                                state.pending_action_potentials.push(Reverse(ap));
                            }
                        }

                        // update system about the current state
                        match system_handle.send(state.clone()) {
                            Ok(_) => {}
                            Err(_) => {
                                println!("System not reachable from neuron {}", state.idx);
                                break;
                            }
                        }
                    }
                    Err(mpsc::RecvTimeoutError::Timeout) => {
                        ///////////////////////////////////////////////////////////////////
                        // action potential arrived at the target neuron (self -> other) //
                        ///////////////////////////////////////////////////////////////////

                        let ap = state.pending_action_potentials.pop().unwrap().0;
                        match axon_handles[ap.target_idx].send(state.idx) {
                            Ok(_) => {}
                            Err(_) => {
                                println!(
                                    "Neuron {} failed to receive action potential from neuron {}",
                                    ap.target_idx, state.idx
                                );
                                break;
                            }
                        }
                    }
                    Err(mpsc::RecvTimeoutError::Disconnected) => {
                        ///////////////////////////////////////////
                        // dendrite disconnected, neuron is dead //
                        ///////////////////////////////////////////
                        println!("Neuron {} dendrite disconnected", state.idx);
                        break;
                    }
                }
            }
        });

        Neuron {
            dendrite: dendrite_handles.0,
            thread,
        }
    }
}
