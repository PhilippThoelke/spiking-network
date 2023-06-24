use crate::neuron::{Neuron, NeuronState};
use rand::Rng;
use rand_distr::{Distribution, WeightedIndex};
use std::collections::HashSet;
use std::sync::mpsc::{self, Receiver};

pub struct Network {
    pub neurons: Vec<Neuron>,
    pub positions: Vec<(f32, f32)>,
    pub system_receiver: mpsc::Receiver<NeuronState>,
}

impl Network {
    pub fn new(aspect_ratio: f32) -> Network {
        let mut rng = rand::thread_rng();
        let (system_sender, system_receiver) = mpsc::channel();

        let mut positions: Vec<(f32, f32)> = Vec::new();
        let mut channels: Vec<(mpsc::Sender<usize>, Receiver<usize>)> = Vec::new();

        // initialize neuron positions and channels
        for _ in 0..crate::NUM_NEURONS {
            // random position in the rectangle with given aspect ratio and height equal to 1
            positions.push((rng.gen_range(0.0..aspect_ratio), rng.gen_range(0.0..1.0)));
            // create channel for the current neuron
            channels.push(mpsc::channel());
        }

        // split channels into senders (axons) and receivers (dendrites)
        let axons = channels.iter().map(|(s, _)| s.clone()).collect::<Vec<_>>();
        let dendrites = channels.into_iter().map(|(_, r)| r).collect::<Vec<_>>();

        // compute distance table
        let mut distances: Vec<Vec<f32>> = Vec::new();
        for i in 0..crate::NUM_NEURONS {
            let mut row: Vec<f32> = Vec::new();
            for j in 0..crate::NUM_NEURONS {
                if i == j {
                    row.push(std::f32::MAX);
                    continue;
                }

                let dx = positions[i].0 - positions[j].0;
                let dy = positions[i].1 - positions[j].1;
                row.push((dx * dx + dy * dy).sqrt());
            }
            distances.push(row);
        }

        // initialize neurons and connections between neurons
        let mut neurons: Vec<Neuron> = Vec::new();
        for (neuron_idx, dendrite_handle) in dendrites.into_iter().enumerate() {
            // map neuron distances to probability distribution
            let weights = distances[neuron_idx]
                .iter()
                .map(|d| {
                    if *d > crate::MAX_CONNECTION_DISTANCE {
                        0.0
                    } else {
                        1.0 / d
                    }
                })
                .collect::<Vec<f32>>();
            let distr = WeightedIndex::new(&weights).unwrap();

            // generate axonal connection indices
            let mut tries = 0;
            let mut target_idxs: HashSet<usize> = HashSet::new();
            while target_idxs.len() < crate::NUM_CONNECTIONS {
                // make sure we don't get stuck here
                if tries > crate::INIT_CONNECTION_RETRIES {
                    if target_idxs.len() == 0 {
                        panic!("No connections generated for {}", neuron_idx);
                    }
                    println!(
                        "Only generated {} connections for neuron {}",
                        target_idxs.len(),
                        neuron_idx
                    );
                    break;
                }

                // sample a new potential index
                let new_idx = distr.sample(&mut rng);
                if new_idx == neuron_idx {
                    tries += 1;
                    continue;
                }
                if !target_idxs.insert(new_idx) {
                    tries += 1;
                }
            }

            // initialize axon and dendrite containers
            let mut axon_handles: Vec<mpsc::Sender<usize>> = Vec::new();
            let mut axon_durations: Vec<std::time::Duration> = Vec::new();
            let mut dendrite_weights: Vec<f32> = Vec::new();

            for target_idx in target_idxs.into_iter() {
                axon_handles.push(axons[target_idx].clone());
                axon_durations.push(std::time::Duration::from_millis(
                    (distances[neuron_idx][target_idx] * 1000.0 / crate::ACTION_POTENTIAL_SPEED)
                        as u64,
                ));

                // generate random weight for the connection
                dendrite_weights
                    .push(rng.gen_range(crate::MIN_WEIGHT_INIT..crate::MAX_WEIGHT_INIT));
            }

            // create neuron
            neurons.push(Neuron::new(
                neuron_idx,
                (axons[neuron_idx].clone(), dendrite_handle),
                dendrite_weights,
                axon_handles,
                axon_durations,
                system_sender.clone(),
            ));
        }

        Network {
            neurons,
            positions,
            system_receiver,
        }
    }
}
