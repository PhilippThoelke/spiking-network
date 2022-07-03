use neuron::*;
use rand::seq::SliceRandom;
use rand::Rng;
use std::collections::{BinaryHeap, HashMap};
use std::sync::{mpsc, Arc, Mutex};
use std::thread;
use std::time::Duration;

mod neuron;

const NUM_NEURONS: usize = 1000;
const MIN_SYNAPSES_PER_NEURON: usize = 5;
const MAX_SYNAPSES_PER_NEURON: usize = 9;
const DECAY_RATE: f32 = 3e-4;
const RECOVERY_RATE: f32 = 1e-3;
const UNDERSHOOT_POTENTIAL: f32 = -0.5;
const AP_THRESHOLD: f32 = 1.1;
const AP_VELOCITY: f32 = 1e-3;

fn insert_ap(action_potentials: &mut BinaryHeap<ActionPotential>, new_ap: ActionPotential) {
    action_potentials.push(new_ap);
}

trait GetAB<T> {
    fn get_ab_mut(&mut self, a: usize, b: usize) -> (&mut T, &mut T);
}

impl GetAB<Neuron> for Vec<Neuron> {
    fn get_ab_mut(&mut self, a: usize, b: usize) -> (&mut Neuron, &mut Neuron) {
        assert_ne!(a, b);
        if a < b {
            let (aslice, bslice) = self.split_at_mut(b);
            (&mut aslice[a], &mut bslice[0])
        } else {
            let (bslice, aslice) = self.split_at_mut(a);
            (&mut aslice[0], &mut bslice[b])
        }
    }
}

fn main() {
    // initialize random number generator
    let mut rng = rand::thread_rng();

    // initialize membrane potentials to 0
    let neurons: Arc<Mutex<Vec<Neuron>>> = Arc::new(Mutex::new(Vec::new()));
    {
        // generate random positions for all neurons on a 2D plane
        let mut positions = [[0f32; 2]; NUM_NEURONS];
        for i in 0..positions.len() {
            positions[i][0] = rng.gen_range(-1f32..1f32);
            positions[i][1] = rng.gen_range(-1f32..1f32);
        }

        // compute the distance matrix between all pairs of neurons
        let mut distances = [[0f32; NUM_NEURONS]; NUM_NEURONS];
        for i in 0..positions.len() {
            for j in 0..positions.len() {
                distances[i][j] = ((positions[i][0] - positions[j][0]).powf(2.)
                    + (positions[i][1] - positions[j][1]).powf(2.))
                .sqrt();
            }
        }

        // generate random edges between neurons weighted by 1 / distance
        let mut edges: Vec<HashMap<usize, f32>> = Vec::new();
        let index_range = (0..NUM_NEURONS).collect::<Vec<_>>();
        for i in 0..NUM_NEURONS {
            edges.push(HashMap::new());
            // determine how many edges should be generated for this neuron
            let num_edges = rng.gen_range(MIN_SYNAPSES_PER_NEURON..MAX_SYNAPSES_PER_NEURON);
            while edges[i].len() < num_edges {
                // generate a random edge index
                let target = index_range
                    .choose_weighted(&mut rng, |idx| {
                        if distances[i][*idx] > 0. {
                            1. / distances[i][*idx]
                        } else {
                            0.
                        }
                    })
                    .unwrap();
                // append edge index if it is new and not a connection to itself
                if !edges[i].contains_key(target) && i != *target {
                    edges[i].insert(*target, 1.);
                }
            }
        }

        for i in 0..NUM_NEURONS {
            neurons.lock().unwrap().push(Neuron {
                index: i,
                potential: 0.,
                last_update: None,
                position: Vector2::new(positions[i][0], positions[i][1]),
                synapses_out: edges.remove(0),
            });
        }
    }

    // handle action potentials
    let (ap_tx, ap_rx): (
        mpsc::Sender<ActionPotential>,
        mpsc::Receiver<ActionPotential>,
    ) = mpsc::channel();

    // clone fields required in and outside of the action potential thread
    let external_neurons = Arc::clone(&neurons);
    let external_ap_tx = ap_tx.clone();

    // start thread handling action potentials
    thread::spawn(move || {
        // create a list of action potentials, sorted by duration
        let mut action_potentials: BinaryHeap<ActionPotential> = BinaryHeap::new();
        loop {
            if action_potentials.len() == 0 {
                // insert the newly received action potential into the sorted list
                insert_ap(&mut action_potentials, ap_rx.recv().unwrap());
            }

            if let Some(next_wait) = action_potentials.peek().unwrap().time_left() {
                // got interrupted by a newly generated action potential
                println!("trying to wait for {}ms", next_wait.as_millis());
                if let Ok(new_ap) = ap_rx.recv_timeout(next_wait) {
                    // insert the newly received action potential into the sorted list
                    insert_ap(&mut action_potentials, new_ap);
                    continue;
                }
            }
            // action potential reached the synapse
            let curr_ap = action_potentials.pop().unwrap();
            {
                let mut lock = neurons.lock().unwrap();
                let (_from, to) = lock.get_ab_mut(curr_ap.from, curr_ap.to);
                if to.receive_ap(&curr_ap) {
                    Neuron::fire(curr_ap.to, lock, &ap_tx);
                    // TODO: update synapse weights in _from
                }
            }
            println!(
                "handled an action potential from {} to {} (number of remaining APs: {})",
                curr_ap.from,
                curr_ap.to,
                action_potentials.len()
            )
        }
    });

    loop {
        // create action potentials in regular intervals
        {
            let lock = external_neurons.lock().unwrap();
            external_ap_tx
                .send(ActionPotential::new(&lock[0], &lock[1], 1.0))
                .unwrap();
        }
        thread::sleep(Duration::from_secs(1));
    }
}
