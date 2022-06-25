use rand::seq::SliceRandom;
use rand::Rng;
use std::cmp::Ordering;
use std::collections::{BinaryHeap, HashMap};
use std::sync::mpsc;
use std::thread;
use std::time::{Duration, Instant};

struct Point {
    x: f32,
    y: f32,
}

struct Neuron {
    index: usize,
    position: Point,
    potential: f32,
    synapses_out: HashMap<usize, f32>,
    last_update: Option<Instant>,
}

impl Neuron {
    fn update(&mut self, decay_rate: f32, recovery_rate: f32, change: Option<f32>) {
        if let Some(last_update) = self.last_update {
            let elapsed = last_update.elapsed().as_millis();
            if self.potential > 0. {
                self.potential = (self.potential - decay_rate * (elapsed as f32)).max(0.);
            } else if self.potential < 0. {
                self.potential = (self.potential + recovery_rate * (elapsed as f32)).min(0.);
            }
        }

        self.potential += change.unwrap_or_default();
        self.last_update = Some(Instant::now());
    }

    fn receive_ap(&mut self, ap: &ActionPotential, decay_rate: f32, recovery_rate: f32) {
        self.update(decay_rate, recovery_rate, Some(ap.strength));
        // TODO: evaluate if a new AP should be fired
    }
}

struct ActionPotential {
    duration: Duration,
    strength: f32,
    from: usize,
    to: usize,
    start_time: Instant,
}

impl ActionPotential {
    fn time_left(&self) -> Option<Duration> {
        self.duration.checked_sub(self.start_time.elapsed())
    }
}

impl Ord for ActionPotential {
    fn cmp(&self, other: &Self) -> Ordering {
        other
            .time_left()
            .unwrap_or(Duration::ZERO)
            .cmp(&self.time_left().unwrap_or(Duration::ZERO))
    }
}

impl PartialOrd for ActionPotential {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl PartialEq for ActionPotential {
    fn eq(&self, other: &Self) -> bool {
        self.duration == other.duration
    }
}

impl Eq for ActionPotential {}

fn insert_ap(action_potentials: &mut BinaryHeap<ActionPotential>, new_ap: ActionPotential) {
    action_potentials.push(new_ap);
}

fn main() {
    const NUM_NEURONS: usize = 100;
    const MIN_SYNAPSES_PER_NEURON: usize = 1;
    const MAX_SYNAPSES_PER_NEURON: usize = 5;
    const DECAY_RATE: f32 = 0.01;
    const RECOVERY_RATE: f32 = 0.01;

    // initialize random number generator
    let mut rng = rand::thread_rng();

    // initialize membrane potentials to 0
    let mut neurons: Vec<Neuron> = Vec::new();
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
            neurons.push(Neuron {
                index: i,
                potential: 0.,
                last_update: None,
                position: Point {
                    x: positions[i][0],
                    y: positions[i][1],
                },
                synapses_out: edges.remove(0),
            });
        }
    }

    // handle action potentials
    let (ap_tx, ap_rx): (
        mpsc::Sender<ActionPotential>,
        mpsc::Receiver<ActionPotential>,
    ) = mpsc::channel();

    let handle = thread::spawn(move || {
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
            // TODO: potentially generate some new action potentials
            neurons[curr_ap.to].receive_ap(&curr_ap, DECAY_RATE, RECOVERY_RATE);
            println!(
                "handled an action potential from {} to {} (number of remaining APs: {})",
                curr_ap.from,
                curr_ap.to,
                action_potentials.len()
            )
        }
    });

    // send some example action potentials for testing
    let ap = ActionPotential {
        from: 0,
        to: 0,
        strength: 1., // TODO: take this from Neuron.synapses_out[i]
        duration: Duration::from_millis(1000),
        start_time: Instant::now(),
    };
    ap_tx.send(ap).unwrap();

    let ap = ActionPotential {
        from: 3,
        to: 2,
        strength: 1., // TODO: take this from Neuron.synapses_out[i]
        duration: Duration::from_millis(1500),
        start_time: Instant::now(),
    };
    ap_tx.send(ap).unwrap();

    let ap = ActionPotential {
        from: 7,
        to: 5,
        strength: 1., // TODO: take this from Neuron.synapses_out[i]
        duration: Duration::from_millis(750),
        start_time: Instant::now(),
    };
    ap_tx.send(ap).unwrap();

    handle.join().unwrap();
}
