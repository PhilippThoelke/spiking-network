use rand::seq::SliceRandom;
use rand::Rng;

fn main() {
    const NUM_NEURONS: usize = 100;
    const MIN_SYNAPSES_PER_NEURON: usize = 1;
    const MAX_SYNAPSES_PER_NEURON: usize = 5;

    // initialize random number generator
    let mut rng = rand::thread_rng();

    // initialize membrane potentials to 0
    let mut potentials = [0f32; NUM_NEURONS];

    // generate random positions for all neurons on a 2D plane
    let mut positions = [[0f32; 2]; NUM_NEURONS];
    for i in 0..positions.len() {
        positions[i][0] = rng.gen_range(-1f32..1f32);
        positions[i][1] = rng.gen_range(-1f32..1f32);
    }

    positions[0][0] = 100.;
    positions[1][0] = 100.;
    positions[2][0] = 100.;

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
    let mut edges: Vec<Vec<usize>> = Vec::new();
    let index_range = (0..NUM_NEURONS).collect::<Vec<_>>();
    for i in 0..NUM_NEURONS {
        edges.push(Vec::new());
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
            if !edges[i].contains(target) && i != *target {
                edges[i].push(*target);
            }
        }
    }
    drop(index_range);
}
