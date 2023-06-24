use network::Network;
use std::time::Duration;

mod network;
mod neuron;

const NUM_NEURONS: usize = 100;
const NUM_CONNECTIONS: usize = 5;

const ACTION_POTENTIAL_THRESHOLD: f32 = 1.0;
const ACTION_POTENTIAL_SPEED: f32 = 0.02;
const MEMBRANE_DECAY_RATE: f32 = 0.7;

const REFRACTORY_POTENTIAL: f32 = -0.5;
const HARD_REFRACTORY_DURATION: Duration = Duration::from_millis(50);
const REFRACTORY_DECAY_RATE: f32 = 0.2;

const MAX_CONNECTION_DISTANCE: f32 = 0.3;
const MIN_WEIGHT_INIT: f32 = -1.0;
const MAX_WEIGHT_INIT: f32 = 1.5;
const INIT_CONNECTION_RETRIES: usize = 50;

fn main() {
    // initialize a spiking network
    let net = Network::new(1.0);

    // fire neuron 0 once to start off the network
    net.neurons[0].dendrite.send(std::usize::MAX).unwrap();

    loop {
        // listen for action potential events from the network
        match net.system_receiver.recv() {
            Ok(state) => {
                if state.firing {
                    println!("neuron {} is firing", state.idx);
                }
            }
            Err(_) => {
                println!("System channel closed");
                break;
            }
        }
    }

    // wait until all neuron threads have shut down
    for neuron in net.neurons {
        neuron.thread.join().unwrap();
    }
}
