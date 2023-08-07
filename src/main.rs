use nannou::prelude::*;
use nannou_egui::{self, egui, Egui};
use network::Network;
use std::time::Duration;
use utils::to_screen_coords;

mod network;
mod neuron;
mod utils;

///////////////////////////////////
// Spiking Network Configuration //
///////////////////////////////////
const NUM_NEURONS: usize = 800;
const NUM_CONNECTIONS: usize = 4;

const ACTION_POTENTIAL_THRESHOLD: f32 = 1.0;
const ACTION_POTENTIAL_SPEED: f32 = 0.25;
const MEMBRANE_DECAY_RATE: f32 = 0.3;

const REFRACTORY_POTENTIAL: f32 = -0.7;
const HARD_REFRACTORY_DURATION: Duration = Duration::from_millis(250);
const REFRACTORY_DECAY_RATE: f32 = 0.5;

const MAX_CONNECTION_DISTANCE: f32 = 0.1;
const MIN_WEIGHT_INIT: f32 = -0.3;
const MAX_WEIGHT_INIT: f32 = 1.2;
const INIT_CONNECTION_RETRIES: usize = 50;

////////////////////
// Self-balancing //
////////////////////
const SELF_BALANCE: bool = true;
const MIN_ACTIVE: usize = 50;
const MAX_ACTIVE: usize = 200;

////////////////////////////
// Visualization Settings //
////////////////////////////
const WINDOW_SIZE: (u32, u32) = (1200, 800);
const ASPECT_RATIO: f32 = (WINDOW_SIZE.0 as f32) / (WINDOW_SIZE.1 as f32);
const NEURON_RADIUS: f32 = 15.0;
const DRAW_EVERYTHING: bool = true;

struct Model {
    net: Network,
    neuron_states: Vec<Option<neuron::NeuronState>>,
    egui: Egui,
    mean: f32,
    std: f32,
}

fn model(app: &App) -> Model {
    // create window
    let win_id = app
        .new_window()
        .size(WINDOW_SIZE.0, WINDOW_SIZE.1)
        .view(view)
        .raw_event(raw_window_event)
        .build()
        .unwrap();
    let window = app.window(win_id).unwrap();
    window.set_resizable(false);

    // initialize egui
    let egui = Egui::from_window(&window);

    // initialize network
    let net = Network::new(ASPECT_RATIO);

    Model {
        net,
        neuron_states: vec![None; NUM_NEURONS],
        egui,
        mean: 0.0,
        std: 1.0,
    }
}

fn event(app: &App, model: &mut Model, event: Event) {
    // get mouse click events
    match event {
        Event::WindowEvent {
            simple: Some(event),
            ..
        } => match event {
            MousePressed(MouseButton::Left) => {
                let pos = app.mouse.position();
                for (i, neuron_pos) in model.net.positions.iter().enumerate() {
                    // compute distance between mouse pointer and neuron
                    let dist = pos.distance(to_screen_coords(
                        vec2(neuron_pos.0, neuron_pos.1),
                        app.window_rect(),
                    ));

                    if dist < NEURON_RADIUS {
                        model.net.neurons[i]
                            .dendrite
                            .as_ref()
                            .unwrap()
                            .send(std::usize::MAX)
                            .unwrap();
                    }
                }
            }
            _ => (),
        },
        _ => (),
    }
}

fn update(_app: &App, model: &mut Model, _update: Update) {
    // listen for action potential events from the network
    model.net.system_receiver.try_iter().for_each(|state| {
        let idx = state.idx;
        model.neuron_states[idx] = Some(state);
    });

    // update neuron states
    let mut n_firing = 0;
    for state in model.neuron_states.iter_mut() {
        if let Some(state) = state {
            neuron::update_neuron(state, None);
            if state.firing {
                n_firing += 1;
            }
        }
    }

    // update egui
    let egui = &mut model.egui;
    egui.set_elapsed_time(_update.since_start);
    let ctx = egui.begin_frame();

    let mut changed = false;
    egui::Window::new("Settings").show(&ctx, |ui| {
        changed |= ui
            .add(egui::Slider::new(&mut model.mean, -1.0..=1.0).text("Mean"))
            .changed();
        changed |= ui
            .add(egui::Slider::new(&mut model.std, 0.0..=2.0).text("Std"))
            .changed();
    });

    // update network parameters
    if SELF_BALANCE {
        if n_firing > MAX_ACTIVE {
            model.mean -= 0.005;
            changed = true;
        } else if n_firing < MIN_ACTIVE {
            model.mean += 0.005;
            changed = true;
        }
    }

    if changed {
        for neuron in model.net.neurons.iter_mut() {
            neuron
                .modifier_sender
                .send((model.mean, model.std))
                .unwrap();
        }
    }
}

fn view(app: &App, model: &Model, frame: Frame) {
    let draw = app.draw();
    draw.background().color(WHITE);

    let win = app.window_rect();

    // draw neurons and connections
    for (pos, neuron) in model.net.positions.iter().zip(model.net.neurons.iter()) {
        // draw connections
        if DRAW_EVERYTHING {
            let from = model.net.positions[neuron.idx];
            for target_idx in neuron.dendrite_idxs.as_ref().unwrap().iter() {
                let to = model.net.positions[*target_idx];
                draw.line()
                    .start(to_screen_coords(vec2(from.0, from.1), win))
                    .end(to_screen_coords(vec2(to.0, to.1), win))
                    .weight(1.0)
                    .color(BLACK);
            }
        }

        // get neuron color
        let col = if let Some(state) = &model.neuron_states[neuron.idx] {
            if state.firing {
                RED
            } else {
                if !DRAW_EVERYTHING && state.membrane_potential == 0.0 {
                    continue;
                }
                Rgb::new(
                    0,
                    (state.membrane_potential.max(0.0) / ACTION_POTENTIAL_THRESHOLD * 255.0) as u8,
                    (state.membrane_potential.min(0.0) / ACTION_POTENTIAL_THRESHOLD * -255.0) as u8,
                )
            }
        } else if DRAW_EVERYTHING {
            BLACK
        } else {
            continue;
        };

        // draw neuron
        draw.ellipse()
            .xy(to_screen_coords(vec2(pos.0, pos.1), win))
            .radius(NEURON_RADIUS)
            .color(col);
    }

    draw.to_frame(app, &frame).unwrap();
    model.egui.draw_to_frame(&frame).unwrap();
}

fn raw_window_event(_app: &App, model: &mut Model, event: &nannou::winit::event::WindowEvent) {
    // let egui handle things like keyboard and mouse input
    model.egui.handle_raw_event(event);
}

fn main() {
    nannou::app(model).event(event).update(update).run();
}
