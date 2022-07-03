use std::cmp::Ordering;
use std::collections::HashMap;
use std::sync::{mpsc, MutexGuard};
use std::time::{Duration, Instant};

#[derive(Copy, Clone)]
pub struct Vector2 {
    x: f32,
    y: f32,
}

impl Vector2 {
    pub fn new(x: f32, y: f32) -> Self {
        Vector2 { x: x, y: y }
    }

    pub fn magnitude(self) -> f32 {
        (self.x * self.x + self.y * self.y).sqrt()
    }
}

impl std::ops::Sub<Vector2> for Vector2 {
    type Output = Vector2;

    fn sub(self, other: Vector2) -> Vector2 {
        Vector2::new(self.x - other.x, self.y - other.y)
    }
}

pub struct Neuron {
    pub index: usize,
    pub position: Vector2,
    pub potential: f32,
    pub synapses_out: HashMap<usize, f32>,
    pub last_update: Option<Instant>,
}

impl Neuron {
    fn update(&mut self, change: Option<f32>) -> f32 {
        if let Some(last_update) = self.last_update {
            let elapsed = last_update.elapsed().as_millis();
            if self.potential > 0. {
                self.potential = (self.potential - crate::DECAY_RATE * (elapsed as f32)).max(0.);
            } else if self.potential < 0. {
                self.potential = (self.potential + crate::RECOVERY_RATE * (elapsed as f32)).min(0.);
            }
        }

        self.potential += change.unwrap_or_default();
        self.last_update = Some(Instant::now());
        return self.potential;
    }

    pub fn receive_ap(&mut self, ap: &ActionPotential) -> bool {
        self.update(Some(ap.strength));

        if self.potential >= crate::AP_THRESHOLD {
            self.last_update = Some(Instant::now());
            self.potential = crate::UNDERSHOOT_POTENTIAL;
            return true;
        }
        return false;
    }

    pub fn fire(
        index: usize,
        neurons: MutexGuard<Vec<Neuron>>,
        ap_tx: &mpsc::Sender<ActionPotential>,
    ) {
        for (target, strength) in &neurons[index].synapses_out {
            ap_tx
                .send(ActionPotential::new(
                    &neurons[index],
                    &neurons[*target],
                    *strength,
                ))
                .unwrap();
        }
    }
}

pub struct ActionPotential {
    duration: Duration,
    strength: f32,
    pub from: usize,
    pub to: usize,
    start_time: Instant,
}

impl ActionPotential {
    pub fn new(from: &Neuron, to: &Neuron, strength: f32) -> Self {
        let dist = (from.position - to.position).magnitude();
        let duration = Duration::from_millis((dist / crate::AP_VELOCITY) as u64);
        Self {
            from: from.index,
            to: to.index,
            strength: strength,
            duration: duration,
            start_time: Instant::now(),
        }
    }

    pub fn time_left(&self) -> Option<Duration> {
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
