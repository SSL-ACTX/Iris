use crate::pid::Pid;
use dashmap::DashMap;

/// A 3D coordinate point in logical space.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Point {
    pub x: f32,
    pub y: f32,
    pub z: f32,
}

impl Point {
    pub fn new(x: f32, y: f32, z: f32) -> Self {
        Self { x, y, z }
    }

    pub fn distance_to(&self, other: &Point) -> f32 {
        ((self.x - other.x).powi(2) + (self.y - other.y).powi(2) + (self.z - other.z).powi(2))
            .sqrt()
    }
}

/// Manages spatial indexing of actors.
pub struct SpatialManager {
    pub(crate) locations: DashMap<Pid, Point>,
}

impl SpatialManager {
    pub fn new() -> Self {
        Self {
            locations: DashMap::new(),
        }
    }

    pub fn set_location(&self, pid: Pid, pos: Point) {
        self.locations.insert(pid, pos);
    }

    pub fn remove_location(&self, pid: Pid) {
        self.locations.remove(&pid);
    }

    pub fn get_location(&self, pid: Pid) -> Option<Point> {
        self.locations.get(&pid).map(|r| *r.value())
    }

    pub fn find_nearby(&self, center: Point, radius: f32) -> Vec<Pid> {
        self.locations
            .iter()
            .filter(|r| r.value().distance_to(&center) <= radius)
            .map(|r| *r.key())
            .collect()
    }
}

impl crate::Runtime {
    pub fn set_location(&self, pid: Pid, x: f32, y: f32, z: f32) {
        self.spatial.set_location(pid, Point::new(x, y, z));
    }

    pub fn get_location(&self, pid: Pid) -> Option<(f32, f32, f32)> {
        self.spatial.get_location(pid).map(|p| (p.x, p.y, p.z))
    }

    pub fn send_to_radius(
        &self,
        x: f32,
        y: f32,
        z: f32,
        radius: f32,
        msg: crate::mailbox::Message,
    ) {
        let center = Point::new(x, y, z);
        let nearby = self.spatial.find_nearby(center, radius);
        for pid in nearby {
            let _ = self.send(pid, msg.clone());
        }
    }
}
