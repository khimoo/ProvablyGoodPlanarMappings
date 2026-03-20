//! UI interaction state: drag operations on handles.

use bevy::prelude::*;

/// Tracks the current drag operation (separated from algorithm state
/// to avoid shared mutable access between input and solver systems).
#[derive(Resource, Default)]
pub struct DragState {
    /// Whether a drag operation is currently in progress.
    pub active: bool,
    /// Index of the currently dragged handle, if any.
    pub handle_index: Option<usize>,
}

impl DragState {
    pub fn start(&mut self, index: usize) {
        self.active = true;
        self.handle_index = Some(index);
    }

    pub fn end(&mut self) {
        self.active = false;
        self.handle_index = None;
    }
}
