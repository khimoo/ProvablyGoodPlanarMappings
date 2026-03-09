//! Marker components for UI widgets.

use bevy::prelude::*;

#[derive(Component)]
pub struct StatusText;

#[derive(Component)]
pub struct ToggleModeButton;

#[derive(Component)]
pub struct ResetButton;

#[derive(Component)]
pub struct KBoundText;

#[derive(Component)]
pub struct KMinusButton;

#[derive(Component)]
pub struct KPlusButton;

#[derive(Component)]
pub struct LambdaText;

#[derive(Component)]
pub struct LambdaDownButton;

#[derive(Component)]
pub struct LambdaUpButton;

#[derive(Component)]
pub struct RegModeButton;

#[derive(Component)]
pub struct BasisTypeButton;

#[derive(Component)]
pub struct ImagePathText;

#[derive(Component)]
pub struct ImageLoadButton;

#[derive(Component)]
pub struct Strategy2Button;

#[derive(Component)]
pub struct KMaxText;

#[derive(Component)]
pub struct KMaxMinusButton;

#[derive(Component)]
pub struct KMaxPlusButton;

/// Shared font handle loaded at startup.
#[derive(Resource)]
pub struct UiFont(pub Handle<Font>);
