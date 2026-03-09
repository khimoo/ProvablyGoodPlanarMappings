//! GUI panel: buttons, parameter controls, and handle gizmos.
//!
//! All user interactions (except handle placement/dragging on the canvas)
//! are performed through on-screen UI elements.  No keyboard shortcuts.

use bevy::prelude::*;

use crate::state::{
    AlgoParams, AppState, DeformationInfo, DeformationState, ImageInfo, ImagePathConfig,
};

// ── Marker components for UI widgets ────────────────────────────────

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

// ── Colours ─────────────────────────────────────────────────────────

const PANEL_BG: Color = Color::srgba(0.08, 0.08, 0.12, 0.92);
const BTN_NORMAL: Color = Color::srgb(0.25, 0.25, 0.35);
const BTN_HOVERED: Color = Color::srgb(0.35, 0.35, 0.50);
const BTN_PRESSED: Color = Color::srgb(0.50, 0.40, 0.20);
const BTN_TEXT: Color = Color::srgb(0.95, 0.95, 0.95);
const LABEL_COLOR: Color = Color::srgb(0.70, 0.70, 0.70);
const VALUE_COLOR: Color = Color::srgb(0.95, 0.85, 0.40);

// ── Spawn the whole panel (called from main.rs Startup) ─────────────

pub fn spawn_control_panel(mut commands: Commands, asset_server: Res<AssetServer>) {
    let font: Handle<Font> = asset_server.load("fonts/FiraCodeNerdFontMono-Regular.ttf");
    commands.insert_resource(UiFont(font.clone()));

    commands
        .spawn((
            Node {
                position_type: PositionType::Absolute,
                right: Val::Px(0.0),
                top: Val::Px(0.0),
                bottom: Val::Px(0.0),
                width: Val::Px(220.0),
                flex_direction: FlexDirection::Column,
                padding: UiRect::all(Val::Px(10.0)),
                row_gap: Val::Px(6.0),
                ..default()
            },
            BackgroundColor(PANEL_BG),
        ))
        .with_children(|panel| {
            // ── Status text ─────────────────────────────────
            panel.spawn((
                Text::new("Mode: SETUP\nHandles: 0\nClick to add handles"),
                TextFont { font: font.clone(), font_size: 14.0, ..default() },
                TextColor(VALUE_COLOR),
                StatusText,
            ));

            separator(panel);

            // ── Toggle mode ─────────────────────────────────
            wide_button(panel, "\u{f04b}  Start Deforming", ToggleModeButton, &font);

            // ── Reset ───────────────────────────────────────
            wide_button(panel, "\u{f0e2}  Reset", ResetButton, &font);

            separator(panel);

            // ── K bound ─────────────────────────────────────
            label(panel, "Distortion bound K", &font);
            param_row(panel, "3.0", KBoundText, "\u{f068}", KMinusButton, "\u{f067}", KPlusButton, &font);

            // ── Lambda ──────────────────────────────────────
            label(panel, "Regularization \u{03bb}", &font);
            param_row(panel, "1.0e-2", LambdaText, "/10", LambdaDownButton, "x10", LambdaUpButton, &font);

            // ── Regularization type ─────────────────────────
            label(panel, "Regularization type", &font);
            wide_button(panel, "ARAP", RegModeButton, &font);

            separator(panel);

            // ── Basis type ──────────────────────────────────
            label(panel, "Basis function", &font);
            wide_button(panel, "Gaussian", BasisTypeButton, &font);

            separator(panel);

            // ── Strategy 2 ─────────────────────────────────
            label(panel, "Strategy 2 (Eq. 14)", &font);
            label(panel, "K_max target", &font);
            param_row(panel, "6.0", KMaxText, "\u{f068}", KMaxMinusButton, "\u{f067}", KMaxPlusButton, &font);
            wide_button(panel, "\u{f0e7}  Refine (Strategy 2)", Strategy2Button, &font);

            separator(panel);

            // ── Image path ──────────────────────────────────
            label(panel, "Image path", &font);
            // Editable path display
            panel.spawn((
                Node {
                    width: Val::Percent(100.0),
                    padding: UiRect::all(Val::Px(4.0)),
                    ..default()
                },
                BackgroundColor(Color::srgb(0.12, 0.12, 0.18)),
            )).with_children(|row| {
                row.spawn((
                    Text::new("texture.png"),
                    TextFont { font: font.clone(), font_size: 11.0, ..default() },
                    TextColor(VALUE_COLOR),
                    ImagePathText,
                ));
            });
            wide_button(panel, "\u{f07c}  Load Image", ImageLoadButton, &font);
        });
}

// ── Builder helpers ─────────────────────────────────────────────────

fn separator(parent: &mut ChildBuilder) {
    parent.spawn((
        Node {
            width: Val::Percent(100.0),
            height: Val::Px(1.0),
            margin: UiRect::vertical(Val::Px(4.0)),
            ..default()
        },
        BackgroundColor(Color::srgba(1.0, 1.0, 1.0, 0.15)),
    ));
}

fn label(parent: &mut ChildBuilder, text: &str, font: &Handle<Font>) {
    parent.spawn((
        Text::new(text.to_string()),
        TextFont { font: font.clone(), font_size: 12.0, ..default() },
        TextColor(LABEL_COLOR),
    ));
}

fn wide_button<M: Component>(parent: &mut ChildBuilder, text: &str, marker: M, font: &Handle<Font>) {
    parent
        .spawn((
            Button,
            Node {
                width: Val::Percent(100.0),
                height: Val::Px(32.0),
                justify_content: JustifyContent::Center,
                align_items: AlignItems::Center,
                ..default()
            },
            BackgroundColor(BTN_NORMAL),
            marker,
        ))
        .with_children(|btn| {
            btn.spawn((
                Text::new(text.to_string()),
                TextFont { font: font.clone(), font_size: 14.0, ..default() },
                TextColor(BTN_TEXT),
            ));
        });
}

fn param_row<TM: Component, LB: Component, RB: Component>(
    parent: &mut ChildBuilder,
    initial: &str,
    text_marker: TM,
    left_label: &str,
    left_marker: LB,
    right_label: &str,
    right_marker: RB,
    font: &Handle<Font>,
) {
    parent
        .spawn(Node {
            width: Val::Percent(100.0),
            flex_direction: FlexDirection::Row,
            justify_content: JustifyContent::SpaceBetween,
            align_items: AlignItems::Center,
            column_gap: Val::Px(4.0),
            ..default()
        })
        .with_children(|row| {
            // Left button
            row.spawn((
                Button,
                Node {
                    width: Val::Px(44.0),
                    height: Val::Px(28.0),
                    justify_content: JustifyContent::Center,
                    align_items: AlignItems::Center,
                    ..default()
                },
                BackgroundColor(BTN_NORMAL),
                left_marker,
            ))
            .with_children(|btn| {
                btn.spawn((
                    Text::new(left_label.to_string()),
                    TextFont { font: font.clone(), font_size: 13.0, ..default() },
                    TextColor(BTN_TEXT),
                ));
            });

            // Value text
            row.spawn((
                Text::new(initial.to_string()),
                TextFont { font: font.clone(), font_size: 14.0, ..default() },
                TextColor(VALUE_COLOR),
                text_marker,
            ));

            // Right button
            row.spawn((
                Button,
                Node {
                    width: Val::Px(44.0),
                    height: Val::Px(28.0),
                    justify_content: JustifyContent::Center,
                    align_items: AlignItems::Center,
                    ..default()
                },
                BackgroundColor(BTN_NORMAL),
                right_marker,
            ))
            .with_children(|btn| {
                btn.spawn((
                    Text::new(right_label.to_string()),
                    TextFont { font: font.clone(), font_size: 13.0, ..default() },
                    TextColor(BTN_TEXT),
                ));
            });
        });
}

// ── Systems: button hover/press visual feedback ─────────────────────

pub fn button_visuals(
    mut query: Query<(&Interaction, &mut BackgroundColor), (Changed<Interaction>, With<Button>)>,
) {
    for (interaction, mut bg) in &mut query {
        *bg = match interaction {
            Interaction::Pressed => BackgroundColor(BTN_PRESSED),
            Interaction::Hovered => BackgroundColor(BTN_HOVERED),
            Interaction::None => BackgroundColor(BTN_NORMAL),
        };
    }
}

// ── Systems: button actions ─────────────────────────────────────────

pub fn on_toggle_mode(
    query: Query<&Interaction, (Changed<Interaction>, With<ToggleModeButton>)>,
    state: Res<State<AppState>>,
    mut next_state: ResMut<NextState<AppState>>,
    mut commands: Commands,
    mut deform_state: ResMut<DeformationState>,
    mut deform_info: ResMut<DeformationInfo>,
    algo_params: Res<AlgoParams>,
    image_info: Option<Res<ImageInfo>>,
) {
    for interaction in &query {
        if *interaction != Interaction::Pressed { continue; }
        match state.get() {
            AppState::Setup => {
                let Some(ref image_info) = image_info else { return };
                if deform_state.source_handles.is_empty() {
                    info!("No handles added yet!");
                    return;
                }
                let is_shape_aware = deform_state.finalize(
                    image_info.width as f64,
                    image_info.height as f64,
                    &algo_params,
                    &image_info.contour,
                );
                commands.insert_resource(
                    crate::deform::UseShapeAwareBasis(is_shape_aware),
                );
                deform_info.k_bound = algo_params.k_bound;
                deform_info.lambda_reg = algo_params.lambda_reg;
                deform_info.reg_mode_label = algo_params.reg_mode.label();
                next_state.set(AppState::Deforming);
            }
            AppState::Deforming => {
                deform_state.dragging = false;
                deform_state.dragging_index = None;
                next_state.set(AppState::Setup);
            }
        }
    }
}

pub fn on_reset(
    query: Query<&Interaction, (Changed<Interaction>, With<ResetButton>)>,
    mut deform_state: ResMut<DeformationState>,
    mut deform_info: ResMut<DeformationInfo>,
    mut next_state: ResMut<NextState<AppState>>,
    image_info: Option<Res<ImageInfo>>,
    mut materials: ResMut<Assets<crate::rendering::DeformMaterial>>,
    mat_query: Query<
        &bevy::sprite::MeshMaterial2d<crate::rendering::DeformMaterial>,
        With<crate::state::DeformedImage>,
    >,
) {
    for interaction in &query {
        if *interaction != Interaction::Pressed { continue; }
        deform_state.source_handles.clear();
        deform_state.target_handles.clear();
        deform_state.algorithm = None;
        deform_state.dragging = false;
        deform_state.dragging_index = None;
        deform_state.needs_solve = false;
        *deform_info = DeformationInfo::default();
        next_state.set(AppState::Setup);

        // Reset shader uniform to identity mapping
        if let (Some(ref image_info), Ok(mat_handle)) = (image_info.as_ref(), mat_query.get_single()) {
            if let Some(material) = materials.get_mut(&mat_handle.0) {
                let mut params = crate::rendering::DeformUniform::default();
                params.image_width = image_info.width;
                params.image_height = image_info.height;
                params.n_rbf = 0;
                // Identity: const=(0,0), x=(1,0), y=(0,1)
                params.coeffs[0] = crate::rendering::RBFCoeff { x: 0.0, y: 0.0, _padding: Vec2::ZERO };
                params.coeffs[1] = crate::rendering::RBFCoeff { x: 1.0, y: 0.0, _padding: Vec2::ZERO };
                params.coeffs[2] = crate::rendering::RBFCoeff { x: 0.0, y: 1.0, _padding: Vec2::ZERO };
                material.params = params;
            }
        }
    }
}

pub fn on_k_bound(
    q_minus: Query<&Interaction, (Changed<Interaction>, With<KMinusButton>)>,
    q_plus: Query<&Interaction, (Changed<Interaction>, With<KPlusButton>)>,
    mut params: ResMut<AlgoParams>,
    mut info: ResMut<DeformationInfo>,
    mut deform_state: ResMut<DeformationState>,
) {
    let mut changed = false;
    for interaction in &q_minus {
        if *interaction == Interaction::Pressed {
            params.k_bound = (params.k_bound - 0.5).max(1.1);
            changed = true;
        }
    }
    for interaction in &q_plus {
        if *interaction == Interaction::Pressed {
            params.k_bound += 0.5;
            changed = true;
        }
    }
    if changed {
        enforce_k_max_invariant(&mut params);
        info.k_bound = params.k_bound;
        push_params(&params, &mut deform_state);
    }
}

pub fn on_lambda(
    q_down: Query<&Interaction, (Changed<Interaction>, With<LambdaDownButton>)>,
    q_up: Query<&Interaction, (Changed<Interaction>, With<LambdaUpButton>)>,
    mut params: ResMut<AlgoParams>,
    mut info: ResMut<DeformationInfo>,
    mut deform_state: ResMut<DeformationState>,
) {
    let mut changed = false;
    for interaction in &q_down {
        if *interaction == Interaction::Pressed {
            params.lambda_reg /= 10.0;
            changed = true;
        }
    }
    for interaction in &q_up {
        if *interaction == Interaction::Pressed {
            params.lambda_reg *= 10.0;
            changed = true;
        }
    }
    if changed {
        info.lambda_reg = params.lambda_reg;
        push_params(&params, &mut deform_state);
    }
}

pub fn on_reg_mode(
    query: Query<&Interaction, (Changed<Interaction>, With<RegModeButton>)>,
    mut params: ResMut<AlgoParams>,
    mut info: ResMut<DeformationInfo>,
    mut deform_state: ResMut<DeformationState>,
) {
    for interaction in &query {
        if *interaction != Interaction::Pressed { continue; }
        params.reg_mode = params.reg_mode.next();
        info.reg_mode_label = params.reg_mode.label();
        push_params(&params, &mut deform_state);
    }
}

/// System: cycle basis function type.
/// Only effective in Setup mode (basis is baked into the Algorithm on finalize).
pub fn on_basis_type(
    query: Query<&Interaction, (Changed<Interaction>, With<BasisTypeButton>)>,
    mut params: ResMut<AlgoParams>,
    state: Res<State<AppState>>,
) {
    for interaction in &query {
        if *interaction != Interaction::Pressed { continue; }
        if *state.get() != AppState::Setup {
            info!("Basis type can only be changed in Setup mode");
            continue;
        }
        params.basis_type = params.basis_type.next();
        info!("Basis type: {}", params.basis_type.label());
    }
}

/// Enforce Strategy 2 invariant: K_max > K (Eq. 14 precondition).
/// Called after any change to k_bound or k_max.
fn enforce_k_max_invariant(params: &mut AlgoParams) {
    let min_k_max = params.k_bound + 0.1;
    if params.k_max < min_k_max {
        params.k_max = min_k_max;
    }
}

fn push_params(params: &AlgoParams, deform_state: &mut DeformationState) {
    if let Some(ref mut algo) = deform_state.algorithm {
        let core_params = pgpm_core::AlgorithmParams {
            distortion_type: pgpm_core::DistortionType::Isometric,
            k_bound: params.k_bound,
            lambda_reg: params.reg_mode.effective_lambda(params.lambda_reg),
            regularization: params.reg_mode.to_core(params.lambda_arap, params.lambda_bh),
        };
        algo.update_params(core_params);
        deform_state.needs_solve = true;
    }
}

/// System: handle the "Load Image" button click.
/// Opens a native file dialog to pick a PNG image, then triggers reload.
pub fn on_image_path(
    query: Query<&Interaction, (Changed<Interaction>, With<ImageLoadButton>)>,
    mut path_config: ResMut<ImagePathConfig>,
    mut path_text: Query<&mut Text, With<ImagePathText>>,
    state: Res<State<AppState>>,
) {
    for interaction in &query {
        if *interaction != Interaction::Pressed { continue; }

        // Only allow image loading in Setup mode
        if *state.get() != AppState::Setup {
            info!("Switch to Setup mode before loading a new image");
            continue;
        }

        // Open native file dialog
        let dialog = rfd::FileDialog::new()
            .add_filter("PNG Images", &["png"])
            .add_filter("All Images", &["png", "jpg", "jpeg", "bmp", "tga"])
            .set_title("Select Image");

        if let Some(path) = dialog.pick_file() {
            let abs_path = path.to_string_lossy().into_owned();
            info!("Selected image: {}", abs_path);

            // Update the display text (show just filename)
            let display_name = path.file_name()
                .map(|n| n.to_string_lossy().into_owned())
                .unwrap_or_else(|| abs_path.clone());
            for mut text in &mut path_text {
                **text = display_name.clone();
            }

            // Trigger reload
            path_config.abs_path = abs_path;
            path_config.needs_reload = true;
        }
    }
}

/// System: adjust K_max parameter via +/- buttons.
pub fn on_k_max(
    q_minus: Query<&Interaction, (Changed<Interaction>, With<KMaxMinusButton>)>,
    q_plus: Query<&Interaction, (Changed<Interaction>, With<KMaxPlusButton>)>,
    mut params: ResMut<AlgoParams>,
) {
    let mut changed = false;
    for interaction in &q_minus {
        if *interaction == Interaction::Pressed {
            params.k_max -= 0.5;
            changed = true;
        }
    }
    for interaction in &q_plus {
        if *interaction == Interaction::Pressed {
            params.k_max += 0.5;
            changed = true;
        }
    }
    if changed {
        enforce_k_max_invariant(&mut params);
        info!("K_max = {:.1}", params.k_max);
    }
}

/// System: update K_max display text.
pub fn update_k_max_text(
    params: Res<AlgoParams>,
    mut query: Query<&mut Text, With<KMaxText>>,
) {
    if !params.is_changed() { return; }
    for mut text in &mut query {
        **text = format!("{:.1}", params.k_max);
    }
}

/// System: handle the "Refine (Strategy 2)" button click.
pub fn on_strategy2(
    query: Query<&Interaction, (Changed<Interaction>, With<Strategy2Button>)>,
    state: Res<State<AppState>>,
    mut deform_state: ResMut<DeformationState>,
    mut deform_info: ResMut<DeformationInfo>,
    params: Res<AlgoParams>,
) {
    for interaction in &query {
        if *interaction != Interaction::Pressed { continue; }

        if *state.get() != AppState::Deforming {
            info!("Strategy 2 is only available in Deforming mode");
            continue;
        }

        if deform_state.algorithm.is_none() {
            info!("No algorithm instance");
            continue;
        };

        let targets = deform_state.target_handles.clone();
        let k_max = params.k_max;

        info!(
            "Strategy 2: K={:.2}, K_max={:.2}, running refinement...",
            params.k_bound, k_max
        );

        let algo = deform_state.algorithm.as_mut().unwrap();
        match algo.refine_strategy2(k_max, &targets) {
            Ok(result) => {
                let msg = format!(
                    "h_req={:.4}, res={}, steps={}\nK_max={:.3}, |||c|||={:.3}",
                    result.required_h,
                    result.required_resolution,
                    result.refinement_steps,
                    result.k_max_achieved,
                    result.c_norm,
                );
                info!("Strategy 2 complete: {}", msg);
                deform_info.strategy2_status = Some(msg);
                deform_state.needs_solve = false;
            }
            Err(e) => {
                let msg = format!("Error: {}", e);
                warn!("Strategy 2 failed: {}", msg);
                deform_info.strategy2_status = Some(msg);
            }
        }
    }
}

// ── Systems: update dynamic text ────────────────────────────────────

pub fn update_status_text(
    state: Res<State<AppState>>,
    deform_info: Res<DeformationInfo>,
    deform_state: Res<DeformationState>,
    mut query: Query<(&mut Text, &mut TextColor), With<StatusText>>,
) {
    let Ok((mut text, mut color)) = query.get_single_mut() else { return };
    match state.get() {
        AppState::Setup => {
            let n = deform_state.source_handles.len();
            **text = format!("Mode: SETUP\nHandles: {n}\nClick to add handles");
            color.0 = Color::srgb(0.4, 1.0, 0.4);
        }
        AppState::Deforming => {
            let mut s = format!(
                "Mode: DEFORM\nmax D = {:.2}\nActive: {} | Stable: {}\nSteps: {}",
                deform_info.max_distortion,
                deform_info.active_set_size,
                deform_info.stable_set_size,
                deform_info.step_count,
            );
            if let Some(ref status) = deform_info.strategy2_status {
                s.push_str(&format!("\n---\nStrategy 2:\n{}", status));
            }
            **text = s;
            color.0 = Color::srgb(1.0, 0.65, 0.4);
        }
    }
}

pub fn update_toggle_label(
    state: Res<State<AppState>>,
    query: Query<Entity, With<ToggleModeButton>>,
    children_q: Query<&Children>,
    mut text_q: Query<&mut Text>,
) {
    for entity in &query {
        if let Ok(children) = children_q.get(entity) {
            for &child in children.iter() {
                if let Ok(mut txt) = text_q.get_mut(child) {
                    **txt = match state.get() {
                        AppState::Setup => "\u{f04b}  Start Deforming".to_string(),
                        AppState::Deforming => "\u{f04d}  Back to Setup".to_string(),
                    };
                }
            }
        }
    }
}

pub fn update_k_text(
    params: Res<AlgoParams>,
    mut query: Query<&mut Text, With<KBoundText>>,
) {
    if !params.is_changed() { return; }
    for mut text in &mut query {
        **text = format!("{:.1}", params.k_bound);
    }
}

pub fn update_lambda_text(
    params: Res<AlgoParams>,
    mut query: Query<&mut Text, With<LambdaText>>,
) {
    if !params.is_changed() { return; }
    for mut text in &mut query {
        **text = format!("{:.1e}", params.lambda_reg);
    }
}

pub fn update_reg_mode_label(
    params: Res<AlgoParams>,
    query: Query<Entity, With<RegModeButton>>,
    children_q: Query<&Children>,
    mut text_q: Query<&mut Text>,
) {
    if !params.is_changed() { return; }
    for entity in &query {
        if let Ok(children) = children_q.get(entity) {
            for &child in children.iter() {
                if let Ok(mut txt) = text_q.get_mut(child) {
                    **txt = params.reg_mode.label().to_string();
                }
            }
        }
    }
}

pub fn update_basis_type_label(
    params: Res<AlgoParams>,
    query: Query<Entity, With<BasisTypeButton>>,
    children_q: Query<&Children>,
    mut text_q: Query<&mut Text>,
) {
    if !params.is_changed() { return; }
    for entity in &query {
        if let Ok(children) = children_q.get(entity) {
            for &child in children.iter() {
                if let Ok(mut txt) = text_q.get_mut(child) {
                    **txt = params.basis_type.label().to_string();
                }
            }
        }
    }
}

// ── Gizmos ──────────────────────────────────────────────────────────

pub fn draw_handles(
    deform_state: Res<DeformationState>,
    state: Res<State<AppState>>,
    image_info: Option<Res<ImageInfo>>,
    mut gizmos: Gizmos,
) {
    let Some(image_info) = image_info else { return };
    let img_w = image_info.width;
    let img_h = image_info.height;

    let (source_color, target_color) = match state.get() {
        AppState::Setup => (Color::srgb(1.0, 1.0, 0.0), Color::srgb(1.0, 1.0, 0.0)),
        AppState::Deforming => (
            Color::srgba(0.3, 0.3, 1.0, 0.4),
            Color::srgb(0.0, 1.0, 1.0),
        ),
    };

    for src in deform_state.source_handles.iter() {
        let wx = src.x as f32 - img_w / 2.0;
        let wy = img_h / 2.0 - src.y as f32;
        gizmos.circle_2d(Vec2::new(wx, wy), 6.0, source_color);
    }

    if *state.get() == AppState::Deforming {
        for (i, tgt) in deform_state.target_handles.iter().enumerate() {
            let wx = tgt.x as f32 - img_w / 2.0;
            let wy = img_h / 2.0 - tgt.y as f32;
            let pos = Vec2::new(wx, wy);
            let color = if deform_state.dragging_index == Some(i) {
                Color::srgb(1.0, 0.0, 0.0)
            } else {
                target_color
            };
            gizmos.circle_2d(pos, 8.0, color);

            let src = &deform_state.source_handles[i];
            let src_wx = src.x as f32 - img_w / 2.0;
            let src_wy = img_h / 2.0 - src.y as f32;
            if (src_wx - wx).abs() > 1.0 || (src_wy - wy).abs() > 1.0 {
                gizmos.line_2d(
                    Vec2::new(src_wx, src_wy),
                    pos,
                    Color::srgba(1.0, 1.0, 1.0, 0.3),
                );
            }
        }
    }
}
