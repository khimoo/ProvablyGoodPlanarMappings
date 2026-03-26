//! 表示更新システム: ステータステキスト、パラメータラベル、ボタンビジュアル。

use bevy::prelude::*;

use crate::ui::{BTN_NORMAL, BTN_HOVERED, BTN_PRESSED};
use crate::ui::markers::*;
use crate::state::{AlgoParams, AlgorithmState, AppState, DeformationInfo};

/// システム: ボタンホバー/プレスの視覚フィードバック。
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

/// システム: メインステータステキストを更新。
pub fn update_status_text(
    state: Res<State<AppState>>,
    deform_info: Res<DeformationInfo>,
    algo_state: Res<AlgorithmState>,
    mut query: Query<(&mut Text, &mut TextColor), With<StatusText>>,
) {
    let Ok((mut text, mut color)) = query.single_mut() else { return };
    match state.get() {
        AppState::Setup => {
            let n = algo_state.source_handles.len();
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

/// システム: モード切替ボタンラベルを更新。
pub fn update_toggle_label(
    state: Res<State<AppState>>,
    query: Query<Entity, With<ToggleModeButton>>,
    children_q: Query<&Children>,
    mut text_q: Query<&mut Text>,
) {
    for entity in &query {
        if let Ok(children) = children_q.get(entity) {
            for child in children.iter() {
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

/// システム: K 上界表示テキストを更新。
pub fn update_k_text(
    params: Res<AlgoParams>,
    mut query: Query<&mut Text, With<KBoundText>>,
) {
    if !params.is_changed() { return; }
    for mut text in &mut query {
        **text = format!("{:.1}", params.k_bound);
    }
}

/// システム: lambda 表示テキストを更新。
pub fn update_lambda_text(
    params: Res<AlgoParams>,
    mut query: Query<&mut Text, With<LambdaText>>,
) {
    if !params.is_changed() { return; }
    for mut text in &mut query {
        **text = format!("{:.1e}", params.lambda_reg);
    }
}

/// システム: 正則化モードボタンラベルを更新。
pub fn update_reg_mode_label(
    params: Res<AlgoParams>,
    query: Query<Entity, With<RegModeButton>>,
    children_q: Query<&Children>,
    mut text_q: Query<&mut Text>,
) {
    if !params.is_changed() { return; }
    for entity in &query {
        if let Ok(children) = children_q.get(entity) {
            for child in children.iter() {
                if let Ok(mut txt) = text_q.get_mut(child) {
                    **txt = params.reg_mode.label().to_string();
                }
            }
        }
    }
}

/// システム: 基底タイプボタンラベルを更新。
pub fn update_basis_type_label(
    params: Res<AlgoParams>,
    query: Query<Entity, With<BasisTypeButton>>,
    children_q: Query<&Children>,
    mut text_q: Query<&mut Text>,
) {
    if !params.is_changed() { return; }
    for entity in &query {
        if let Ok(children) = children_q.get(entity) {
            for child in children.iter() {
                if let Ok(mut txt) = text_q.get_mut(child) {
                    **txt = params.basis_type.label().to_string();
                }
            }
        }
    }
}

/// システム: K_max 表示テキストを更新。
pub fn update_k_max_text(
    params: Res<AlgoParams>,
    mut query: Query<&mut Text, With<KMaxText>>,
) {
    if !params.is_changed() { return; }
    for mut text in &mut query {
        **text = format!("{:.1}", params.k_max);
    }
}
