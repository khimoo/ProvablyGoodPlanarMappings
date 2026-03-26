//! パネル構築: 起動時に右側コントロールパネルを生成。

use bevy::prelude::*;

use crate::ui::{PANEL_BG, BTN_NORMAL, BTN_TEXT, LABEL_COLOR, VALUE_COLOR};
use crate::ui::markers::*;

/// 起動システム: コントロールパネル全体を生成。
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
                width: Val::Px(super::PANEL_WIDTH),
                flex_direction: FlexDirection::Column,
                padding: UiRect::all(Val::Px(10.0)),
                row_gap: Val::Px(6.0),
                ..default()
            },
            BackgroundColor(PANEL_BG),
        ))
        .with_children(|panel| {
            // ステータステキスト
            panel.spawn((
                Text::new("Mode: SETUP\nHandles: 0\nClick to add handles"),
                TextFont { font: font.clone(), font_size: 14.0, ..default() },
                TextColor(VALUE_COLOR),
                StatusText,
            ));

            separator(panel);

            // モード切替
            wide_button(panel, "\u{f04b}  Start Deforming", ToggleModeButton, &font);

            // リセット
            wide_button(panel, "\u{f0e2}  Reset", ResetButton, &font);

            separator(panel);

            // K 上界
            label(panel, "Distortion bound K", &font);
            param_row(panel, "3.0", KBoundText, "\u{f068}", KMinusButton, "\u{f067}", KPlusButton, &font);

            // Lambda（正則化係数）
            label(panel, "Regularization \u{03bb}", &font);
            param_row(panel, "1.0e-2", LambdaText, "/10", LambdaDownButton, "x10", LambdaUpButton, &font);

            // 正則化タイプ
            label(panel, "Regularization type", &font);
            wide_button(panel, "ARAP", RegModeButton, &font);

            separator(panel);

            // 基底関数タイプ
            label(panel, "Basis function", &font);
            wide_button(panel, "Gaussian", BasisTypeButton, &font);

            separator(panel);

            // Strategy 2
            label(panel, "Strategy 2 (Eq. 14)", &font);
            label(panel, "K_max target", &font);
            param_row(panel, "6.0", KMaxText, "\u{f068}", KMaxMinusButton, "\u{f067}", KMaxPlusButton, &font);
            wide_button(panel, "\u{f0e7}  Refine (Strategy 2)", Strategy2Button, &font);

            separator(panel);

            // 画像パス
            label(panel, "Image path", &font);
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

// ビルダーヘルパー

fn separator(parent: &mut ChildSpawnerCommands) {
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

fn label(parent: &mut ChildSpawnerCommands, text: &str, font: &Handle<Font>) {
    parent.spawn((
        Text::new(text.to_string()),
        TextFont { font: font.clone(), font_size: 12.0, ..default() },
        TextColor(LABEL_COLOR),
    ));
}

fn wide_button<M: Component>(parent: &mut ChildSpawnerCommands, text: &str, marker: M, font: &Handle<Font>) {
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
    parent: &mut ChildSpawnerCommands,
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
            // 左ボタン
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

            // 値テキスト
            row.spawn((
                Text::new(initial.to_string()),
                TextFont { font: font.clone(), font_size: 14.0, ..default() },
                TextColor(VALUE_COLOR),
                text_marker,
            ));

            // 右ボタン
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
