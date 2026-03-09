//! Panel construction: spawns the right-side control panel at startup.

use bevy::prelude::*;

use crate::ui::{PANEL_BG, BTN_NORMAL, BTN_TEXT, LABEL_COLOR, VALUE_COLOR};
use crate::ui::markers::*;

/// Startup system: spawn the entire control panel.
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
            // Status text
            panel.spawn((
                Text::new("Mode: SETUP\nHandles: 0\nClick to add handles"),
                TextFont { font: font.clone(), font_size: 14.0, ..default() },
                TextColor(VALUE_COLOR),
                StatusText,
            ));

            separator(panel);

            // Toggle mode
            wide_button(panel, "\u{f04b}  Start Deforming", ToggleModeButton, &font);

            // Reset
            wide_button(panel, "\u{f0e2}  Reset", ResetButton, &font);

            separator(panel);

            // K bound
            label(panel, "Distortion bound K", &font);
            param_row(panel, "3.0", KBoundText, "\u{f068}", KMinusButton, "\u{f067}", KPlusButton, &font);

            // Lambda
            label(panel, "Regularization \u{03bb}", &font);
            param_row(panel, "1.0e-2", LambdaText, "/10", LambdaDownButton, "x10", LambdaUpButton, &font);

            // Regularization type
            label(panel, "Regularization type", &font);
            wide_button(panel, "ARAP", RegModeButton, &font);

            separator(panel);

            // Basis type
            label(panel, "Basis function", &font);
            wide_button(panel, "Gaussian", BasisTypeButton, &font);

            separator(panel);

            // Strategy 2
            label(panel, "Strategy 2 (Eq. 14)", &font);
            label(panel, "K_max target", &font);
            param_row(panel, "6.0", KMaxText, "\u{f068}", KMaxMinusButton, "\u{f067}", KMaxPlusButton, &font);
            wide_button(panel, "\u{f0e7}  Refine (Strategy 2)", Strategy2Button, &font);

            separator(panel);

            // Image path
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

// Builder helpers

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
