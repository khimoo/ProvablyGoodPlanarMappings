use bevy::prelude::*;
use crossbeam_channel::{Receiver, Sender};
use pyo3::prelude::*;
use pyo3::types::{PyList, PyModule};
use std::env;

use super::commands::{PyCommand, PyResult};

#[derive(Resource)]
pub struct PythonChannels {
    pub tx_command: tokio::sync::mpsc::Sender<PyCommand>,
    pub rx_result: Receiver<PyResult>,
}

pub async fn python_thread_loop(
    mut rx_cmd: tokio::sync::mpsc::Receiver<PyCommand>,
    tx_res: Sender<PyResult>,
) {
    pyo3::prepare_freethreaded_python();

    let bridge: PyObject = Python::with_gil(|py| {
        let sys = py.import("sys").unwrap();
        let current_dir = env::current_dir().unwrap();
        let script_dir = current_dir.join("scripts");
        if let Ok(path) = sys.getattr("path") {
            if let Ok(path_list) = path.downcast::<PyList>() {
                let _ = path_list.insert(0, script_dir);
            }
        }
        let module = PyModule::import(py, "bevy_bridge").expect("Failed to import bevy_bridge");
        let bridge_class = module.getattr("BevyBridge").expect("No BevyBridge class");
        let bridge_instance = bridge_class.call0().expect("Failed to init BevyBridge");
        bridge_instance.into()
    });

    println!("Rust: Python Thread Ready");

    fn extract_from_dict<T>(dict: &pyo3::Bound<'_, pyo3::types::PyDict>, key: &str) -> Option<T>
    where
        T: for<'a> pyo3::FromPyObject<'a>,
    {
        dict.get_item(key).ok().flatten().and_then(|v| v.extract::<T>().ok())
    }

    loop {
        let Some(cmd) = rx_cmd.recv().await else {
            break;
        };

        Python::with_gil(|py| {
            let bridge_bound = bridge.bind(py);
            match cmd {
                PyCommand::InitializeDomain { width, height, epsilon } => {
                    println!("Rust->Py: InitializeDomain {}x{}, eps={}", width, height, epsilon);
                    if let Err(e) = bridge_bound.call_method1("initialize_domain", (width, height, epsilon)) {
                        eprintln!("Py Error (InitializeDomain): {}", e);
                    } else {
                        let _ = tx_res.send(PyResult::DomainInitialized);
                    }
                }
                PyCommand::SetContour { contour } => {
                    println!("Rust->Py: SetContour with {} points", contour.len());
                    if let Err(e) = bridge_bound.call_method1("set_contour", (contour,)) {
                        eprintln!("Py Error (SetContour): {}", e);
                    }
                }
                PyCommand::AddControlPoint { index, x, y } => {
                    if let Err(e) = bridge_bound.call_method1("add_control_point", (index, x, y)) {
                        eprintln!("Py Error (Add): {}", e);
                    }
                }
                PyCommand::FinalizeSetup => {
                    println!("Rust: Finalizing Setup...");
                    if let Err(e) = bridge_bound.call_method0("finalize_setup") {
                        eprintln!("Py Error (Finalize): {}", e);
                    } else {
                        let _ = tx_res.send(PyResult::SetupFinalized);
                    }
                }
                PyCommand::StartDrag => {
                    if let Err(e) = bridge_bound.call_method0("start_drag_operation") {
                        eprintln!("Py Error (StartDrag): {}", e);
                    }
                }
                PyCommand::UpdatePoint { control_index, x, y } => {
                    if let Err(e) = bridge_bound.call_method1("update_control_point", (control_index, x, y)) {
                        eprintln!("Py Error (UpdatePoint): {}", e);
                    } else {
                        if let Ok(res) = bridge_bound.call_method0("solve_frame") {
                            if let Ok(params) = res.downcast::<pyo3::types::PyDict>() {
                                let coeffs = extract_from_dict(params, "coefficients");
                                let centers = extract_from_dict(params, "centers");
                                let s = extract_from_dict(params, "s_param");
                                let n = extract_from_dict(params, "n_rbf");
                                let w = extract_from_dict(params, "image_width");
                                let h = extract_from_dict(params, "image_height");
                                let inv_grid = extract_from_dict(params, "inverse_grid");
                                let grid_w = extract_from_dict(params, "grid_width");
                                let grid_h = extract_from_dict(params, "grid_height");

                                if let (Some(coeffs), Some(centers), Some(s), Some(n), Some(w), Some(h), Some(inv_grid), Some(grid_w), Some(grid_h)) =
                                    (coeffs, centers, s, n, w, h, inv_grid, grid_w, grid_h) {
                                    let _ = tx_res.send(PyResult::MappingParameters {
                                        coefficients: coeffs,
                                        centers,
                                        s_param: s,
                                        n_rbf: n,
                                        image_width: w,
                                        image_height: h,
                                        inverse_grid: inv_grid,
                                        grid_width: grid_w,
                                        grid_height: grid_h,
                                    });
                                }
                            }
                        }
                    }
                }
                PyCommand::EndDrag => {
                    if let Err(e) = bridge_bound.call_method0("end_drag_operation") {
                        eprintln!("Py Error (EndDrag): {}", e);
                    } else {
                        if let Ok(res) = bridge_bound.call_method0("solve_frame") {
                            if let Ok(params) = res.downcast::<pyo3::types::PyDict>() {
                                let coeffs = extract_from_dict(params, "coefficients");
                                let centers = extract_from_dict(params, "centers");
                                let s = extract_from_dict(params, "s_param");
                                let n = extract_from_dict(params, "n_rbf");
                                let w = extract_from_dict(params, "image_width");
                                let h = extract_from_dict(params, "image_height");
                                let inv_grid = extract_from_dict(params, "inverse_grid");
                                let grid_w = extract_from_dict(params, "grid_width");
                                let grid_h = extract_from_dict(params, "grid_height");

                                if let (Some(coeffs), Some(centers), Some(s), Some(n), Some(w), Some(h), Some(inv_grid), Some(grid_w), Some(grid_h)) =
                                    (coeffs, centers, s, n, w, h, inv_grid, grid_w, grid_h) {
                                    let _ = tx_res.send(PyResult::MappingParameters {
                                        coefficients: coeffs,
                                        centers,
                                        s_param: s,
                                        n_rbf: n,
                                        image_width: w,
                                        image_height: h,
                                        inverse_grid: inv_grid,
                                        grid_width: grid_w,
                                        grid_height: grid_h,
                                    });
                                }
                            }
                        }
                    }
                }
                PyCommand::Reset => {
                    if let Err(e) = bridge_bound.call_method0("reset_mesh") {
                        eprintln!("Py Error (Reset): {}", e);
                    }
                }
            }
        });
    }
}
