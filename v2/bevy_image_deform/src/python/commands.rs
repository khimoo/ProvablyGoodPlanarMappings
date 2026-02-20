pub enum PyCommand {
    InitializeDomain { width: f32, height: f32, epsilon: f32 },
    SetContour { contour: Vec<(f32, f32)> },
    AddControlPoint { index: usize, x: f32, y: f32 },
    FinalizeSetup,
    StartDrag,
    UpdatePoint { control_index: usize, x: f32, y: f32 },
    EndDrag,
    Reset,
}

pub enum PyResult {
    DomainInitialized,
    SetupFinalized,
    MappingParameters {
        coefficients: Vec<Vec<f32>>,
        centers: Vec<Vec<f32>>,
        s_param: f32,
        n_rbf: usize,
        image_width: f32,
        image_height: f32,
        inverse_grid: Vec<Vec<Vec<f32>>>,
        grid_width: usize,
        grid_height: usize,
    },
}
