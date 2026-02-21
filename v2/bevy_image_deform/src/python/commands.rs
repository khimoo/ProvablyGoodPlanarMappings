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
    BasisFunctionParameters {
        coefficients: Vec<Vec<f32>>,      // (2, N+3)
        centers: Vec<Vec<f32>>,           // (N, 2)
        s_param: f32,                     // Gaussian width
        n_rbf: usize,                     // Number of RBF basis functions
    },
}
