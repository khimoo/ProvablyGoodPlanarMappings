pub enum PyCommand {
    InitializeDomain {
        width: f32,
        height: f32,
        epsilon: f32,
        strategy: String,           // "strategy1" or "strategy2"
        strategy_params: String,    // JSON format
    },
    SetContour { contour: Vec<(f32, f32)> },
    AddControlPoint { index: usize, x: f32, y: f32 },
    FinalizeSetup,
    StartDrag,
    UpdatePoint { control_index: usize, x: f32, y: f32 },
    EndDrag,
    Reset,
    GetDebugVisualization,  // 視覚化データをリクエスト
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
    DebugVisualization {
        collocation_points: Vec<(f32, f32)>,  // グリッド点
        active_set: Vec<(f32, f32)>,          // 歪みが大きい点
        contour: Vec<(f32, f32)>,             // 輪郭線
    },
}
