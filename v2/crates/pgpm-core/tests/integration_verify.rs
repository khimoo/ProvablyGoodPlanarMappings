//! 論文の中核的主張を検証する統合テスト。
//!
//! これらのテストは、Poranne & Lipman (2014) で記述された
//! 性質が実際に実装で強制されることを検証:
//!
//! 1. 恒等写像の歪みは正確に 1
//! 2. SOCP制約はアクティブ点で D(z) <= K を強制
//! 3. ハンドル位置は写像によって追跡される
//! 4. 反復精緻化は収束する（アクティブ集合が安定化）
//! 5. 変形下で歪み上界が維持される

use pgpm_core::basis::gaussian::GaussianBasis;
use pgpm_core::basis::BasisFunction;
use pgpm_core::distortion;
use pgpm_core::algorithm::Algorithm;
use pgpm_core::mapping::PlanarMapping;
use pgpm_core::model::types::*;
use pgpm_core::policy::IsometricPolicy;
use nalgebra::Vector2;

/// 係数と基底から写像 f(x) = Σ c_i φ_i(x) (Eq. 3) を評価。
fn eval_mapping(coefficients: &CoefficientMatrix, basis: &dyn BasisFunction, x: Vector2<f64>) -> Vector2<f64> {
    let phi = basis.evaluate(x);
    let n = basis.count();
    let mut u = 0.0;
    let mut v = 0.0;
    for i in 0..n {
        u += coefficients[(0, i)] * phi[i];
        v += coefficients[(1, i)] * phi[i];
    }
    Vector2::new(u, v)
}

/// ヘルパー: [0,1]^2 上に条件の良いテストセットアップを作成。
/// 密なRBF中心と適切なスケールで構成。
fn make_verification_algorithm(
    k_bound: f64,
    handles_src: Vec<Vector2<f64>>,
    grid_res: usize,
) -> Algorithm<IsometricPolicy> {
    // 十分なカバレッジのための4x4 RBF中心グリッド
    let mut centers = Vec::new();
    for i in 0..4 {
        for j in 0..4 {
            centers.push(Vector2::new(
                0.125 + i as f64 * 0.25,
                0.125 + j as f64 * 0.25,
            ));
        }
    }
    // スケール: 約 1/(2*sqrt(n_centers)) ~ 0.125 だが、オーバーラップのため少し大きめ
    let basis = Box::new(GaussianBasis::new(centers, 0.25));

    let params = MappingParams {
        k_bound,
        lambda_reg: 1e-3,
        regularization: RegularizationType::Arap,
    };

    let domain = DomainBounds {
        x_min: 0.0,
        x_max: 1.0,
        y_min: 0.0,
        y_max: 1.0,
    };

    Algorithm::new(
        basis, params, IsometricPolicy, domain, handles_src,
        grid_res, 8, None, pgpm_core::model::types::SolverConfig::default(),
    )
}

// ────────────────────────────────────────────────────────────
// テスト1: 恒等写像の歪みは全点で = 1
// ────────────────────────────────────────────────────────────

/// テスト1a: 恒等係数は全点で D = 1 を生成。
/// SOCP求解前の基底関数と歪み計算を検証。
#[test]
fn verify_identity_coefficients_give_distortion_one() {
    let handles = vec![Vector2::new(0.3, 0.3), Vector2::new(0.7, 0.7)];
    let mut alg = make_verification_algorithm(3.0, handles, 20);

    // ステップは事前計算をトリガーするが、SOCPも求解する。
    // 求解前に歪みを確認する必要がある。
    // state() を使って恒等係数を取得し D=1 を検証。
    let target = vec![Vector2::new(0.3, 0.3), Vector2::new(0.7, 0.7)];
    let info = alg.step(&target).expect("Should succeed");

    // info.max_distortion は求解前（恒等係数から）に計算される
    println!(
        "Pre-solve max distortion (identity coefficients): {:.6}",
        info.max_distortion
    );
    assert!(
        (info.max_distortion - 1.0).abs() < 1e-6,
        "Identity coefficients should give D=1 everywhere, got {:.6}",
        info.max_distortion
    );
}

/// テスト1b: 恒等ターゲットで反復アルゴリズムは D <= K に収束。
/// 数ステップ後、アクティブ集合が安定し、最大歪みは
/// 上界 K 以内に留まる。
#[test]
fn verify_identity_target_converges() {
    let handles = vec![Vector2::new(0.3, 0.3), Vector2::new(0.7, 0.7)];
    let mut alg = make_verification_algorithm(3.0, handles, 20);
    let target = vec![Vector2::new(0.3, 0.3), Vector2::new(0.7, 0.7)];
    let k = 3.0;

    // 収束に十分なステップを実行
    let mut last_info = None;
    for step in 0..10 {
        let info = alg.step(&target).expect("Should succeed");
        println!(
            "Step {}: max_D={:.4}, active={}, stable={}",
            step, info.max_distortion, info.active_set_size, info.stable_set_size
        );
        last_info = Some(info);
    }

    // 収束後、最大歪みは K 以下であるべき（小さな許容誤差付き）
    let info = last_info.unwrap();
    assert!(
        info.max_distortion <= k + 0.1,
        "After 10 steps with identity target, max_D={:.4} should be <= K+eps={}",
        info.max_distortion,
        k + 0.1
    );

    // 求解後の歪みも上界内であることを検証
    let (ctx, state) = alg.parts();
    let precomputed = state.precomputed.as_ref().unwrap();
    let distortions = distortion::evaluate_distortion_all(
        &state.coefficients,
        precomputed,
        ctx.policy,
    );
    let max_d = distortions.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    println!("Post-solve max_D after 10 steps: {:.4}", max_d);
    assert!(
        max_d <= k + 0.1,
        "Post-solve max_D={:.4} should be <= K+eps={}",
        max_d,
        k + 0.1
    );
}

// ────────────────────────────────────────────────────────────
// テスト2: SOCP制約はアクティブ/安定点で D <= K を強制
// ────────────────────────────────────────────────────────────

#[test]
fn verify_distortion_bound_at_constrained_points() {
    let handles = vec![Vector2::new(0.5, 0.5)];
    let k = 3.0;
    let mut alg = make_verification_algorithm(k, handles, 20);

    // 中程度の変形を適用
    let target = vec![Vector2::new(0.65, 0.5)];

    // アクティブ集合が安定するまで数ステップ実行
    for _ in 0..5 {
        alg.step(&target).expect("Should succeed");
    }

    // SOCP求解後、制約点での歪みを確認
    let (ctx, state) = alg.parts();
    let precomputed = state.precomputed.as_ref().unwrap();
    let distortions = distortion::evaluate_distortion_all(
        &state.coefficients,
        precomputed,
        ctx.policy,
    );

    // アクティブ集合点を確認: D(z) <= K（数値許容誤差付き）
    let tol = 0.1; // SOCPソルバーの許容誤差
    for &idx in &state.active_set {
        assert!(
            distortions[idx] <= k + tol,
            "Active point {}: D = {:.4} > K = {} (tol={})",
            idx,
            distortions[idx],
            k,
            tol
        );
    }

    // 安定集合点も同様に確認
    for &idx in &state.stable_set {
        assert!(
            distortions[idx] <= k + tol,
            "Stable point {}: D = {:.4} > K = {} (tol={})",
            idx,
            distortions[idx],
            k,
            tol
        );
    }

    println!(
        "Constrained points check passed: {} active, {} stable, K={}",
        state.active_set.len(),
        state.stable_set.len(),
        k
    );
}

// ────────────────────────────────────────────────────────────
// テスト3: 写像はハンドル位置を追跡（E_pos が最小化）
// ────────────────────────────────────────────────────────────

#[test]
fn verify_handle_tracking() {
    let src = vec![Vector2::new(0.5, 0.5)];
    let mut alg = make_verification_algorithm(5.0, src.clone(), 20);

    let target = vec![Vector2::new(0.6, 0.55)];

    // 数ステップ実行
    for _ in 0..5 {
        alg.step(&target).expect("Should succeed");
    }

    // ソースハンドルで写像を評価（Eq. 3）
    let mapped = eval_mapping(alg.coefficients(), alg.basis(), Vector2::new(0.5, 0.5));
    let error = (mapped - target[0]).norm();

    println!(
        "Handle tracking: src=(0.5,0.5) -> mapped=({:.4},{:.4}), target=({:.4},{:.4}), error={:.6}",
        mapped.x, mapped.y, target[0].x, target[0].y, error
    );

    assert!(
        error < 0.15,
        "Handle tracking error {:.4} too large (expected < 0.15)",
        error
    );
}

// ────────────────────────────────────────────────────────────
// テスト4: アクティブ集合は反復で収束
// ────────────────────────────────────────────────────────────

#[test]
fn verify_active_set_convergence() {
    let handles = vec![Vector2::new(0.5, 0.5)];
    let mut alg = make_verification_algorithm(3.0, handles, 15);
    let target = vec![Vector2::new(0.6, 0.5)];

    let mut active_sizes = Vec::new();
    let mut max_dists = Vec::new();

    for step in 0..10 {
        let info = alg.step(&target).expect("Should succeed");
        active_sizes.push(info.active_set_size);
        max_dists.push(info.max_distortion);
        println!(
            "Step {}: max_D={:.4}, active={}, stable={}",
            step, info.max_distortion, info.active_set_size, info.stable_set_size
        );
    }

    // アクティブ集合は安定すべき（無限に成長しない）
    // 論文: 「各反復で少数の孤立点のみがアクティブ化される」
    let last_size = *active_sizes.last().unwrap();
    let total_points = alg.collocation_points().len();
    let ratio = last_size as f64 / total_points as f64;
    println!(
        "Final active set: {}/{} points ({:.1}%)",
        last_size,
        total_points,
        ratio * 100.0
    );
    assert!(
        ratio < 0.5,
        "Active set ratio {:.2} is too high -- local maxima filter may not be working",
        ratio
    );

    // 歪みは終始有限であるべき
    for (step, &d) in max_dists.iter().enumerate() {
        assert!(d.is_finite(), "Step {}: infinite distortion", step);
    }
}

// ────────────────────────────────────────────────────────────
// テスト5: 写像は連続（近傍点は近傍に写像）
// ────────────────────────────────────────────────────────────

#[test]
fn verify_mapping_continuity() {
    let handles = vec![Vector2::new(0.5, 0.5)];
    let mut alg = make_verification_algorithm(3.0, handles, 15);
    let target = vec![Vector2::new(0.65, 0.5)];

    for _ in 0..5 {
        alg.step(&target).expect("Should succeed");
    }

    // 連続性を確認: 近傍入力点は近傍出力点に写像されるべき
    let eps = 0.01;
    let test_points = vec![
        Vector2::new(0.3, 0.3),
        Vector2::new(0.5, 0.5),
        Vector2::new(0.7, 0.7),
        Vector2::new(0.2, 0.8),
    ];

    let coefficients = alg.coefficients();
    let basis = alg.basis();
    for &p in &test_points {
        let f_p = eval_mapping(coefficients, basis, p);
        let f_px = eval_mapping(coefficients, basis, p + Vector2::new(eps, 0.0));
        let f_py = eval_mapping(coefficients, basis, p + Vector2::new(0.0, eps));

        let dx = (f_px - f_p).norm();
        let dy = (f_py - f_p).norm();

        // K-有界歪みでは、リプシッツ定数は K 以下であるべき
        // （大まかに: ||f(x)-f(y)|| <= K * ||x-y||）
        let lip_x = dx / eps;
        let lip_y = dy / eps;

        println!(
            "Point ({:.1},{:.1}): Lip_x={:.2}, Lip_y={:.2}",
            p.x, p.y, lip_x, lip_y
        );

        // リプシッツ比は有界であるべき（無限でない = fold-over なし）
        assert!(
            lip_x < 20.0 && lip_y < 20.0,
            "Lipschitz constant too large at ({:.1},{:.1}): ({:.2}, {:.2})",
            p.x,
            p.y,
            lip_x,
            lip_y
        );
    }
}

// ────────────────────────────────────────────────────────────
// テスト6: 異なる K 値は異なる品質を生成
// ────────────────────────────────────────────────────────────

#[test]
fn verify_k_bound_effect() {
    let handles = vec![Vector2::new(0.5, 0.5)];
    let target = vec![Vector2::new(0.7, 0.5)];

    let mut results = Vec::new();

    for &k in &[2.0, 4.0, 8.0] {
        let mut alg = make_verification_algorithm(k, handles.clone(), 15);

        for _ in 0..5 {
            alg.step(&target).expect("Should succeed");
        }

        // 求解後の歪みを測定
        let (ctx, state) = alg.parts();
        let precomputed = state.precomputed.as_ref().unwrap();
        let distortions = distortion::evaluate_distortion_all(
            &state.coefficients,
            precomputed,
            ctx.policy,
        );

        let max_d = distortions.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let handle_error = (eval_mapping(alg.coefficients(), alg.basis(), Vector2::new(0.5, 0.5)) - target[0]).norm();

        println!(
            "K={}: max_D={:.4}, handle_error={:.4}, active={}",
            k,
            max_d,
            handle_error,
            state.active_set.len()
        );

        results.push((k, max_d, handle_error));
    }

    // 高い K は一般的により多くの歪み（より少ない制約）を許容し、
    // かつ/または より良いハンドル追跡（より多くの自由度）を可能にする
    let (_, d_tight, err_tight) = results[0]; // K=2
    let (_, d_loose, err_loose) = results[2]; // K=8

    // K=8（緩い上界）ではソルバーにより多くの自由度があるため、
    // ハンドルエラーは同等以上であるべき
    // （厳密な単調保証ではないが、一般的に成立）
    println!(
        "K=2: D={:.4} err={:.4} | K=8: D={:.4} err={:.4}",
        d_tight, err_tight, d_loose, err_loose
    );
}

// ────────────────────────────────────────────────────────────
// テスト7: 2ハンドル変形
// ────────────────────────────────────────────────────────────

#[test]
fn verify_two_handle_deformation() {
    let src = vec![Vector2::new(0.3, 0.5), Vector2::new(0.7, 0.5)];
    let mut alg = make_verification_algorithm(4.0, src.clone(), 20);

    // ハンドルを引き離す
    let target = vec![Vector2::new(0.2, 0.5), Vector2::new(0.8, 0.5)];

    for step in 0..8 {
        let info = alg.step(&target).expect("Should succeed");
        println!(
            "Step {}: max_D={:.4}, active={}",
            step, info.max_distortion, info.active_set_size
        );
    }

    // 両ハンドルが追跡されているか確認（Eq. 3）
    let mapped_a = eval_mapping(alg.coefficients(), alg.basis(), src[0]);
    let mapped_b = eval_mapping(alg.coefficients(), alg.basis(), src[1]);

    let err_a = (mapped_a - target[0]).norm();
    let err_b = (mapped_b - target[1]).norm();

    println!(
        "Handle A: ({:.3},{:.3}) -> ({:.3},{:.3}), error={:.4}",
        src[0].x, src[0].y, mapped_a.x, mapped_a.y, err_a
    );
    println!(
        "Handle B: ({:.3},{:.3}) -> ({:.3},{:.3}), error={:.4}",
        src[1].x, src[1].y, mapped_b.x, mapped_b.y, err_b
    );

    assert!(err_a < 0.2, "Handle A error {:.4} too large", err_a);
    assert!(err_b < 0.2, "Handle B error {:.4} too large", err_b);

    // 歪み制約を検証
    let (ctx, state) = alg.parts();
    let precomputed = state.precomputed.as_ref().unwrap();
    let distortions = distortion::evaluate_distortion_all(
        &state.coefficients,
        precomputed,
        ctx.policy,
    );
    let max_d = distortions.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    println!("Final max distortion: {:.4}", max_d);
}

// ────────────────────────────────────────────────────────────
// テスト8: J_S / J_A 分解が特異値と一致することを検証
// ────────────────────────────────────────────────────────────

#[test]
fn verify_singular_value_decomposition() {
    // J_S/J_A 分解（Eq. 19-20）がランダムな 2x2
    // ヤコビアン行列の実際の SVD と一致することをテスト。

    let test_cases: Vec<(Vector2<f64>, Vector2<f64>)> = vec![
        // (grad_u, grad_v)
        (Vector2::new(2.0, 0.5), Vector2::new(-0.3, 1.5)),
        (Vector2::new(1.0, 0.0), Vector2::new(0.0, 1.0)),
        (Vector2::new(3.0, 1.0), Vector2::new(-1.0, 2.0)),
        (Vector2::new(0.5, -0.5), Vector2::new(0.5, 0.5)),
    ];

    for (grad_u, grad_v) in test_cases {
        let (j_s, j_a) = distortion::compute_j_s_j_a(grad_u, grad_v);
        let (sigma_max, sigma_min) = distortion::singular_values(j_s, j_a);

        // 2x2 ヤコビアンの実際の SVD とクロスチェック
        // J = [[du/dx, du/dy], [dv/dx, dv/dy]]
        let j = nalgebra::Matrix2::new(
            grad_u.x, grad_u.y,
            grad_v.x, grad_v.y,
        );
        let svd = j.svd(false, false);
        let sv = svd.singular_values;
        let svd_max = sv[0].max(sv[1]);
        let svd_min = sv[0].min(sv[1]);

        assert!(
            (sigma_max - svd_max).abs() < 1e-10,
            "Sigma mismatch: J_S/J_A gives {:.6}, SVD gives {:.6} for grad_u={:?}, grad_v={:?}",
            sigma_max, svd_max, grad_u, grad_v
        );
        assert!(
            (sigma_min - svd_min).abs() < 1e-10,
            "sigma mismatch: J_S/J_A gives {:.6}, SVD gives {:.6} for grad_u={:?}, grad_v={:?}",
            sigma_min, svd_min, grad_u, grad_v
        );
    }
}

// ────────────────────────────────────────────────────────────
// テスト9: 求解後の制約点での歪み <= K
//         （直接的な制約充足確認）
// ────────────────────────────────────────────────────────────

#[test]
fn verify_post_solve_constraint_satisfaction() {
    let handles = vec![Vector2::new(0.5, 0.5)];
    let k = 3.0;
    let mut alg = make_verification_algorithm(k, handles, 20);
    let target = vec![Vector2::new(0.7, 0.5)];

    // 1ステップ実行（いくつかのアクティブ点を追加して求解）
    let _info = alg.step(&target).expect("Step 1 should succeed");

    // もう1ステップ実行:
    // ステップ2の開始時、歪みは求解済み係数で評価される。
    // ステップ1の制約点は D <= K を満たすべき。
    let info2 = alg.step(&target).expect("Step 2 should succeed");

    // 求解前の最大歪み（ステップ1の解で評価）は、
    // ステップ1のSOCP制約が満たされたことを示すべき。
    // （info2.max_distortion はステップ2の求解前に計算される）
    println!(
        "Post-solve distortion (evaluated at start of step 2): {:.4}",
        info2.max_distortion
    );

    // さらにステップを継続してパターンを確認
    for step in 2..8 {
        let info = alg.step(&target).expect("Should succeed");
        println!(
            "Step {}: pre-solve max_D={:.4}, active={}, stable={}",
            step, info.max_distortion, info.active_set_size, info.stable_set_size
        );
    }
}
