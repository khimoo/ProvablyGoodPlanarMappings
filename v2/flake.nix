{
  description = "Bevy + Python (PyO3) Image Deformation Dev Environment";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    rust-overlay.url = "github:oxalica/rust-overlay";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs = { self, nixpkgs, rust-overlay, flake-utils, ... }:
    flake_utils.lib.eachDefaultSystem (system:
      let
        overlays = [ (import rust-overlay) ];
        pkgs = import nixpkgs {
          inherit system overlays;
        };

        # 1. Python環境の定義
        # userのコードで必要なライブラリを全て含めます
        myPython = pkgs.python3.withPackages (ps: with ps; [
          numpy
          scipy
          matplotlib
          cvxpy
          ecos # cvxpyで指定されていたソルバー
          # その他必要ならここに追加 (例: jupyter, ipythonなど)
        ]);

        # 2. Rustツールチェーンの定義
        rustToolchain = pkgs.rust-bin.stable.latest.default.override {
          extensions = [ "rust-src" "rust-analyzer" ];
        };

        # 3. Bevyに必要なネイティブライブラリ群 (Linux用)
        nativeBuildInputs = with pkgs; [
          pkg-config
          rustToolchain
          clang
          lld
        ];

        buildInputs = with pkgs; [
          udev
          alsa-lib
          vulkan-loader
          # X11 dependencies
          xorg.libX11
          xorg.libXcursor
          xorg.libXi
          xorg.libXrandr
          # Wayland dependencies (optional but recommended)
          libxkbcommon
          wayland

          # Python本体 (ヘッダーファイルと共有ライブラリ)
          myPython
        ];

      in
      {
        devShells.default = pkgs.mkShell {
          inherit nativeBuildInputs buildInputs;

          # 環境変数設定

          # PyO3がビルド時および実行時に正しいPythonを見つけられるようにする
          PYO3_PYTHON = "${myPython}/bin/python";

          # LinuxでBevyが実行時に共有ライブラリを見つけられるようにする魔法の設定
          # これがないと "vulkan icd not found" や "alsa error" で落ちます
          LD_LIBRARY_PATH = pkgs.lib.makeLibraryPath buildInputs;

          shellHook = ''
            echo "=========================================================="
            echo "🚀 Bevy + Python Dev Shell Activated"
            echo "Python: $(python --version)"
            echo "Rust: $(rustc --version)"
            echo "=========================================================="

            # wgpu (Bevyのレンダリングバックエンド) がVulkanを見つけやすくする
            export WGPU_BACKEND=vulkan
          '';
        };
      }
    );
}
