{
  description = "Provably Good Planar Mappings v3 – pgpm-core + bevy-pgpm";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    rust-overlay.url = "github:oxalica/rust-overlay";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs = { self, nixpkgs, rust-overlay, flake-utils, ... }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        overlays = [ (import rust-overlay) ];
        pkgs = import nixpkgs {
          inherit system overlays;
        };

        # Rustツールチェーン
        rustToolchain = pkgs.rust-bin.stable.latest.default.override {
          extensions = [ "rust-src" "rust-analyzer" ];
        };

        # ビルドツール
        nativeBuildInputs = with pkgs; [
          pkg-config
          rustToolchain
          clang
          mold
        ];

        # Nerd Fonts (FiraCode — ASCII + Unicode記号 + Nerdアイコン all-in-one)
        nerdFonts = pkgs.nerd-fonts.fira-code;

        # Bevyに必要なネイティブライブラリ群 (Linux用)
        buildInputs = with pkgs; [
          udev
          alsa-lib
          vulkan-loader
          # X11
          libx11
          libxcursor
          libxi
          libxrandr
          # Wayland
          libxkbcommon
          wayland
        ];

      in
      {
        devShells.default = pkgs.mkShell {
          inherit nativeBuildInputs buildInputs;

          RUSTFLAGS = "-C link-arg=-fuse-ld=mold";
          LD_LIBRARY_PATH = pkgs.lib.makeLibraryPath buildInputs;

          shellHook = ''
            echo "=========================================================="
            echo "🚀 PGPM v3 Dev Shell Activated"
            echo "Rust: $(rustc --version)"
            echo "=========================================================="
            export WGPU_BACKEND=vulkan

            # Nerd Fonts を assets/fonts/ にリンク
            mkdir -p crates/bevy-pgpm/assets/fonts
            ln -sfn ${nerdFonts}/share/fonts/truetype/NerdFonts/FiraCode/FiraCodeNerdFontMono-Regular.ttf \
              crates/bevy-pgpm/assets/fonts/FiraCodeNerdFontMono-Regular.ttf
          '';
        };

        # nix run で bevy-pgpm を実行
        apps.default = {
          type = "app";
          program = "${pkgs.writeShellScriptBin "run-bevy-pgpm" ''
            export PATH="${pkgs.lib.makeBinPath nativeBuildInputs}:$PATH"
            export PKG_CONFIG_PATH="${pkgs.lib.makeSearchPathOutput "dev" "lib/pkgconfig" buildInputs}:${pkgs.lib.makeSearchPathOutput "dev" "share/pkgconfig" buildInputs}''${PKG_CONFIG_PATH:+:$PKG_CONFIG_PATH}"
            export LD_LIBRARY_PATH="${pkgs.lib.makeLibraryPath buildInputs}''${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
            export RUSTFLAGS="-C link-arg=-fuse-ld=mold"
            export WGPU_BACKEND=vulkan

            # nix run が実行されたディレクトリ（= flake.nix のあるディレクトリ）で cargo run
            exec cargo run -p bevy-pgpm
          ''}/bin/run-bevy-pgpm";
        };
      }
    );
}
