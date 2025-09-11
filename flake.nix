{
  description = "python packages";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-25.05";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs = { self, nixpkgs, flake-utils, ... }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        pkgs = import nixpkgs { # system パラメータを渡す
          inherit system;
        };
      in
      {
        devShells.default = pkgs.mkShell {
          packages = with pkgs; [
            python3
          (pkgs.python3.withPackages (python-pkgs: [
            python-pkgs.matplotlib
            python-pkgs.numpy
            python-pkgs.networkx
            python-pkgs.scipy
            python-pkgs.cvxpy
            python-pkgs.ecos
          ]))
          ];
        };
      }
    );
}
