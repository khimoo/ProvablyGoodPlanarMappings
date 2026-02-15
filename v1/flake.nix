{
  description = "python packages";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-25.05";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs = { self, nixpkgs, flake-utils, ... }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        pkgs = import nixpkgs {
          inherit system;
        };

        pythonEnv = pkgs.python3.withPackages (python-pkgs: [
          python-pkgs.matplotlib
          python-pkgs.numpy
          python-pkgs.networkx
          python-pkgs.scipy
          python-pkgs.cvxpy
          python-pkgs.ecos
        ]);

        app = pkgs.stdenv.mkDerivation {
          name = "provably-good-planar-mappings";
          src = ./.;
          buildInputs = [ pythonEnv ];
          installPhase = ''
            mkdir -p $out/bin $out/share
            cp *.py $out/share/
            
            echo "#!${pkgs.bash}/bin/bash" > $out/bin/run-app
            echo "exec ${pythonEnv}/bin/python $out/share/ui-for-refactored2.py" >> $out/bin/run-app
            chmod +x $out/bin/run-app
          '';
        };
      in
      {
        packages.default = app;

        apps.default = flake-utils.lib.mkApp {
          drv = app;
          exePath = "/bin/run-app";
        };

        devShells.default = pkgs.mkShell {
          packages = [
            pythonEnv
          ];
        };
      }
    );
}
