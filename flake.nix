{
  description = "Dev Shell for python using Flake";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-24.05";
  };

  outputs = { self, nixpkgs }:
  let
    system = "x86_64-linux";
    pkgs = nixpkgs.legacyPackages.${system};
    py = pkgs.python3Packages;
  in
  {
    devShells.${system}.default = pkgs.mkShell {

      buildInputs = [
        py.python
        # Used to create a python enviroment, we can use pip normally with this
        py.venvShellHook

        # python packages that don't need to be controlled by pip
        py.numpy
        py.pandas
        py.matplotlib
        py.scapy
        py.scipy
        py.cython
        py.tensorflow
        py.scikit-learn

        # Any other package that my be needed in the shell can be added here too
        pkgs.wireshark
      ];

      venvDir = "./.env"; # Path to the python envirotment

      # As the shellHook is used to activate the Python Enviroment we enter our shell after it
      postShellHook = ''
        export SHELL=$(which nu)
        exec $SHELL # Run nushell at the start of the dev shell
      '';

    };

  };
}
