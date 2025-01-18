{
  description = "OramaCore";

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
          config = {
            allowUnfree = true;
            cudaSupport = true;
          };
        };
        
        rustToolchain = pkgs.rust-bin.stable."1.84.0".default.override {
          extensions = [ "rust-src" ];
        };

        pythonEnv = pkgs.python311.withPackages (ps: with ps; [
          numpy
          grpcio-tools
          grpcio
          python-dotenv
          grpcio-reflection
          pyyaml
          cachetools
          pillow
          pytest
          pytest-mock
          transformers
          protobuf
        ]);

      in {
        devShell = pkgs.mkShell {
          buildInputs = with pkgs; [
            rustToolchain
            pkg-config
            openssl
            gflags
            snappy
            zlib
            bzip2
            zstd
            libxml2
            protobuf
            pythonEnv
            grpcurl
          ];

          shellHook = ''
            export RUST_SRC_PATH="${rustToolchain}/lib/rustlib/src/rust/library"
            export LD_LIBRARY_PATH="${pkgs.lib.makeLibraryPath [
              pkgs.openssl
              pkgs.gflags
              pkgs.snappy
              pkgs.zlib
              pkgs.bzip2
              pkgs.zstd
              pkgs.libxml2
            ]}"
            export PYTHONPATH="$PYTHONPATH:${pythonEnv}/${pythonEnv.sitePackages}"
          '';
        };
      }
    );
}