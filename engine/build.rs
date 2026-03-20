use std::fs;
use std::path::PathBuf;
use std::process::Command;

fn main() {
    println!("cargo:rerun-if-env-changed=PYO3_PYTHON");
    println!("cargo:rerun-if-env-changed=PYTHON_SYS_EXECUTABLE");
    println!("cargo:rerun-if-env-changed=PYO3_BUILD_EXTENSION_MODULE");

    let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let repo_root = manifest_dir
        .parent()
        .and_then(|p| p.parent())
        .expect("repo root");
    let proto_path = repo_root.join("proto").join("cambc.proto");
    println!("cargo:rerun-if-changed={}", proto_path.display());
    let out_dir = PathBuf::from(std::env::var("OUT_DIR").expect("OUT_DIR missing"));
    let expected = out_dir.join("battlecode.rs");
    if expected.exists() {
        let _ = fs::remove_file(&expected);
    }
    let mut config = prost_build::Config::new();
    config.out_dir(&out_dir);
    config
        .compile_protos(&[proto_path], &[repo_root.join("proto")])
        .expect("failed to compile cambc.proto");
    if !expected.exists() {
        let mut generated = None;
        if let Ok(entries) = fs::read_dir(&out_dir) {
            for entry in entries.flatten() {
                let path = entry.path();
                if path.extension().and_then(|e| e.to_str()) == Some("rs") {
                    generated = Some(path);
                    break;
                }
            }
        }
        if let Some(path) = generated {
            fs::copy(&path, &expected)
                .unwrap_or_else(|_| panic!("failed to copy {} to battlecode.rs", path.display()));
        } else {
            panic!(
                "prost-build did not generate battlecode.rs in {}",
                out_dir.display()
            );
        }
    }

    // Only emit python link flags for the standalone binary (titan_runner).
    // Extension modules (cdylib for the CLI) must not link libpython — PyO3's
    // extension-module feature handles this. Maturin sets this env var.
    if std::env::var("PYO3_BUILD_EXTENSION_MODULE").is_ok() {
        return;
    }

    let output = Command::new("python3-config")
        .args(["--embed", "--ldflags"])
        .output();

    let output = match output {
        Ok(out) if out.status.success() => out,
        _ => return,
    };

    let flags = String::from_utf8_lossy(&output.stdout);
    for token in flags.split_whitespace() {
        if let Some(path) = token.strip_prefix("-L") {
            println!("cargo:rustc-link-search=native={}", path);
        } else if let Some(lib) = token.strip_prefix("-l") {
            println!("cargo:rustc-link-lib={}", lib);
        }
    }
}
