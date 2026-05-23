#[cfg(feature = "compare-c")]
fn main() {
    use std::path::PathBuf;

    let src_dir = PathBuf::from("src/liblbfgs");
    if !src_dir.join("lib").exists() {
        panic!("liblbfgs submodule not initialized. Run: git submodule update --init --recursive");
    }

    let mut cmake_cfg = cmake::Config::new(".");
    cmake_cfg.define("CMAKE_POLICY_VERSION_MINIMUM", "3.5");
    if cfg!(target_os = "macos") && cfg!(target_arch = "x86_64") {
        cmake_cfg.define("CMAKE_C_FLAGS", "-mmacosx-version-min=10.9");
    }
    let dst = cmake_cfg.build();

    println!(
        "cargo:rustc-link-search=native={}",
        dst.join("lib").display()
    );
    println!("cargo:rustc-link-lib=static=lbfgs");
    println!("cargo:include={}", dst.join("include").display());
}

#[cfg(not(feature = "compare-c"))]
fn main() {}
