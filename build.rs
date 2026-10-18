//! FluidAudio's Swift bridge links against the OS Swift runtime, which
//! macOS ships under /usr/lib/swift but which is not on a Rust binary's
//! rpath by default. A library's build script can only set link arguments
//! for its own targets, so this covers euhadra's tests and examples;
//! downstream binaries enabling `fluidaudio` must add the same rpath.
fn main() {
    println!("cargo:rerun-if-changed=build.rs");
    let fluidaudio = std::env::var_os("CARGO_FEATURE_FLUIDAUDIO").is_some();
    let macos = std::env::var("CARGO_CFG_TARGET_OS").as_deref() == Ok("macos");
    if fluidaudio && macos {
        println!("cargo:rustc-link-arg=-Wl,-rpath,/usr/lib/swift");
    }
}
