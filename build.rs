// build.rs
fn main() {
    #[cfg(feature = "node")]
    napi_build::setup();
}