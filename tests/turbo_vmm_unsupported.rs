#![cfg(feature = "turbo")]

//! On modern NVIDIA hardware (Pascal+, driver ≥ 460) VMM is supported and
//! `SlabAllocator::new` returns Ok. This test only covers the *negative*
//! routing of `VmmError::Unsupported` so callers can pattern-match on it
//! without panic.

use inference_flame::turbo::vmm::VmmError;

#[test]
fn unsupported_variant_displays_cleanly() {
    let e = VmmError::Unsupported;
    let s = format!("{e}");
    assert!(s.contains("not supported"), "{s}");

    // Pattern match must still distinguish.
    match e {
        VmmError::Unsupported => {}
        _ => panic!("variant did not match"),
    }
}
