pub struct PlatformInfo {
    pub os: &'static str,
    pub arch: &'static str,
}

pub fn detect() -> PlatformInfo {
    PlatformInfo {
        os: std::env::consts::OS,
        arch: std::env::consts::ARCH,
    }
}

pub fn print() {
    let platform: PlatformInfo = detect();
    println!("Platform: {}: ({})", platform.os, platform.arch);
}
