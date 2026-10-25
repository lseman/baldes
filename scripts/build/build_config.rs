//! Dependency-free Make frontend: saved configuration, terminal menu and Cargo invocation.
//! Compiled directly by rustc so configuring does not build BALDES or its LP backends.
use std::collections::{BTreeMap, BTreeSet};
use std::env;
use std::ffi::{OsStr, OsString};
use std::fs;
use std::io::{self, IsTerminal, Read, Write};
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::time::{Duration, Instant};

mod build_config_ui;
use build_config_ui::{Item, State, View};

type Result<T> = std::result::Result<T, String>;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum SimdMode {
    Off,
    Avx2,
    Avx512,
}

#[derive(Clone, Copy)]
struct CpuSimd {
    avx2: bool,
    avx512: bool,
}

impl CpuSimd {
    fn detect() -> Self {
        #[cfg(target_arch = "x86_64")]
        return Self {
            avx2: std::is_x86_feature_detected!("avx2"),
            avx512: std::is_x86_feature_detected!("avx512f"),
        };
        #[cfg(not(target_arch = "x86_64"))]
        Self {
            avx2: false,
            avx512: false,
        }
    }

    fn supports(self, mode: SimdMode) -> bool {
        match mode {
            SimdMode::Off => true,
            SimdMode::Avx2 => self.avx2,
            SimdMode::Avx512 => self.avx512,
        }
    }
}

impl SimdMode {
    fn name(self) -> &'static str {
        match self {
            Self::Off => "off",
            Self::Avx2 => "avx2",
            Self::Avx512 => "avx512",
        }
    }

    fn next(self, cpu: CpuSimd, host: bool) -> Self {
        let modes = [Self::Off, Self::Avx2, Self::Avx512];
        let current = modes.iter().position(|&mode| mode == self).unwrap();
        (1..=3)
            .map(|step| modes[(current + step) % 3])
            .find(|&mode| mode == Self::Off || (host && cpu.supports(mode)))
            .unwrap()
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct Config {
    clp: bool,
    highs: bool,
    gurobi: bool,
    cli: bool,
    multi_time_windows: bool,
    /// Enforce Ryan–Foster branches through master rows instead of pricing.
    ryan_foster_master: bool,
    /// Enforce accumulated-resource branches through master rows only.
    accumulated_resource_master: bool,
    release: bool,
    native_cpu: bool,
    simd: SimdMode,
    release_debug: bool,
    offline: bool,
    locked: bool,
    jobs: usize,
    resource_dimensions: usize,
    target: String,
    highs_dir: String,
    gurobi_home: String,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            clp: false,
            highs: false,
            gurobi: false,
            cli: false,
            multi_time_windows: false,
            ryan_foster_master: false,
            accumulated_resource_master: false,
            release: true,
            native_cpu: false,
            simd: SimdMode::Off,
            release_debug: false,
            offline: false,
            locked: true,
            jobs: 0,
            resource_dimensions: 2,
            target: String::new(),
            highs_dir: String::new(),
            gurobi_home: String::new(),
        }
    }
}

impl Config {
    fn parse(text: &str) -> Result<Self> {
        let mut config = Self::default();
        let mut seen = BTreeSet::new();
        for (index, line) in text.lines().enumerate() {
            let line = line.trim();
            if line.is_empty() || line.starts_with('#') {
                continue;
            }
            let (key, value) = line
                .split_once('=')
                .ok_or_else(|| format!("line {}: expected CONFIG_NAME=value", index + 1))?;
            let key = key.trim();
            let value = value.trim();
            if !seen.insert(key) {
                return Err(format!("duplicate setting {key}"));
            }
            let boolean = || match value {
                "y" => Ok(true),
                "n" => Ok(false),
                _ => Err(format!("{key} must be y or n")),
            };
            match key {
                "CONFIG_CLP" => config.clp = boolean()?,
                "CONFIG_HIGHS" => config.highs = boolean()?,
                "CONFIG_GUROBI" => config.gurobi = boolean()?,
                "CONFIG_CLI" => config.cli = boolean()?,
                "CONFIG_MULTI_TIME_WINDOWS" => config.multi_time_windows = boolean()?,
                "CONFIG_ACCUMULATED_RESOURCE" => {
                    config.accumulated_resource_master = match value {
                        "pricing" => false,
                        "master" => true,
                        _ => {
                            return Err(
                                "CONFIG_ACCUMULATED_RESOURCE must be pricing or master".into()
                            )
                        }
                    }
                }
                "CONFIG_RYAN_FOSTER" => {
                    config.ryan_foster_master = match value {
                        "pricing" => false,
                        "master" => true,
                        _ => return Err("CONFIG_RYAN_FOSTER must be pricing or master".into()),
                    }
                }
                "CONFIG_SIMD" => {
                    config.simd = match value {
                        "off" | "n" => SimdMode::Off,
                        "avx2" | "y" => SimdMode::Avx2,
                        "avx512" => SimdMode::Avx512,
                        _ => return Err("CONFIG_SIMD must be off, avx2 or avx512".into()),
                    }
                }
                "CONFIG_NATIVE_CPU" => config.native_cpu = boolean()?,
                "CONFIG_RELEASE_DEBUG" => config.release_debug = boolean()?,
                "CONFIG_OFFLINE" => config.offline = boolean()?,
                "CONFIG_LOCKED" => config.locked = boolean()?,
                "CONFIG_PROFILE" => {
                    config.release = match value {
                        "release" => true,
                        "dev" => false,
                        _ => return Err("CONFIG_PROFILE must be release or dev".into()),
                    }
                }
                "CONFIG_JOBS" => {
                    config.jobs = value
                        .parse()
                        .map_err(|_| "CONFIG_JOBS must be a nonnegative integer (0 = automatic)")?
                }
                "CONFIG_RESOURCE_DIMENSIONS" => {
                    config.resource_dimensions = value
                        .parse()
                        .map_err(|_| "CONFIG_RESOURCE_DIMENSIONS must be a nonnegative usize")?;
                }
                "CONFIG_TARGET" => config.target = parse_string(value)?,
                "CONFIG_HIGHS_DIR" => config.highs_dir = parse_string(value)?,
                "CONFIG_GUROBI_HOME" => config.gurobi_home = parse_string(value)?,
                _ => return Err(format!("unknown setting {key}")),
            }
        }
        config.validate()?;
        Ok(config)
    }

    fn validate(&self) -> Result<()> {
        self.validate_simd(CpuSimd::detect())?;
        if self.cli && !self.clp {
            return Err("the baldes CLI requires CONFIG_CLP=y".into());
        }
        if self.native_cpu && !self.target.is_empty() {
            return Err(
                "native CPU optimization cannot be combined with a cross-compilation target".into(),
            );
        }
        if !self
            .target
            .chars()
            .all(|c| c.is_ascii_alphanumeric() || c == '-' || c == '_' || c == '.')
        {
            return Err(
                "target must be a Rust target triple (letters, numbers, '-', '_' or '.')".into(),
            );
        }
        if self.gurobi_home.chars().any(char::is_control) {
            return Err("Gurobi directory cannot contain control characters".into());
        }
        if self.highs_dir.chars().any(char::is_control) {
            return Err("HiGHS directory cannot contain control characters".into());
        }
        Ok(())
    }

    fn validate_simd(&self, cpu: CpuSimd) -> Result<()> {
        if self.simd != SimdMode::Off && !self.target.is_empty() {
            return Err(
                "SIMD modes require the host target; clear Target triple or select Off".into(),
            );
        }
        if !cpu.supports(self.simd) {
            return Err(format!(
                "SIMD mode {} is not supported by this CPU; select off",
                self.simd.name()
            ));
        }
        Ok(())
    }

    fn load(path: &Path) -> Result<Self> {
        match fs::read_to_string(path) {
            Ok(text) => Self::parse(&text).map_err(|e| format!("{}: {e}", path.display())),
            Err(e) if e.kind() == io::ErrorKind::NotFound => Ok(Self::default()),
            Err(e) => Err(format!("{}: {e}", path.display())),
        }
    }

    fn text(&self) -> String {
        let yn = |b| if b { "y" } else { "n" };
        format!(
            "# BALDES build configuration; edit with make config.\n\
            CONFIG_PROFILE={}\nCONFIG_CLP={}\nCONFIG_HIGHS={}\nCONFIG_GUROBI={}\nCONFIG_CLI={}\nCONFIG_MULTI_TIME_WINDOWS={}\n\
            CONFIG_RYAN_FOSTER={}\nCONFIG_ACCUMULATED_RESOURCE={}\n\
            CONFIG_SIMD={}\nCONFIG_NATIVE_CPU={}\nCONFIG_RELEASE_DEBUG={}\nCONFIG_OFFLINE={}\nCONFIG_LOCKED={}\n\
            CONFIG_JOBS={}\nCONFIG_RESOURCE_DIMENSIONS={}\nCONFIG_TARGET={}\nCONFIG_HIGHS_DIR={}\nCONFIG_GUROBI_HOME={}\n",
            if self.release { "release" } else { "dev" },
            yn(self.clp),
            yn(self.highs),
            yn(self.gurobi),
            yn(self.cli),
            yn(self.multi_time_windows),
            if self.ryan_foster_master { "master" } else { "pricing" },
            if self.accumulated_resource_master { "master" } else { "pricing" },
            self.simd.name(),
            yn(self.native_cpu),
            yn(self.release_debug),
            yn(self.offline),
            yn(self.locked),
            self.jobs,
            self.resource_dimensions,
            quoted(&self.target),
            quoted(&self.highs_dir),
            quoted(&self.gurobi_home)
        )
    }

    fn save(&self, path: &Path) -> Result<()> {
        self.validate()?;
        // Stage next to the destination so replacement is atomic on this filesystem.
        let temporary = path.with_file_name(format!(
            ".{}.{}.tmp",
            path.file_name()
                .ok_or("config path must name a file")?
                .to_string_lossy(),
            std::process::id()
        ));
        fs::write(&temporary, self.text()).map_err(|e| e.to_string())?;
        if let Err(e) = fs::rename(&temporary, path) {
            let _ = fs::remove_file(&temporary);
            return Err(e.to_string());
        }
        Ok(())
    }

    fn cargo(&self, action: &str) -> Result<Invocation> {
        self.validate()?;
        let mut args: Vec<OsString> = vec![action.into(), "-p".into(), "baldes".into()];
        if action == "clean" {
            return Ok(Invocation {
                args,
                vars: BTreeMap::new(),
            });
        }
        args.push("--lib".into());
        if self.cli {
            args.extend(["--bin".into(), "baldes".into()]);
        }
        if action == "test" || action == "clippy" {
            args.push("--tests".into());
        }
        args.extend([
            "--profile".into(),
            if self.release {
                "release".into()
            } else {
                "dev".into()
            },
            "--no-default-features".into(),
        ]);
        let features = [
            (self.clp, "clp"),
            (self.highs, "highs"),
            (self.gurobi, "gurobi"),
            (self.multi_time_windows, "multi-time-windows"),
            (self.ryan_foster_master, "ryan-foster-master"),
            (self.accumulated_resource_master, "accumulated-resource-master"),
            (self.simd == SimdMode::Avx2, "simd"),
            (self.simd == SimdMode::Avx512, "simd-avx512"),
        ]
        .into_iter()
        .filter_map(|(enabled, name)| enabled.then_some(name))
        .collect::<Vec<_>>()
        .join(",");
        if !features.is_empty() {
            args.extend(["--features".into(), features.into()]);
        }
        if self.offline {
            args.push("--offline".into());
        }
        if self.locked {
            args.push("--locked".into());
        }
        if self.jobs > 0 {
            args.extend(["--jobs".into(), self.jobs.to_string().into()]);
        }
        if !self.target.is_empty() {
            args.extend(["--target".into(), self.target.clone().into()]);
        }
        let mut vars = BTreeMap::new();
        vars.insert(
            "BALDES_RESOURCE_DIMENSIONS".into(),
            self.resource_dimensions.to_string().into(),
        );
        if self.release_debug {
            vars.insert("CARGO_PROFILE_RELEASE_DEBUG".into(), "1".into());
        }
        if self.highs && !self.highs_dir.is_empty() {
            vars.insert("HIGHS_DIR".into(), self.highs_dir.clone().into());
        }
        if self.gurobi && !self.gurobi_home.is_empty() {
            vars.insert("GUROBI_HOME".into(), self.gurobi_home.clone().into());
        }
        if self.native_cpu {
            // Encoded flags take precedence in Cargo; extend them if supplied.
            if let Some(mut flags) = env::var_os("CARGO_ENCODED_RUSTFLAGS") {
                if !flags.is_empty() {
                    flags.push("\x1f");
                }
                flags.push("-C\x1ftarget-cpu=native");
                vars.insert("CARGO_ENCODED_RUSTFLAGS".into(), flags);
            } else {
                let mut flags = env::var_os("RUSTFLAGS").unwrap_or_default();
                if !flags.is_empty() {
                    flags.push(" ");
                }
                flags.push("-C target-cpu=native");
                vars.insert("RUSTFLAGS".into(), flags);
            }
        }
        Ok(Invocation { args, vars })
    }

    fn edit(&mut self, selected: usize, input: Option<String>) -> Result<()> {
        match selected {
            0 => self.release = !self.release,
            1 => {
                self.clp = !self.clp;
                if !self.clp {
                    self.cli = false;
                }
            }
            2 => self.highs = !self.highs,
            3 => {
                self.cli = !self.cli;
                if self.cli {
                    self.clp = true;
                }
            }
            4 => {
                if !self.native_cpu && !self.target.is_empty() {
                    return Err(
                        "Clear Target triple before enabling native CPU optimization".into(),
                    );
                }
                self.native_cpu = !self.native_cpu;
            }
            5 => self.release_debug = !self.release_debug,
            6 => self.offline = !self.offline,
            7 => self.locked = !self.locked,
            8 => {
                if let Some(value) = input {
                    self.jobs = value
                        .parse()
                        .map_err(|_| "Jobs must be a nonnegative integer")?;
                }
            }
            9 => {
                if let Some(value) = input {
                    let mut next = self.clone();
                    next.target = value;
                    next.validate()?;
                    *self = next;
                }
            }
            10 => {
                if let Some(value) = input {
                    let mut next = self.clone();
                    next.highs_dir = value;
                    next.validate()?;
                    *self = next;
                }
            }
            11 => {
                if let Some(value) = input {
                    self.resource_dimensions = value
                        .parse()
                        .map_err(|_| "Resource dimensions must be a nonnegative usize")?;
                }
            }
            13 => self.gurobi = !self.gurobi,
            14 => {
                if let Some(value) = input {
                    let mut next = self.clone();
                    next.gurobi_home = value;
                    next.validate()?;
                    *self = next;
                }
            }
            15 => self.multi_time_windows = !self.multi_time_windows,
            16 => self.ryan_foster_master = !self.ryan_foster_master,
            17 => self.accumulated_resource_master = !self.accumulated_resource_master,
            12 => self.simd = self.simd.next(CpuSimd::detect(), self.target.is_empty()),
            _ => {}
        }
        Ok(())
    }

    fn rows(&self) -> Vec<Item> {
        vec![
            Item::value(0, "BUILD", "Build profile", if self.release { "Release" } else { "Development" }, "Release optimizes solver performance. Development compiles faster and keeps debug information."),
            Item::toggle(3, "BUILD", "Command-line program", self.cli, "Build the baldes executable alongside the library. Enabling the CLI selects Clp; disabling Clp also disables the CLI."),
            Item::toggle(1, "BACKEND", "Clp backend", self.clp, "LP solver and routing model store. Builds the Clp and CoinUtils submodules; requires a C++14 compiler."),
            Item::toggle(2, "BACKEND", "HiGHS backend", self.highs, "LP and mixed-integer solver. Requires libhighs with 32-bit HighsInt, discovered through the installation path, pkg-config or system libraries."),
            Item::toggle(13, "BACKEND", "Gurobi backend", self.gurobi, "LP master solver. Requires a Gurobi installation and valid license. Set Gurobi installation below or export GUROBI_HOME. Routing CLI also requires Clp."),
            Item::toggle(15, "BUILD", "Multi-time windows", self.multi_time_windows, "Disjoint service-start domains for scalar pricing resources."),
            Item::value(16, "BUILD", "Ryan–Foster enforcement", if self.ryan_foster_master { "Master" } else { "Pricing" }, "Fixed at compile time. Pricing keeps branch-violating columns out of labeling. Master enforces branches only through master rows; pricing ignores them, so bounds at branched nodes can be weaker."),
            Item::value(17, "BUILD", "Accumulated-resource branching", if self.accumulated_resource_master { "Master" } else { "Pricing" }, "Fixed at compile time. Pricing tightens the branched arc windows during labeling. Master enforces branches only through master rows; pricing keeps the original windows, so bounds at branched nodes can be weaker."),
            Item::value(12, "PERFORMANCE", "SIMD acceleration", self.simd.name(), "Cycle Off / AVX2 / AVX-512, skipping modes unsupported by this CPU. Host target only. AVX-512 requires Rust 1.89+. Runtime detection keeps portable fallback."),
            Item::toggle(4, "PERFORMANCE", "Native CPU", self.native_cpu, "Optimize for this computer with target-cpu=native. Clear the target triple before enabling; use portable builds for distribution."),
            Item::toggle(5, "PERFORMANCE", "Release debug symbols", self.release_debug, "Keep debug symbols in release builds for perf and other profilers. The development profile retains Cargo's normal debug settings."),
            Item::value(8, "PERFORMANCE", "Parallel jobs", if self.jobs == 0 { "Automatic".into() } else { self.jobs.to_string() }, "Maximum parallel Cargo compilation jobs. Enter 0 to let Cargo choose automatically."),
            Item::value(11, "TOOLCHAIN", "Resource dimensions", self.resource_dimensions.to_string(), "Compiled scalar capacity for Python and default Rust types. Explicit const-generic Rust capacities remain independent. Default: 2."),
            Item::value(9, "TOOLCHAIN", "Target triple", if self.target.is_empty() { "Host" } else { &self.target }, "Leave empty to build for this host. Cross compilation requires the Rust target, a linker and compatible native solver libraries."),
            Item::value(10, "TOOLCHAIN", "HiGHS installation", if self.highs_dir.is_empty() { "Automatic" } else { &self.highs_dir }, "Optional installation root containing lib/libhighs. Leave empty to preserve HIGHS_DIR and normal library discovery."),
            Item::value(14, "TOOLCHAIN", "Gurobi installation", if self.gurobi_home.is_empty() { "From GUROBI_HOME" } else { &self.gurobi_home }, "Gurobi installation root containing lib/libgurobi. Leave empty to preserve GUROBI_HOME from your shell. GUROBI_LIB_NAME may select a library version."),
            Item::toggle(6, "CARGO", "Offline builds", self.offline, "Use Cargo's offline mode. All dependencies must already be available in the local cache."),
            Item::toggle(7, "CARGO", "Locked dependencies", self.locked, "Require Cargo.lock to stay unchanged. Recommended for reproducible builds."),
        ]
    }
}

fn quoted(text: &str) -> String {
    format!("\"{}\"", text.replace('\\', "\\\\").replace('"', "\\\""))
}
fn parse_string(text: &str) -> Result<String> {
    let text = text
        .strip_prefix('"')
        .and_then(|s| s.strip_suffix('"'))
        .ok_or("string values must be double-quoted")?;
    let mut result = String::new();
    let mut chars = text.chars();
    while let Some(c) = chars.next() {
        match c {
            '\\' => match chars.next() {
                Some(c @ ('\\' | '"')) => result.push(c),
                _ => return Err("only escaped quotes and backslashes are supported".into()),
            },
            '"' => return Err("unescaped quote in string".into()),
            c if c.is_control() => return Err("control character in string".into()),
            _ => result.push(c),
        }
    }
    Ok(result)
}

#[derive(Debug)]
struct Invocation {
    args: Vec<OsString>,
    vars: BTreeMap<OsString, OsString>,
}
impl Invocation {
    fn print(&self, cargo: &OsStr) {
        for (key, value) in &self.vars {
            print!("{}={} ", key.to_string_lossy(), shell_quote(value));
        }
        print!("{}", shell_quote(cargo));
        for arg in &self.args {
            print!(" {}", shell_quote(arg));
        }
        println!();
    }
    fn run(&self, cargo: &OsStr) -> Result<i32> {
        self.print(cargo);
        let status = Command::new(cargo)
            .args(&self.args)
            .envs(&self.vars)
            .status()
            .map_err(|e| format!("cannot run {}: {e}", cargo.to_string_lossy()))?;
        Ok(status.code().unwrap_or(1))
    }
}
fn shell_quote(s: &OsStr) -> String {
    let s = s.to_string_lossy();
    if !s.is_empty()
        && s.chars()
            .all(|c| c.is_ascii_alphanumeric() || "_./,:=-".contains(c))
    {
        s.into_owned()
    } else {
        format!("'{}'", s.replace('\'', "'\"'\"'"))
    }
}

// The terminal menu uses ANSI drawing and POSIX stty. No curses dependency,
// unsafe termios binding, desktop session, or Cargo package download is needed.
struct Terminal {
    saved: String,
    color: bool,
}
impl Terminal {
    fn enter() -> Result<Self> {
        if !io::stdin().is_terminal() || !io::stdout().is_terminal() {
            return Err("make config needs an interactive terminal; use make defconfig and edit .config for automation".into());
        }
        if matches!(env::var("TERM").as_deref(), Ok("dumb") | Err(_)) {
            return Err("a terminal with ANSI support is required (TERM is unset or dumb)".into());
        }
        let saved = stty(&["-g"])?;
        let terminal = Self {
            saved: saved.trim().into(),
            color: env::var_os("NO_COLOR").is_none(),
        };
        stty(&["raw", "-echo", "min", "0", "time", "1"])?;
        print!("\x1b[?1049h\x1b[?25l");
        io::stdout().flush().map_err(|e| e.to_string())?;
        Ok(terminal)
    }
    fn size(&self) -> (usize, usize) {
        let size = stty(&["size"]).unwrap_or_default();
        let mut dims = size.split_whitespace();
        (
            dims.next().and_then(|s| s.parse().ok()).unwrap_or(24),
            dims.next().and_then(|s| s.parse().ok()).unwrap_or(80),
        )
    }
}
impl Drop for Terminal {
    fn drop(&mut self) {
        let _ = stty(&[&self.saved]);
        print!("\x1b[0m\x1b[?25h\x1b[?1049l");
        let _ = io::stdout().flush();
    }
}
fn stty(args: &[&str]) -> Result<String> {
    let output = Command::new("stty")
        .args(args)
        .stdin(Stdio::inherit())
        .output()
        .map_err(|e| format!("cannot run stty: {e}"))?;
    if !output.status.success() {
        return Err(format!(
            "stty: {}",
            String::from_utf8_lossy(&output.stderr).trim()
        ));
    }
    Ok(String::from_utf8_lossy(&output.stdout).into_owned())
}

#[derive(Debug, PartialEq, Eq)]
enum Key {
    Byte(u8),
    Up,
    Down,
    Left,
    Right,
    Home,
    End,
    PageUp,
    PageDown,
    Escape,
    Idle,
}
fn byte() -> io::Result<Option<u8>> {
    let mut b = [0];
    match io::stdin().read(&mut b) {
        Ok(0) => Ok(None),
        Ok(_) => Ok(Some(b[0])),
        Err(e) if e.kind() == io::ErrorKind::Interrupted => Ok(None),
        Err(e) => Err(e),
    }
}
fn key() -> Result<Key> {
    let read = || byte().map_err(|e| e.to_string());
    Ok(match read()? {
        None => Key::Idle,
        Some(27) => match read()? {
            Some(b'[' | b'O') => match read()? {
                Some(b'A') => Key::Up,
                Some(b'B') => Key::Down,
                Some(b'C') => Key::Right,
                Some(b'D') => Key::Left,
                Some(b'H') => Key::Home,
                Some(b'F') => Key::End,
                Some(code @ (b'1' | b'4' | b'5' | b'6' | b'7' | b'8')) => {
                    if read()? != Some(b'~') {
                        Key::Idle
                    } else {
                        match code {
                            b'1' | b'7' => Key::Home,
                            b'4' | b'8' => Key::End,
                            b'5' => Key::PageUp,
                            _ => Key::PageDown,
                        }
                    }
                }
                _ => Key::Idle,
            },
            _ => Key::Escape,
        },
        Some(b) => Key::Byte(b),
    })
}

struct Screen {
    size: (usize, usize),
    checked: Instant,
    dirty: bool,
}
impl Screen {
    fn new() -> Self {
        Self {
            size: (0, 0),
            checked: Instant::now(),
            dirty: true,
        }
    }
    fn paint(&mut self, terminal: &Terminal, view: &View<'_>, state: &mut State) -> Result<()> {
        // stty is portable and dependency-free; poll only a few times per
        // second, and redraw only after input or an actual terminal resize.
        if self.dirty || self.checked.elapsed() >= Duration::from_millis(250) {
            let size = terminal.size();
            if self.dirty || size != self.size {
                print!(
                    "{}",
                    build_config_ui::render(size.1, size.0, view, state, terminal.color)
                );
                io::stdout().flush().map_err(|e| e.to_string())?;
            }
            self.size = size;
            self.checked = Instant::now();
            self.dirty = false;
        }
        Ok(())
    }
}

fn input(
    terminal: &Terminal,
    view: &View<'_>,
    state: &mut State,
    prompt: &str,
    current: &str,
) -> Result<Option<String>> {
    let mut value = current.to_string();
    let mut screen = Screen::new();
    loop {
        let editing = View {
            editing: Some((prompt, &value)),
            ..*view
        };
        screen.paint(terminal, &editing, state)?;
        let pressed = key()?;
        if pressed == Key::Idle {
            continue;
        }
        screen.dirty = true;
        match pressed {
            Key::Byte(b'\r' | b'\n') => return Ok(Some(value)),
            Key::Escape | Key::Byte(3) => return Ok(None),
            Key::Byte(127 | 8) => {
                value.pop();
            }
            Key::Byte(21) => value.clear(), // Ctrl-U
            Key::Byte(b @ 32..=126) => value.push(b as char),
            _ => {}
        }
    }
}

fn menu(config: &mut Config, path: &Path) -> Result<()> {
    let terminal = Terminal::enter()?;
    let original = config.clone();
    let mut state = State::default();
    let mut screen = Screen::new();
    let mut message = String::new();
    loop {
        let view = View {
            config,
            original: &original,
            path,
            message: &message,
            editing: None,
        };
        screen.paint(&terminal, &view, &mut state)?;
        let pressed = key()?;
        if pressed == Key::Idle {
            continue;
        }
        screen.dirty = true;
        let rows = config.rows();
        match pressed {
            Key::Escape | Key::Byte(b'q' | b'Q' | 3) => {
                drop(terminal);
                println!("Configuration unchanged.");
                return Ok(());
            }
            Key::Byte(b's' | b'S') => match config.save(path) {
                Ok(()) => {
                    drop(terminal);
                    println!("Saved {}. Run make to compile.", path.display());
                    return Ok(());
                }
                Err(e) => message = e,
            },
            Key::Up | Key::Byte(b'k' | b'K') => {
                state.selected = (state.selected + rows.len() - 1) % rows.len();
                message.clear();
            }
            Key::Down | Key::Byte(b'j' | b'J' | b'\t') => {
                state.selected = (state.selected + 1) % rows.len();
                message.clear();
            }
            Key::Home => state.selected = 0,
            Key::End => state.selected = rows.len() - 1,
            Key::PageUp => state.selected = state.selected.saturating_sub(5),
            Key::PageDown => state.selected = (state.selected + 5).min(rows.len() - 1),
            Key::Left | Key::Right | Key::Byte(b' ' | b'\r' | b'\n') => {
                let id = rows[state.selected].id;
                let value = match id {
                    8 => input(
                        &terminal,
                        &view,
                        &mut state,
                        "Parallel jobs (0 = automatic)",
                        &config.jobs.to_string(),
                    )?,
                    9 => input(
                        &terminal,
                        &view,
                        &mut state,
                        "Target triple (empty = host)",
                        &config.target,
                    )?,
                    10 => input(
                        &terminal,
                        &view,
                        &mut state,
                        "HiGHS root (empty = automatic)",
                        &config.highs_dir,
                    )?,
                    14 => input(
                        &terminal,
                        &view,
                        &mut state,
                        "Gurobi root (empty = shell GUROBI_HOME)",
                        &config.gurobi_home,
                    )?,
                    11 => input(
                        &terminal,
                        &view,
                        &mut state,
                        "Resource dimensions (default = 2)",
                        &config.resource_dimensions.to_string(),
                    )?,
                    _ => None,
                };
                message = match config.edit(id, value) {
                    Ok(()) => String::new(),
                    Err(e) => e,
                };
            }
            _ => {}
        }
    }
}

fn run() -> Result<i32> {
    let mut args = env::args_os().skip(1);
    let mut path = PathBuf::from(".config");
    let mut cargo = OsString::from("cargo");
    let mut action = None;
    while let Some(arg) = args.next() {
        match arg.to_str() {
            Some("--config") => path = PathBuf::from(args.next().ok_or("--config needs a path")?),
            Some("--cargo") => cargo = args.next().ok_or("--cargo needs an executable")?,
            _ if action.is_none() => {
                action = Some(arg.to_str().ok_or("invalid action")?.to_string())
            }
            _ => return Err("unexpected argument".into()),
        }
    }
    let action = action
        .ok_or("expected menu, defaults, show, command, build, check, test, clippy or clean")?;
    if action == "defaults" {
        Config::default().save(&path)?;
        println!("Saved defaults to {}", path.display());
        return Ok(0);
    }
    let mut config = Config::load(&path)?;
    match action.as_str() {
        "menu" => menu(&mut config, &path)?,
        "show" => print!("{}", config.text()),
        "command" => config.cargo("build")?.print(&cargo),
        "build" | "check" | "test" | "clippy" | "clean" => {
            if !path.exists() {
                println!(
                    "No {}: using library-only release defaults. Run make config to customize.",
                    path.display()
                );
            }
            if config.clp
                && action != "clean"
                && (!Path::new("vendor/Clp/Clp/src").is_dir()
                    || !Path::new("vendor/CoinUtils/CoinUtils/src").is_dir())
            {
                return Err("Clp submodules missing; run git submodule update --init vendor/Clp vendor/CoinUtils".into());
            }
            return config.cargo(&action)?.run(&cargo);
        }
        _ => return Err(format!("unknown action {action}")),
    }
    Ok(0)
}
fn main() {
    let code = match run() {
        Ok(code) => code,
        Err(e) => {
            eprintln!("build config: {e}");
            1
        }
    };
    std::process::exit(code);
}

#[cfg(test)]
mod tests {
    use super::*;
    fn args(c: &Config, action: &str) -> Vec<String> {
        c.cargo(action)
            .unwrap()
            .args
            .iter()
            .map(|x| x.to_string_lossy().into_owned())
            .collect()
    }

    #[test]
    fn simd_modes_follow_cpu_support() {
        for (cpu, expected) in [
            (
                CpuSimd {
                    avx2: false,
                    avx512: false,
                },
                vec![SimdMode::Off],
            ),
            (
                CpuSimd {
                    avx2: true,
                    avx512: false,
                },
                vec![SimdMode::Off, SimdMode::Avx2],
            ),
            (
                CpuSimd {
                    avx2: true,
                    avx512: true,
                },
                vec![SimdMode::Off, SimdMode::Avx2, SimdMode::Avx512],
            ),
        ] {
            let mut mode = SimdMode::Off;
            for &next in expected.iter().cycle().skip(1).take(expected.len()) {
                mode = mode.next(cpu, true);
                assert_eq!(mode, next);
            }
            for mode in [SimdMode::Off, SimdMode::Avx2, SimdMode::Avx512] {
                let c = Config {
                    simd: mode,
                    ..Config::default()
                };
                assert_eq!(c.validate_simd(cpu).is_ok(), cpu.supports(mode));
                assert_eq!(mode.next(cpu, false), SimdMode::Off);
            }
        }
    }

    #[test]
    fn optional_simd_round_trips_and_reaches_cargo() {
        for (mode, feature) in [(SimdMode::Avx2, "simd"), (SimdMode::Avx512, "simd-avx512")] {
            let mut c = Config {
                simd: mode,
                ..Config::default()
            };
            if !CpuSimd::detect().supports(mode) {
                assert!(Config::parse(&c.text()).is_err());
                assert!(c.cargo("build").is_err());
                continue;
            }
            assert_eq!(Config::parse(&c.text()).unwrap(), c);
            assert!(args(&c, "build")
                .windows(2)
                .any(|w| w == ["--features", feature]));
            c.target = "aarch64-unknown-linux-gnu".into();
            assert!(c.validate().is_err());
        }
        assert_eq!(Config::parse("CONFIG_SIMD=n").unwrap().simd, SimdMode::Off);
        if CpuSimd::detect().avx2 {
            assert_eq!(Config::parse("CONFIG_SIMD=y").unwrap().simd, SimdMode::Avx2);
        }
        assert!(Config::parse("CONFIG_SIMD=unknown").is_err());
    }

    #[test]
    fn defaults_build_library_without_optional_backends() {
        let c = Config::default();
        assert_eq!(Config::parse(&c.text()).unwrap(), c);
        assert_eq!(
            args(&c, "build"),
            [
                "build",
                "-p",
                "baldes",
                "--lib",
                "--profile",
                "release",
                "--no-default-features",
                "--locked"
            ]
        );
    }
    #[test]
    fn cli_dependency_and_menu_transitions() {
        assert!(Config::parse("CONFIG_CLI=y").is_err());
        let mut c = Config::default();
        c.edit(3, None).unwrap();
        assert!(c.cli && c.clp);
        c.edit(1, None).unwrap();
        assert!(!c.cli && !c.clp);
    }
    #[test]
    fn settings_become_cargo_arguments_and_environment() {
        let c = Config {
            clp: true,
            highs: true,
            cli: true,
            release: false,
            offline: true,
            locked: false,
            jobs: 3,
            resource_dimensions: 5,
            target: "aarch64-unknown-linux-gnu".into(),
            highs_dir: "/opt/HiGHS build".into(),
            release_debug: true,
            ..Config::default()
        };
        assert_eq!(Config::parse(&c.text()).unwrap(), c);
        let a = args(&c, "test");
        assert!(a.windows(2).any(|w| w == ["--features", "clp,highs"]));
        assert!(a.windows(2).any(|w| w == ["--profile", "dev"]));
        assert!(a.contains(&"--offline".into()));
        assert!(!a.contains(&"--locked".into()));
        assert!(a.windows(2).any(|w| w == ["--jobs", "3"]));
        assert!(a.windows(2).any(|w| w == ["--bin", "baldes"]));
        assert!(a
            .windows(2)
            .any(|w| w == ["--target", "aarch64-unknown-linux-gnu"]));
        assert_eq!(
            c.cargo("build").unwrap().vars.get(OsStr::new("HIGHS_DIR")),
            Some(&OsString::from("/opt/HiGHS build"))
        );
    }
    #[test]
    fn gurobi_menu_settings_reach_cargo_and_preserve_shell_defaults() {
        // Existing configurations need no migration.
        let mut c = Config::parse("CONFIG_CLP=y\nCONFIG_HIGHS=y").unwrap();
        assert!(!c.gurobi);
        c.edit(13, None).unwrap();
        c.edit(14, Some("/opt/Gurobi build/$(literal)".into()))
            .unwrap();
        assert_eq!(Config::parse(&c.text()).unwrap(), c);
        for action in ["build", "check", "test", "clippy"] {
            assert!(args(&c, action)
                .windows(2)
                .any(|w| w == ["--features", "clp,highs,gurobi"]));
            assert_eq!(
                c.cargo(action).unwrap().vars.get(OsStr::new("GUROBI_HOME")),
                Some(&OsString::from(&c.gurobi_home))
            );
        }
        c.edit(13, None).unwrap();
        assert!(!c
            .cargo("build")
            .unwrap()
            .vars
            .contains_key(OsStr::new("GUROBI_HOME")));
        c.edit(13, None).unwrap();
        c.edit(14, Some(String::new())).unwrap();
        assert!(!c
            .cargo("build")
            .unwrap()
            .vars
            .contains_key(OsStr::new("GUROBI_HOME")));
        assert!(c.cargo("clean").unwrap().vars.is_empty());
        assert!(Config::parse("CONFIG_GUROBI=maybe").is_err());
        assert!(c.edit(14, Some("bad\npath".into())).is_err());
        assert!(c.gurobi_home.is_empty());
        // The standalone Gurobi library does not require Clp.
        assert!(args(&Config::parse("CONFIG_GUROBI=y").unwrap(), "build")
            .windows(2)
            .any(|w| w == ["--features", "gurobi"]));
    }

    #[test]
    fn malformed_settings_are_rejected_instead_of_executed() {
        for text in [
            "CONFIG_CLP=maybe",
            "CONFIG_PROFILE=fast",
            "CONFIG_JOBS=-1",
            "CONFIG_TARGET=bad",
            "CONFIG_MISSING=y",
            "CONFIG_CLP=y\nCONFIG_CLP=n",
            "CONFIG_TARGET=\"$(touch file)\"",
            "CONFIG_NATIVE_CPU=y\nCONFIG_TARGET=\"aarch64-unknown-linux-gnu\"",
        ] {
            assert!(Config::parse(text).is_err(), "{text}");
        }
        let c = Config::parse("CONFIG_HIGHS_DIR=\"/opt/$(literal)/HiGHS\"").unwrap();
        assert_eq!(c.highs_dir, "/opt/$(literal)/HiGHS");
    }
    #[test]
    fn quoted_paths_round_trip() {
        let c = Config {
            highs_dir: "/opt/HiGHS \\\" build".into(),
            ..Config::default()
        };
        assert_eq!(Config::parse(&c.text()).unwrap(), c);
        assert!(parse_string("\"bad\\q\"").is_err());
    }
    #[test]
    fn cross_compilation_conflicts_with_native_cpu() {
        let mut c = Config {
            native_cpu: true,
            ..Config::default()
        };
        assert!(c.edit(9, Some("aarch64-unknown-linux-gnu".into())).is_err());
        assert!(c.target.is_empty());
        c.native_cpu = false;
        c.edit(9, Some("aarch64-unknown-linux-gnu".into())).unwrap();
        assert!(c.edit(4, None).is_err());
    }
    #[test]
    fn resource_capacity_round_trips_and_reaches_every_build_action() {
        let mut c = Config::default();
        assert_eq!(c.resource_dimensions, 2);
        c.edit(11, Some("5".into())).unwrap();
        assert_eq!(Config::parse(&c.text()).unwrap(), c);
        for action in ["build", "check", "test", "clippy"] {
            assert_eq!(
                c.cargo(action)
                    .unwrap()
                    .vars
                    .get(OsStr::new("BALDES_RESOURCE_DIMENSIONS")),
                Some(&OsString::from("5"))
            );
        }
        assert!(c.edit(11, Some("-1".into())).is_err());
        assert_eq!(c.resource_dimensions, 5);
        assert!(Config::parse("CONFIG_RESOURCE_DIMENSIONS=arbitrary").is_err());
        assert!(Config::parse("CONFIG_RESOURCE_DIMENSIONS=184467440737095516160").is_err());
        c.edit(11, Some("0".into())).unwrap();
        assert_eq!(c.resource_dimensions, 0);
        assert!(c.cargo("clean").unwrap().vars.is_empty());
        assert_eq!(c.rows().len(), 18);
    }
    #[test]
    fn ryan_foster_enforcement_round_trips_and_selects_its_feature() {
        let mut c = Config::default();
        assert!(!args(&c, "build").iter().any(|a| a.contains("ryan-foster-master")));
        c.edit(16, None).unwrap();
        assert!(c.ryan_foster_master);
        assert!(c.text().contains("CONFIG_RYAN_FOSTER=master"));
        assert_eq!(Config::parse(&c.text()).unwrap(), c);
        assert!(args(&c, "build").iter().any(|a| a.contains("ryan-foster-master")));
        assert!(Config::parse("CONFIG_RYAN_FOSTER=labels").is_err());
        assert!(!Config::parse("CONFIG_RYAN_FOSTER=pricing").unwrap().ryan_foster_master);
    }
    #[test]
    fn accumulated_resource_enforcement_round_trips_and_selects_its_feature() {
        let mut c = Config::default();
        assert!(!args(&c, "build").iter().any(|a| a.contains("accumulated-resource-master")));
        c.edit(17, None).unwrap();
        assert!(c.accumulated_resource_master);
        assert!(c.text().contains("CONFIG_ACCUMULATED_RESOURCE=master"));
        assert_eq!(Config::parse(&c.text()).unwrap(), c);
        assert!(args(&c, "build").iter().any(|a| a.contains("accumulated-resource-master")));
        assert!(Config::parse("CONFIG_ACCUMULATED_RESOURCE=labels").is_err());
        assert!(!Config::parse("CONFIG_ACCUMULATED_RESOURCE=pricing").unwrap().accumulated_resource_master);
    }
    #[test]
    fn clean_does_not_receive_build_flags() {
        assert_eq!(args(&Config::default(), "clean"), ["clean", "-p", "baldes"]);
    }
    #[test]
    fn save_load_and_missing_file_defaults() {
        let path = env::temp_dir().join(format!("baldes-build-config-test-{}", std::process::id()));
        assert_eq!(Config::load(&path).unwrap(), Config::default());
        let c = Config {
            jobs: 2,
            ..Config::default()
        };
        c.save(&path).unwrap();
        assert_eq!(Config::load(&path).unwrap(), c);
        fs::remove_file(path).unwrap();
    }
}
