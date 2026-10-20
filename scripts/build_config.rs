//! Dependency-free Make frontend: saved configuration, terminal menu and Cargo invocation.
//! Compiled directly by rustc so configuring does not build BALDES or its LP backends.
use std::collections::{BTreeMap, BTreeSet};
use std::env;
use std::ffi::{OsStr, OsString};
use std::fs;
use std::io::{self, IsTerminal, Read, Write};
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};

type Result<T> = std::result::Result<T, String>;

#[derive(Clone, Debug, PartialEq, Eq)]
struct Config {
    clp: bool,
    highs: bool,
    cli: bool,
    release: bool,
    native_cpu: bool,
    simd: bool,
    release_debug: bool,
    offline: bool,
    locked: bool,
    jobs: usize,
    resource_dimensions: usize,
    target: String,
    highs_dir: String,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            clp: false,
            highs: false,
            cli: false,
            release: true,
            native_cpu: false,
            simd: false,
            release_debug: false,
            offline: false,
            locked: true,
            jobs: 0,
            resource_dimensions: 2,
            target: String::new(),
            highs_dir: String::new(),
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
                "CONFIG_CLI" => config.cli = boolean()?,
                "CONFIG_SIMD" => config.simd = boolean()?,
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
                _ => return Err(format!("unknown setting {key}")),
            }
        }
        config.validate()?;
        Ok(config)
    }

    fn validate(&self) -> Result<()> {
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
        if self.highs_dir.chars().any(char::is_control) {
            return Err("HiGHS directory cannot contain control characters".into());
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
            CONFIG_PROFILE={}\nCONFIG_CLP={}\nCONFIG_HIGHS={}\nCONFIG_CLI={}\n\
            CONFIG_SIMD={}\nCONFIG_NATIVE_CPU={}\nCONFIG_RELEASE_DEBUG={}\nCONFIG_OFFLINE={}\nCONFIG_LOCKED={}\n\
            CONFIG_JOBS={}\nCONFIG_RESOURCE_DIMENSIONS={}\nCONFIG_TARGET={}\nCONFIG_HIGHS_DIR={}\n",
            if self.release { "release" } else { "dev" },
            yn(self.clp),
            yn(self.highs),
            yn(self.cli),
            yn(self.simd),
            yn(self.native_cpu),
            yn(self.release_debug),
            yn(self.offline),
            yn(self.locked),
            self.jobs,
            self.resource_dimensions,
            quoted(&self.target),
            quoted(&self.highs_dir)
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
            (self.simd, "simd"),
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
            12 => self.simd = !self.simd,
            _ => {}
        }
        Ok(())
    }

    fn rows(&self) -> Vec<(String, &'static str)> {
        let check = |b| if b { "[*]" } else { "[ ]" };
        vec![
            (format!("Build profile              <{}>",if self.release {"release"}else{"dev"}),"Release enables optimizations; dev builds faster for development."),
            (format!("{} Clp LP backend",check(self.clp)),"Builds vendor/Clp and vendor/CoinUtils. Needs their submodules and a C++14 compiler."),
            (format!("{} HiGHS LP / MIP backend",check(self.highs)),"Needs libhighs (32-bit HighsInt): HIGHS_DIR/lib, vendor/highs/lib, pkg-config or system."),
            (format!("{} baldes command-line program",check(self.cli)),"Selects Clp automatically. Disabling Clp also disables the CLI."),
            (format!("{} Optimize for this CPU",check(self.native_cpu)),"Adds -C target-cpu=native. Use for local builds; disabled for cross compilation."),
            (format!("{} Debug symbols in release",check(self.release_debug)),"Adds release debug symbols for profiling. Dev profile keeps Cargo's debug defaults."),
            (format!("{} Offline Cargo builds",check(self.offline)),"Passes --offline. All required dependencies must already be cached."),
            (format!("{} Lock dependency versions",check(self.locked)),"Passes --locked so Cargo cannot modify Cargo.lock."),
            (format!("Parallel jobs              <{}>",if self.jobs==0 {"automatic".into()}else{self.jobs.to_string()}),"Cargo compilation jobs; 0 uses Cargo's automatic setting."),
            (format!("Target triple              <{}>",if self.target.is_empty(){"host"}else{&self.target}),"Empty means host. Otherwise install that Rust target and its linker first."),
            (format!("HiGHS installation         <{}>",if self.highs_dir.is_empty(){"auto / environment"}else{&self.highs_dir}),"Optional installation root holding lib/libhighs. Empty preserves HIGHS_DIR from your shell."),
            (format!("Scalar resource dimensions <{}>", self.resource_dimensions),"Compiled scalar capacity for C/Python and Rust defaults; explicit Rust const counts remain arbitrary."),
            (format!("{} Optional AVX2 dominance scans", check(self.simd)), "Runtime CPU detection; portable fallback on CPUs without AVX2. Default off."),
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
        print!("\x1b[?25h\x1b[?1049l");
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
                _ => Key::Idle,
            },
            _ => Key::Escape,
        },
        Some(b) => Key::Byte(b),
    })
}

fn next_key() -> Result<Key> {
    loop {
        let pressed = key()?;
        if pressed != Key::Idle {
            return Ok(pressed);
        }
    }
}

fn line(row: usize, text: &str, width: usize, highlight: bool) {
    // Config fields never become terminal control sequences.
    let safe: String = text
        .chars()
        .filter(|c| !c.is_control())
        .take(width.saturating_sub(4))
        .collect();
    print!(
        "\x1b[{};3H\x1b[{}m{:<fill$}\x1b[0m",
        row,
        if highlight { "7" } else { "0" },
        safe,
        fill = width.saturating_sub(4)
    );
}
fn draw(terminal: &Terminal, config: &Config, selected: usize, message: &str) -> Result<()> {
    let (height, width) = terminal.size();
    print!("\x1b[H\x1b[2J");
    if height < 23 || width < 65 {
        line(
            1,
            "Resize terminal to at least 65 columns x 23 rows.",
            width,
            false,
        );
        line(2, "Q cancels without saving.", width, false);
    } else {
        let title = format!(
            "BALDES build configuration {}",
            if config.release { "(release)" } else { "(dev)" }
        );
        line(1, &title, width, true);
        line(
            3,
            "Arrow keys / J,K move | Space / Enter change",
            width,
            false,
        );
        line(4, "S saves and exits | Q / Escape cancels", width, false);
        for (i, (text, _)) in config.rows().iter().enumerate() {
            line(6 + i, text, width, i == selected);
        }
        line(19, config.rows()[selected].1, width, false);
        line(
            21,
            "Core pricing, HGS, cuts and native IPM are always included.",
            width,
            false,
        );
        line(22, message, width, false);
    }
    io::stdout().flush().map_err(|e| e.to_string())
}

fn input(terminal: &Terminal, prompt: &str, current: &str) -> Result<Option<String>> {
    let mut value = current.to_string();
    loop {
        let (height, width) = terminal.size();
        line(
            height.saturating_sub(1),
            &format!("{prompt}: {value}_"),
            width,
            true,
        );
        io::stdout().flush().map_err(|e| e.to_string())?;
        match next_key()? {
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
    let mut selected = 0;
    let mut message =
        String::from("Settings apply to subsequent make builds. Ctrl-U clears a text field.");
    loop {
        draw(&terminal, config, selected, &message)?;
        match next_key()? {
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
                selected = (selected + config.rows().len() - 1) % config.rows().len()
            }
            Key::Down | Key::Byte(b'j' | b'J' | b'\t') => {
                selected = (selected + 1) % config.rows().len()
            }
            Key::Left | Key::Right | Key::Byte(b' ' | b'\r' | b'\n') => {
                let value = match selected {
                    8 => input(&terminal, "Jobs (0 = automatic)", &config.jobs.to_string())?,
                    9 => input(&terminal, "Target triple (empty = host)", &config.target)?,
                    10 => input(&terminal, "HiGHS root (empty = auto)", &config.highs_dir)?,
                    11 => input(
                        &terminal,
                        "Scalar capacity (default = 2)",
                        &config.resource_dimensions.to_string(),
                    )?,
                    _ => None,
                };
                message = match config.edit(selected, value) {
                    Ok(()) => "Unsaved changes. Press S to save.".into(),
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
    fn optional_simd_round_trips_and_reaches_cargo() {
        let mut c = Config::default();
        assert!(!c.simd);
        c.edit(12, None).unwrap();
        assert_eq!(Config::parse(&c.text()).unwrap(), c);
        assert!(args(&c, "build")
            .windows(2)
            .any(|w| w == ["--features", "simd"]));
        c.clp = true;
        assert!(args(&c, "test")
            .windows(2)
            .any(|w| w == ["--features", "clp,simd"]));
        c.edit(12, None).unwrap();
        assert!(!c.simd);
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
        assert_eq!(c.rows().len(), 13);
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
