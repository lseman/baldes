mod ce_cli;
mod cli;
mod hgs_cli;
mod pstep_cli;

fn main() {
    if let Err(error) = cli::run() {
        eprintln!("{error}");
        std::process::exit(1);
    }
}
