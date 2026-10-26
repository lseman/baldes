mod ce_cli;
mod cli;
mod pstep_cli;

fn main() {
    if let Err(error) = cli::run() {
        eprintln!("{error}");
        std::process::exit(1);
    }
}
