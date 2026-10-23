mod ce_cli;
mod mp_cli;
mod pstep_cli;
mod routing;

fn main() {
    if let Err(error) = routing::run() {
        eprintln!("{error}");
        std::process::exit(1);
    }
}
