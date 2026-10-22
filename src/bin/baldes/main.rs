mod ce_cli;
mod multi_pricing_cli;
mod pstep_cli;
mod routing;

fn main() {
    if let Err(error) = routing::run() {
        eprintln!("{error}");
        std::process::exit(1);
    }
}
