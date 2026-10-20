# R201 master regression

`r201_master.txt` is the exact LP captured immediately before the native IPM
failed with `iterate left the strict interior` on R201 (100 customers), using:

```sh
cargo build --offline --release --features clp --bin baldes
target/release/baldes vrptw instances/R201.txt --master ipm --time 120
```

The capture used revision `b2c3dfafd714596cc4d00f3404fe085038d48bee` plus
temporary LP-capture instrumentation. It has 101 rows, 1,694 columns, and 22,923
nonzeros, including artificial columns. Every variable has bounds `[0, 1e12]`.
The failing solve stalled at original-coordinate primal residual `3.0938e-9`
and homogeneous scale `7.0501e-10`, then lost strict positivity.

Format: the first line contains row count, column count, and requested gap
tolerance. The next two lines contain RHS values and row senses (`0` equality,
`1` less-than-or-equal, `2` greater-than-or-equal). Each remaining line describes
one column: cost, lower bound, upper bound, then zero-based `row:coefficient`
entries. Decimal costs round-trip to the captured f64 values.

Independent reference: SciPy 1.18.1 `linprog(method="highs")`, with primal and
dual feasibility tolerances `1e-9`, returns objective `937.6319384978376` and
zero maximum equality residual. Tests use this recorded value; SciPy is not a
test dependency. The regression checks both the captured gap tolerance `0.5`
and the default strict gap tolerance, retaining primal/dual tolerance `1e-9`.
