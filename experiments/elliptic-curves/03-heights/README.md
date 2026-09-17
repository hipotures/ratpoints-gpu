# 03 — Heights

Starting from `P=(3,5)` on `y^2=x^3-2`, this experiment computes `nP` exactly and displays the growth of the rational x-coordinate.

For `x=u/v` in lowest terms it uses the logarithmic naive x-height

```text
h_x(P) = log(max(|u|, v)).
```

The table includes `h_x(nP)/n^2`. Its stabilization illustrates why canonical height is quadratic under multiplication. This script is an experiment, not an exact canonical-height implementation. Point coordinates are exact; logarithms are floating-point presentation values.

Example:

```bash
python3 experiments/elliptic-curves/03-heights/experiment.py --max-multiple 80
```
