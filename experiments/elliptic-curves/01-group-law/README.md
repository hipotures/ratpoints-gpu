# 01 — Group law

We work with

```text
y^2 = x^3 - 2
```

and the rational point `P=(3,5)`.

For a short Weierstrass curve `y^2=x^3+Ax+B`, distinct finite points `P=(x1,y1)` and `Q=(x2,y2)` use

```text
m  = (y2-y1)/(x2-x1)
x3 = m^2-x1-x2
y3 = m(x1-x3)-y1.
```

For doubling,

```text
m = (3*x1^2+A)/(2*y1).
```

The point at infinity `O` is the identity and `(x,-y)` is the inverse of `(x,y)`.

The script performs all coordinates exactly with rational numbers and demonstrates double-and-add scalar multiplication. It also checks associativity on a finite sample of multiples of `P`. That bounded check validates the implementation on those examples; it is not a proof of associativity in general.
