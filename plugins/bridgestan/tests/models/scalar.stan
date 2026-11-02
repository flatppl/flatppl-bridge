parameters {
  real<lower=0> s;
}
model {
  s ~ exponential(2);
}
