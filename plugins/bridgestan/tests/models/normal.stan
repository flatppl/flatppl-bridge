parameters {
  real mu;
  real<lower=0> sigma;
}
model {
  mu ~ normal(1.5, 2);
  sigma ~ lognormal(0, 1);
}
