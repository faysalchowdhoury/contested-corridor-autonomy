### Table 1 - Main benchmark across contested regimes (test split)

| Method | Regime | RMSE | MAE | MAPE | PhysViolRate | CtrlStabCost |
|---|---|---|---|---|---|---|
| Dec-PINC-V2X | clean | 8.943 | 5.373 | 43.145 | 0.014 | 0.517 |
| Persistence (LOCF) | clean | 8.093 | 4.024 | 23.223 | 0.029 | 0.349 |
| Ridge | clean | 7.813 | 4.570 | 34.994 | 0.012 | 0.328 |
| GBM (LightGBM) | clean | 7.862 | 4.828 | 32.861 | 0.019 | 0.314 |
| Centralized NN (no physics) | clean | 7.428 | 4.796 | 40.831 | 0.017 | 0.325 |
| Decentralized NN (no physics) | clean | 7.590 | 4.768 | 34.661 | 0.009 | 0.454 |
| Dec-PINC-V2X | drop30 | 11.359 | 6.750 | 58.183 | 0.047 | 0.500 |
| Persistence (LOCF) | drop30 | 11.296 | 5.779 | 39.550 | 0.090 | 0.374 |
| Ridge | drop30 | 10.489 | 6.085 | 48.921 | 0.069 | 0.343 |
| GBM (LightGBM) | drop30 | 10.343 | 6.241 | 47.523 | 0.066 | 0.334 |
| Centralized NN (no physics) | drop30 | 8.633 | 5.701 | 51.476 | 0.011 | 0.292 |
| Decentralized NN (no physics) | drop30 | 9.869 | 5.978 | 46.725 | 0.045 | 0.441 |
| Dec-PINC-V2X | drop60 | 14.465 | 8.767 | 70.206 | 0.087 | 0.420 |
| Persistence (LOCF) | drop60 | 13.889 | 7.586 | 47.088 | 0.100 | 0.378 |
| Ridge | drop60 | 12.790 | 7.651 | 55.423 | 0.084 | 0.340 |
| GBM (LightGBM) | drop60 | 12.445 | 7.757 | 54.401 | 0.072 | 0.342 |
| Centralized NN (no physics) | drop60 | 10.203 | 6.846 | 54.168 | 0.019 | 0.254 |
| Decentralized NN (no physics) | drop60 | 12.064 | 7.419 | 52.829 | 0.057 | 0.432 |
| Dec-PINC-V2X | high_latency | 15.052 | 9.567 | 82.207 | 0.051 | 1.302 |
| Persistence (LOCF) | high_latency | 15.372 | 8.886 | 61.661 | 0.097 | 1.459 |
| Ridge | high_latency | 14.284 | 8.833 | 68.216 | 0.080 | 1.304 |
| GBM (LightGBM) | high_latency | 13.935 | 8.893 | 67.732 | 0.078 | 1.242 |
| Centralized NN (no physics) | high_latency | 12.197 | 8.236 | 65.347 | 0.022 | 0.795 |
| Decentralized NN (no physics) | high_latency | 13.884 | 8.843 | 66.293 | 0.039 | 1.629 |
| Dec-PINC-V2X | spoofing | 12.995 | 7.593 | 70.619 | 0.118 | 0.447 |
| Persistence (LOCF) | spoofing | 20.095 | 10.467 | 57.760 | 0.250 | 0.189 |
| Ridge | spoofing | 17.334 | 9.750 | 64.800 | 0.231 | 0.184 |
| GBM (LightGBM) | spoofing | 11.360 | 6.675 | 53.758 | 0.118 | 0.282 |
| Centralized NN (no physics) | spoofing | 10.996 | 7.677 | 66.950 | 0.019 | 0.260 |
| Decentralized NN (no physics) | spoofing | 11.187 | 7.039 | 52.229 | 0.168 | 0.342 |
