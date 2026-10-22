### T23_class_metrics: class-imbalance metrics, fair re-test (seeds 3-5, long tail, IF = 100)

Each method with its validation-selected readout (as in T22); test set, mean over three seeds, in %. Balanced accuracy = macro recall. G-mean = geometric mean of class recalls (each floored at 1%). Tail = mean recall of the rarest 40% of classes.

**MNIST**

| method | readout | balanced acc. | macro-F1 | G-mean | worst class | tail classes |
|---|---|---|---|---|---|---|
| **Ours** | ccvr | 95.6 | 95.5 | 95.5 | 86.4 | 91.9 |
| GeFL-F | ccvr | 92.2 | 92.1 | 91.9 | 78.3 | 86.2 |
| GeFL-F, decoupled decay + LA | ccvr | 93.5 | 93.5 | 93.4 | 82.9 | 88.9 |
| Real data only (no generator) | ccvr | 90.9 | 90.8 | 90.5 | 76.5 | 85.3 |
| Real data only + LA / balanced softmax | ccvr | 90.9 | 90.8 | 90.6 | 76.9 | 85.7 |
| FedLC | ccvr | 90.9 | 90.8 | 90.6 | 76.2 | 85.3 |
| FedRS | ccvr | 90.9 | 90.8 | 90.5 | 76.4 | 85.3 |
| Gaussian generator + LA | ccvr | 92.6 | 92.6 | 92.5 | 81.3 | 88.0 |
| FedDF | ccvr | 93.6 | 93.6 | 93.4 | 82.8 | 88.8 |
| FedDF + LA | ccvr | 93.6 | 93.5 | 93.4 | 83.4 | 88.9 |

Same metrics with every head as trained (no readout), MNIST:

| method | balanced acc. | macro-F1 | G-mean | worst class | tail classes |
|---|---|---|---|---|---|
| **Ours** | 95.2 | 95.1 | 95.1 | 87.4 | 91.5 |
| GeFL-F | 72.1 | 66.8 | 50.0 | 26.1 | 43.7 |
| GeFL-F, decoupled decay + LA | 88.9 | 88.7 | 88.3 | 72.1 | 81.6 |
| Real data only (no generator) | 60.9 | 53.8 | 34.2 | 21.7 | 37.1 |
| Real data only + LA / balanced softmax | 64.8 | 58.7 | 40.0 | 29.6 | 44.3 |
| FedLC | 61.4 | 54.5 | 35.1 | 22.5 | 38.1 |
| FedRS | 60.9 | 53.8 | 34.2 | 21.7 | 37.2 |
| Gaussian generator + LA | 88.2 | 88.1 | 87.6 | 73.1 | 82.1 |
| FedDF | 76.2 | 71.4 | 57.3 | 23.1 | 45.7 |
| FedDF + LA | 83.5 | 81.5 | 76.6 | 44.1 | 64.3 |

Per-class recall, MNIST (class 0 = most common, class 9 = rarest):

| method | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 |
|---|---|---|---|---|---|---|---|---|---|---|
| **Ours** | 99 | 99 | 98 | 97 | 98 | 97 | 97 | 95 | 89 | 87 |
| GeFL-F | 98 | 98 | 97 | 95 | 95 | 93 | 95 | 90 | 78 | 82 |
| GeFL-F, decoupled decay + LA | 98 | 98 | 96 | 96 | 96 | 94 | 95 | 91 | 84 | 85 |
| Real data only (no generator) | 98 | 98 | 95 | 93 | 94 | 90 | 93 | 89 | 77 | 82 |
| Real data only + LA / balanced softmax | 98 | 98 | 95 | 92 | 94 | 90 | 93 | 90 | 78 | 81 |
| FedLC | 98 | 98 | 95 | 93 | 94 | 90 | 93 | 89 | 77 | 82 |
| FedRS | 98 | 98 | 95 | 93 | 94 | 90 | 93 | 89 | 77 | 82 |
| Gaussian generator + LA | 98 | 98 | 95 | 95 | 95 | 93 | 95 | 91 | 82 | 83 |
| FedDF | 99 | 99 | 97 | 96 | 96 | 94 | 96 | 91 | 83 | 86 |
| FedDF + LA | 98 | 99 | 97 | 96 | 96 | 95 | 96 | 91 | 83 | 86 |

**FashionMNIST**

| method | readout | balanced acc. | macro-F1 | G-mean | worst class | tail classes |
|---|---|---|---|---|---|---|
| **Ours** | ccvr | 82.1 | 81.6 | 80.0 | 40.0 | 75.9 |
| GeFL-F | ccvr | 81.1 | 80.6 | 79.1 | 39.9 | 75.3 |
| GeFL-F, decoupled decay + LA | ccvr | 81.6 | 81.2 | 79.7 | 41.4 | 76.0 |
| Real data only (no generator) | ccvr | 79.5 | 79.0 | 77.2 | 37.5 | 73.9 |
| Real data only + LA / balanced softmax | ccvr | 79.5 | 79.1 | 77.2 | 36.8 | 73.7 |
| FedLC | ccvr | 79.5 | 79.0 | 77.2 | 37.2 | 73.8 |
| FedRS | ccvr | 79.4 | 79.0 | 77.2 | 37.8 | 73.8 |
| Gaussian generator + LA | ccvr | 80.8 | 80.4 | 78.8 | 40.5 | 75.6 |
| FedDF | ccvr | 81.8 | 81.4 | 79.9 | 41.6 | 76.5 |
| FedDF + LA | ccvr | 81.8 | 81.4 | 80.0 | 42.4 | 76.5 |

Same metrics with every head as trained (no readout), FashionMNIST:

| method | balanced acc. | macro-F1 | G-mean | worst class | tail classes |
|---|---|---|---|---|---|
| **Ours** | 80.0 | 79.1 | 76.4 | 31.0 | 73.6 |
| GeFL-F | 64.4 | 60.3 | 46.9 | 17.3 | 43.5 |
| GeFL-F, decoupled decay + LA | 76.7 | 75.6 | 72.6 | 29.4 | 69.0 |
| Real data only (no generator) | 52.1 | 45.2 | 25.9 | 13.2 | 31.4 |
| Real data only + LA / balanced softmax | 57.5 | 52.0 | 34.9 | 17.2 | 39.3 |
| FedLC | 52.7 | 46.1 | 27.1 | 13.6 | 32.3 |
| FedRS | 52.1 | 45.2 | 25.9 | 13.2 | 31.3 |
| Gaussian generator + LA | 74.3 | 73.2 | 69.7 | 28.4 | 68.2 |
| FedDF | 65.7 | 61.6 | 46.4 | 11.2 | 45.5 |
| FedDF + LA | 73.1 | 70.9 | 64.1 | 23.8 | 59.0 |

Per-class recall, FashionMNIST (class 0 = most common, class 9 = rarest):

| method | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 |
|---|---|---|---|---|---|---|---|---|---|---|
| **Ours** | 86 | 96 | 81 | 86 | 74 | 94 | 40 | 91 | 87 | 87 |
| GeFL-F | 85 | 96 | 79 | 86 | 73 | 91 | 40 | 89 | 85 | 87 |
| GeFL-F, decoupled decay + LA | 86 | 95 | 80 | 85 | 74 | 92 | 41 | 90 | 86 | 87 |
| Real data only (no generator) | 83 | 95 | 76 | 84 | 70 | 91 | 38 | 88 | 84 | 85 |
| Real data only + LA / balanced softmax | 84 | 95 | 76 | 84 | 71 | 91 | 37 | 88 | 84 | 86 |
| FedLC | 83 | 95 | 76 | 84 | 70 | 91 | 37 | 88 | 84 | 86 |
| FedRS | 83 | 95 | 76 | 84 | 70 | 91 | 38 | 88 | 84 | 85 |
| Gaussian generator + LA | 84 | 95 | 77 | 85 | 73 | 91 | 41 | 89 | 87 | 86 |
| FedDF | 85 | 96 | 79 | 86 | 74 | 92 | 42 | 90 | 87 | 87 |
| FedDF + LA | 85 | 96 | 80 | 86 | 74 | 92 | 42 | 90 | 86 | 88 |

**SVHN**

| method | readout | balanced acc. | macro-F1 | G-mean | worst class | tail classes |
|---|---|---|---|---|---|---|
| **Ours** | ccvr | 65.6 | 65.1 | 61.6 | 30.0 | 47.2 |
| GeFL-F | ccvr | 64.0 | 63.5 | 60.7 | 29.9 | 48.1 |
| GeFL-F, decoupled decay + LA | ccvr | 64.1 | 63.5 | 60.5 | 29.4 | 47.2 |
| Real data only (no generator) | er | 54.3 | 53.6 | 49.3 | 20.2 | 36.4 |
| Real data only + LA / balanced softmax | er | 54.3 | 53.6 | 49.2 | 20.2 | 36.4 |
| FedLC | er | 54.3 | 53.6 | 49.2 | 20.3 | 36.4 |
| FedRS | er | 54.3 | 53.6 | 49.3 | 20.3 | 36.4 |
| Gaussian generator + LA | er | 53.2 | 52.4 | 48.4 | 20.2 | 35.5 |
| FedDF | ccvr | 65.0 | 64.5 | 61.8 | 30.3 | 49.7 |
| FedDF + LA | ccvr | 65.8 | 65.2 | 62.5 | 30.2 | 50.2 |

Same metrics with every head as trained (no readout), SVHN:

| method | balanced acc. | macro-F1 | G-mean | worst class | tail classes |
|---|---|---|---|---|---|
| **Ours** | 64.2 | 63.8 | 61.0 | 32.2 | 47.5 |
| GeFL-F | 50.2 | 47.8 | 33.8 | 6.4 | 24.2 |
| GeFL-F, decoupled decay + LA | 60.6 | 59.8 | 57.3 | 32.6 | 47.2 |
| Real data only (no generator) | 30.1 | 23.0 | 9.7 | 2.1 | 7.9 |
| Real data only + LA / balanced softmax | 33.8 | 27.9 | 14.6 | 4.6 | 12.9 |
| FedLC | 30.5 | 23.5 | 10.2 | 2.4 | 8.3 |
| FedRS | 30.1 | 23.0 | 9.7 | 2.1 | 7.9 |
| Gaussian generator + LA | 41.1 | 39.0 | 33.8 | 18.3 | 25.2 |
| FedDF | 41.2 | 37.8 | 20.8 | 0.8 | 9.5 |
| FedDF + LA | 55.8 | 54.4 | 46.1 | 11.8 | 31.5 |

Per-class recall, SVHN (class 0 = most common, class 9 = rarest):

| method | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 |
|---|---|---|---|---|---|---|---|---|---|---|
| **Ours** | 85 | 83 | 84 | 69 | 77 | 68 | 61 | 67 | 30 | 31 |
| GeFL-F | 81 | 81 | 81 | 64 | 74 | 67 | 57 | 69 | 30 | 37 |
| GeFL-F, decoupled decay + LA | 82 | 81 | 81 | 64 | 76 | 67 | 58 | 68 | 29 | 34 |
| Real data only (no generator) | 74 | 81 | 71 | 52 | 65 | 54 | 41 | 62 | 20 | 23 |
| Real data only + LA / balanced softmax | 74 | 80 | 71 | 52 | 65 | 54 | 41 | 62 | 20 | 23 |
| FedLC | 74 | 80 | 71 | 52 | 65 | 54 | 41 | 62 | 20 | 22 |
| FedRS | 74 | 81 | 71 | 52 | 65 | 54 | 41 | 62 | 20 | 23 |
| Gaussian generator + LA | 76 | 79 | 69 | 50 | 64 | 51 | 40 | 58 | 22 | 22 |
| FedDF | 81 | 82 | 82 | 65 | 75 | 67 | 59 | 71 | 30 | 39 |
| FedDF + LA | 83 | 82 | 82 | 66 | 77 | 68 | 60 | 72 | 30 | 39 |

**CIFAR-10**

| method | readout | balanced acc. | macro-F1 | G-mean | worst class | tail classes |
|---|---|---|---|---|---|---|
| **Ours** | er | 53.1 | 52.9 | 51.9 | 41.2 | 51.8 |
| GeFL-F | er | 50.7 | 50.5 | 49.5 | 38.6 | 48.4 |
| GeFL-F, decoupled decay + LA | er | 51.6 | 51.4 | 50.4 | 38.4 | 49.6 |
| Real data only (no generator) | er | 51.5 | 51.3 | 50.3 | 39.8 | 51.4 |
| Real data only + LA / balanced softmax | er | 51.6 | 51.3 | 50.4 | 39.8 | 51.4 |
| FedLC | er | 51.5 | 51.3 | 50.3 | 40.0 | 51.4 |
| FedRS | er | 51.5 | 51.3 | 50.3 | 39.7 | 51.3 |
| Gaussian generator + LA | er | 52.6 | 52.4 | 51.4 | 41.0 | 52.1 |
| FedDF | er | 50.8 | 50.6 | 49.5 | 38.1 | 48.2 |
| FedDF + LA | er | 51.4 | 51.2 | 50.2 | 38.8 | 48.9 |

Same metrics with every head as trained (no readout), CIFAR-10:

| method | balanced acc. | macro-F1 | G-mean | worst class | tail classes |
|---|---|---|---|---|---|
| **Ours** | 44.8 | 43.2 | 39.8 | 27.1 | 35.9 |
| GeFL-F | 33.6 | 26.9 | 15.6 | 4.8 | 13.8 |
| GeFL-F, decoupled decay + LA | 39.8 | 37.0 | 32.2 | 15.8 | 27.1 |
| Real data only (no generator) | 33.9 | 27.1 | 14.9 | 6.0 | 15.6 |
| Real data only + LA / balanced softmax | 36.2 | 30.3 | 19.2 | 9.6 | 19.9 |
| FedLC | 34.1 | 27.3 | 15.3 | 6.2 | 15.9 |
| FedRS | 33.9 | 27.1 | 15.0 | 6.0 | 15.7 |
| Gaussian generator + LA | 43.2 | 41.4 | 37.6 | 23.1 | 33.1 |
| FedDF | 37.3 | 30.5 | 18.1 | 1.7 | 10.5 |
| FedDF + LA | 42.1 | 38.1 | 31.0 | 9.1 | 20.7 |

Per-class recall, CIFAR-10 (class 0 = most common, class 9 = rarest):

| method | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 |
|---|---|---|---|---|---|---|---|---|---|---|
| **Ours** | 68 | 76 | 46 | 42 | 46 | 47 | 60 | 52 | 54 | 41 |
| GeFL-F | 66 | 74 | 45 | 40 | 45 | 44 | 56 | 48 | 51 | 39 |
| GeFL-F, decoupled decay + LA | 67 | 75 | 46 | 41 | 45 | 45 | 58 | 51 | 52 | 38 |
| Real data only (no generator) | 65 | 74 | 43 | 40 | 44 | 45 | 60 | 51 | 52 | 42 |
| Real data only + LA / balanced softmax | 65 | 74 | 43 | 40 | 44 | 45 | 60 | 51 | 53 | 42 |
| FedLC | 65 | 74 | 43 | 40 | 44 | 45 | 60 | 51 | 52 | 42 |
| FedRS | 65 | 74 | 43 | 40 | 44 | 45 | 60 | 51 | 52 | 42 |
| Gaussian generator + LA | 67 | 75 | 45 | 41 | 44 | 46 | 60 | 53 | 54 | 41 |
| FedDF | 67 | 75 | 44 | 41 | 45 | 44 | 56 | 48 | 50 | 38 |
| FedDF + LA | 68 | 75 | 45 | 41 | 45 | 45 | 57 | 49 | 50 | 39 |

**CIFAR-100-LT**

| method | readout | balanced acc. | macro-F1 | G-mean | worst class | tail classes |
|---|---|---|---|---|---|---|
| **Ours** | er | 18.7 | 16.4 | 10.7 | 0.2 | 8.4 |
| GeFL-F | er | 14.8 | 12.9 | 8.2 | 0.2 | 6.7 |
| GeFL-F, decoupled decay + LA | er | 15.7 | 13.8 | 9.0 | 0.1 | 7.1 |
| Real data only (no generator) | er | 16.9 | 14.9 | 9.8 | 0.2 | 8.6 |
| Real data only + LA / balanced softmax | er | 17.1 | 15.1 | 10.0 | 0.2 | 8.7 |
| FedLC | er | 16.9 | 14.9 | 9.8 | 0.2 | 8.6 |
| FedRS | er | 17.0 | 15.0 | 9.9 | 0.3 | 8.6 |
| Gaussian generator + LA | er | 17.1 | 15.1 | 10.0 | 0.3 | 8.5 |
| FedDF | er | 15.3 | 13.3 | 8.6 | 0.2 | 6.8 |
| FedDF + LA | er | 15.2 | 13.2 | 8.5 | 0.2 | 6.8 |

Same metrics with every head as trained (no readout), CIFAR-100-LT:

| method | balanced acc. | macro-F1 | G-mean | worst class | tail classes |
|---|---|---|---|---|---|
| **Ours** | 14.2 | 12.4 | 7.6 | 0.1 | 6.0 |
| GeFL-F | 7.6 | 4.6 | 2.9 | 0.0 | 1.6 |
| GeFL-F, decoupled decay + LA | 9.6 | 7.2 | 4.3 | 0.1 | 4.0 |
| Real data only (no generator) | 8.4 | 5.2 | 2.9 | 0.0 | 1.7 |
| Real data only + LA / balanced softmax | 8.9 | 5.8 | 3.3 | 0.0 | 2.3 |
| FedLC | 8.4 | 5.3 | 3.0 | 0.0 | 1.8 |
| FedRS | 8.4 | 5.2 | 2.9 | 0.0 | 1.7 |
| Gaussian generator + LA | 13.0 | 11.3 | 7.1 | 0.4 | 7.8 |
| FedDF | 9.5 | 5.7 | 3.2 | 0.0 | 0.7 |
| FedDF + LA | 10.7 | 7.1 | 4.0 | 0.0 | 1.4 |

