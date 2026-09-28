Avenue test clips: 103, calibration clips: 48, threads: 1, threshold percentile: 95.0

| Model | Precision | Size (MB) | Latency / clip (ms) | Latency / frame (ms) | Mean MSE | Pearson r vs orig. | Rel. score err. | Decision agreement | Accuracy loss |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Keras (original) | fp32 | 12.87 | 273 | 27.3 | 0.02090 | 1.0000 | 0.00% | 100.0% | 0.0% |
| TFLite direct | int8 dyn-range | 5.15 | 374 | 37.4 | 0.02091 | 1.0000 | 0.06% | 100.0% | 0.0% |
| TFLite edge graph | fp32 | 4.52 | 431 | 43.1 | 0.02090 | 1.0000 | 0.00% | 100.0% | 0.0% |
| TFLite edge graph | int8 dyn-range | 1.34 | 118 | 11.8 | 0.02107 | 1.0000 | 1.05% | 100.0% | 0.0% |
| PyTorch port | fp32 | 4.28 | 376 | 37.6 | 0.02090 | 1.0000 | 0.00% | 100.0% | 0.0% |
| PyTorch quantize_dynamic | int8 dynamic | 4.28 | 372 | 37.2 | 0.02090 | 1.0000 | 0.00% | 100.0% | 0.0% |
| PyTorch static PTQ | int8 static | 1.21 | 226 | 22.6 | 0.02181 | 1.0000 | 5.85% | 100.0% | 0.0% |

- TFLite edge graph int8 dyn-range: size -70.4%, latency 431 -> 118 ms/clip (3.66x), accuracy loss 0.0%
- PyTorch static PTQ int8 static: size -71.8%, latency 376 -> 226 ms/clip (1.66x), accuracy loss 0.0%
