# Pytorch_GPU_Benchmark
toy benchmark
```
python3 benchmark.py --gpu 0 --mix_precision
```
for AMD NAVI based gpu (except navi 21) add:

```
HSA_OVERRIDE_GFX_VERSION=10.3.0
```
