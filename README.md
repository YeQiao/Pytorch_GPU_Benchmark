# Pytorch_GPU_Benchmark
toy benchmark
```
python benchmark.py --gpu 0 --mix_precision
```
for AMD NAVI based gpu (except navi 21) add:

```
HSA_OVERRIDE_GFX_VERSION=10.3.0

```
| GPU model | network  | epochs | pricision | batch sizes | initial lr | typical power | time |
|:--------:|:----------:|:--:|:----:|:----:|:---:|:----:|:------:|
| rtx 2060 | resnet 18  | 30 | fp32 |  512 | 0.1 | 165w | 17m25s |
| rtx 2060 | resnet 18  | 30 | fp16 |  512 | 0.1 | 165w |  8m26s |
|          |            |    |      |      |     |      |        |
| rtx 3080 | resnet 18  | 30 | fp32 |  512 | 0.1 | 336w |  7m59s |
| rtx 3080 | resnet 18  | 30 | fp16 |  512 | 0.1 | 250w |  6m48s |
| rtx 3080 | resnet 18  | 30 | fp32 | 2048 | 0.1 | 320w |  7m22s |
| rtx 3080 | resnet 18  | 30 | fp16 | 2048 | 0.1 | 240w |  5m49s |
|          |            |    |      |      |     |      |        |
| rtx 4070 | resnet 18  | 30 | fp16 | 256 | 0.1 | 200w |  6m22s |
| rtx 4070 | resnet 18  | 30 | fp16 | 512 | 0.1 | 200w |  6m30s |
| rtx 4070 | resnet 18  | 30 | fp16 | 2048 | 0.1 | 200w |  6m32s |
|          |            |    |      |      |     |      |        |
| rtx 4060 | resnet 18  | 30 | fp16 | 512 | 0.1 | 100w |  8m41s |
| rtx 4060 | resnet 18  | 30 | fp16 | 1024 | 0.1 | 100w |  8m38s |
|          |            |    |      |      |     |      |        |
|          |            |    |      |      |     |      |        |
|  rx6600  | resnet 18  | 30 | fp32 |  512 | 0.1 |  80w | 28m32s |
|  rx6600  | resnet 18  | 30 | fp16 |  512 | 0.1 |  80w | 21m08s |
|  rx6600  | resnet 18  | 30 | fp16 | 2048 | 0.1 |  90w | 24m12s |
|          |            |    |      |      |     |      |        |
| rx6700xt | resnet 18  | 30 | fp32 |  512 | 0.1 | 140w |        |
| rx6700xt | resnet 18  | 30 | fp16 |  512 | 0.1 | 155w | 10m59s |
| rx6700xt | resnet 18  | 30 | fp16 | 2048 | 0.1 | 140w | 15m05s |
| rx6700xt | resnet 18  | 30 | fp16 |  256 | 0.1 | 140w | 15m05s |
|          |            |    |      |      |     |      |        |
| rx6800xt | resnet 18  | 30 | fp16 |  512 | 0.1 | 270w |  9m43s  |
|          |            |    |      |      |     |      |        |
| rx6900xt | resnet 18  | 30 | fp16 |  512 | 0.1 | 270w |  7m55s  |
| rx6900xt | resnet 18  | 30 | fp16 | 2048 | 0.1 | 270w |  10m05s |
| rx6900xt | resnet 18  | 30 | fp16 |  256 | 0.1 | 270w |  7m45s  |
|          |            |    |      |      |     |      |        |
| rx7900xt | resnet 18  | 30 | fp16 |  512 | 0.1 | 250w |  5m48s  |
| rx7900xt | resnet 18  | 30 | fp16 |  2048 | 0.1 | 250w |  5m52s  |
