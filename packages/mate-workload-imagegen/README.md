# mate-workload-imagegen

Image generation workload plugin for [mate-bench](https://github.com/T0nd3/mate-bench).

**Open mode only.** Closed mode is intentionally unsupported — image model weights are
not standardised across installations, making reproducible comparison impossible without
a canonical checkpoint reference.

## Metrics

| Metric | Description |
|--------|-------------|
| `images_per_second` | Generation throughput (higher is better) |
| `steps_per_second` | Sampler step throughput |
| `time_per_image_s` | Wall-clock time per image |

## Profiles

| Profile | Prompts | Resolution | Steps |
|---------|---------|------------|-------|
| `quick-512` | 5 | 512×512 | 20 |
| `standard-1024` | 3 | 1024×1024 | 20 |

## Usage

```bash
mate-bench run imagegen --profile quick-512 --mode open --model "v1-5-pruned-emaonly.ckpt"
mate-bench run imagegen --profile standard-1024 --mode open --model "sdxl_base_1.0.safetensors"
```
