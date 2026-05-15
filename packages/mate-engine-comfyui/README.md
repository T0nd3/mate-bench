# mate-engine-comfyui

[ComfyUI](https://github.com/comfyanonymous/ComfyUI) engine plugin for
[mate-bench](https://github.com/T0nd3/mate-bench).

Connects to a running ComfyUI server via its REST API and submits txt2img workflows,
measuring wall-clock generation time per image.

## Usage

Start ComfyUI first, then run mate-bench:

```bash
# ComfyUI running on default port 8188
mate-bench run imagegen --profile quick-512 --mode open \
    --model "v1-5-pruned-emaonly.ckpt" --engine comfyui
```

## Configuration

| Option | Default | Description |
|--------|---------|-------------|
| `host` | `127.0.0.1` | ComfyUI server hostname |
| `port` | `8188` | ComfyUI server port |
| `timeout` | `600.0` | Per-request timeout in seconds |

## Requirements

- A running [ComfyUI](https://github.com/comfyanonymous/ComfyUI) instance
- The checkpoint file accessible to ComfyUI (placed in its `models/checkpoints/` dir)
