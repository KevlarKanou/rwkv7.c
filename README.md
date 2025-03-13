# rwkv7.c

*Inspired by [llama2.c](https://github.com/karpathy/llama2.c).*

Inference RWKV v7 in **pure C**, WIP.

## Tested model

```
RWKV-x070-World-0.1B-v2.8-20241210-ctx4096.pth
RWKV-x070-World-0.4B-v2.9-20250107-ctx4096.pth
RWKV-x070-World-2.9B-v3-20250211-ctx4096.pth
rwkv7-g1-0.1b-20250307-ctx4096.pth
```

## Usage

``` shell
wget "https://huggingface.co/BlinkDL/rwkv-7-world/resolve/main/RWKV-x070-World-0.4B-v2.9-20250107-ctx4096.pth" -O model.pth
python ./utils/export.py ./model.pth ./model.bin # pip install torch rwkv
make
# generate mode
./rwkv7 ./model.bin -i "Once upon a time," --temperature 1.3 --top-p 0.8
# chat mode
./rwkv7 ./model.bin --chat -i "Where is the capital of France?" --temperature 0.8 --top-p 0.2
# reasoner mode (rwkv7-g1)
./rwkv7 ./model-g1.bin --reasoner -i "What is RWKV?" # default params: temperature = 1.0, top-p = 0.7
```

## TODO
- Repetition penalty
- Optimize with vector instruction (AVX, NEON...)
- Load generic model format (.pth, .gguf...)