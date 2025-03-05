# rwkv7.c

*Inspired by [llama2.c](https://github.com/karpathy/llama2.c).*

Inference RWKV v7 in **pure C**.

WIP, only implement prefilling now.

``` shell
wget "https://huggingface.co/BlinkDL/rwkv-7-world/resolve/main/RWKV-x070-World-0.4B-v2.9-20250107-ctx4096.pth" -O model.pth
python ./utils/export.py ./model.pth ./model.bin # pip install torch rwkv
make
./rwkv7 ./model.bin
```

# TODO
- Decoding
- Optimize with vector instruction (AVX, NEON...)
- Load generic model format (.pth, .gguf...)