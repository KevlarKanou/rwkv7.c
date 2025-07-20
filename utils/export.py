import os
os.environ["RWKV_V7_ON"] = "1"
import sys
import gc
import struct
import torch
from rwkv.model import RWKV

def serialize_fp32(file, tensor):
    d = tensor.detach().cpu().view(-1).to(torch.float32).numpy()
    b = struct.pack(f'{len(d)}f', *d)
    file.write(b)

def serialize_fp16(file, tensor):
    d = tensor.detach().cpu().view(-1).to(torch.float16).numpy()
    b = struct.pack(f'{len(d)}e', *d)
    file.write(b)

def export(model, export_path, quant):
    if quant not in ['fp32', 'fp16']:
        raise ValueError("data type must be 'fp32' or 'fp16'")
    quant_idx = 0 if quant == 'fp32' else 1

    # r  w  k  v  7  .  c
    # 72 77 6B 76 37 2E 63
    magic_number = 0x00632E37766B7772

    w_lora_rank = model['blocks.0.att.w1'].size()[1]
    a_lora_rank = model['blocks.0.att.a1'].size()[1]
    g_lora_rank = model['blocks.0.att.g1'].size()[1]
    v_lora_rank = model['blocks.1.att.v1'].size()[1]

    export_model = open(export_path, 'wb')
    header = struct.pack(
        'Liiiiiiiii',
        magic_number, quant_idx, head_size, n_embd, n_layer, vocab_size,
        w_lora_rank, a_lora_rank, g_lora_rank, v_lora_rank
    )
    export_model.write(header)

    weights = [
        model['emb.weight'],                 

        model['blocks.0.ln0.weight'],
        model['blocks.0.ln0.bias'],
    ]

    for i in range(n_layer):
        weights.extend([
            model[f'blocks.{i}.ln1.weight'],
            model[f'blocks.{i}.ln1.bias'],
            model[f'blocks.{i}.ln2.weight'],
            model[f'blocks.{i}.ln2.bias'],

            model[f'blocks.{i}.att.x_r'],
            model[f'blocks.{i}.att.x_w'],
            model[f'blocks.{i}.att.x_k'],
            model[f'blocks.{i}.att.x_v'],
            model[f'blocks.{i}.att.x_a'],
            model[f'blocks.{i}.att.x_g'],
            model[f'blocks.{i}.att.w0'],
            model[f'blocks.{i}.att.r_k'],
            model[f'blocks.{i}.att.w1'],
            model[f'blocks.{i}.att.w2'],
            model[f'blocks.{i}.att.a1'],
            model[f'blocks.{i}.att.a2'],
            model[f'blocks.{i}.att.a0'],
            model[f'blocks.{i}.att.g1'],
            model[f'blocks.{i}.att.g2'],
        ])
        if i != 0:
            weights.extend([
                model[f'blocks.{i}.att.v2'],
                model[f'blocks.{i}.att.v1'],
                model[f'blocks.{i}.att.v0'],
            ])
        weights.extend([
            model[f'blocks.{i}.att.k_k'],
            model[f'blocks.{i}.att.k_a'],
            model[f'blocks.{i}.att.receptance.weight'],
            model[f'blocks.{i}.att.key.weight'],
            model[f'blocks.{i}.att.value.weight'],
            model[f'blocks.{i}.att.output.weight'],
            model[f'blocks.{i}.att.ln_x.weight'],
            model[f'blocks.{i}.att.ln_x.bias'],

            model[f'blocks.{i}.ffn.x_k'],
            model[f'blocks.{i}.ffn.key.weight'],
            model[f'blocks.{i}.ffn.value.weight'],
        ])

    weights.extend([
        model['ln_out.weight'],
        model['ln_out.bias'],
        model['head.weight'],
    ])

    for w in weights:
        if quant == 'fp32':
            serialize_fp32(export_model, w)
        elif quant == 'fp16':
            serialize_fp16(export_model, w)

    print("Model export done: ", export_path)
    export_model.close()

if len(sys.argv) < 4:
    print('Usage: python export.py <rwkv_model> <export_model> <data_type>')
    sys.exit(1)

rwkv_model = os.path.abspath(sys.argv[1])
if not os.path.exists(rwkv_model):
    print(f'File {rwkv_model} not found')
    sys.exit(1)

print("Loading model...")
model = RWKV(model=rwkv_model[:-4], strategy='cpu fp32')
head_size = model.args.head_size
n_embd = model.args.n_embd
n_layer = model.args.n_layer
vocab_size = model.args.vocab_size
del model
gc.collect()
model = torch.load(rwkv_model, map_location='cpu', weights_only=True)

print("Exporting model...")
export(model, os.path.abspath(sys.argv[2]), sys.argv[3])

