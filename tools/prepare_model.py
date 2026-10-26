#!/usr/bin/env python3
"""Offline model preparation (never needed at runtime).

Downloads a Hugging Face DINOv2 ONNX export and exposes intermediate transformer
layers as extra graph outputs so the C++ engine can fuse multi-layer patch
features (VisionAD / Dinomaly2 style). Requires only `pip install onnx`.

    python3 tools/prepare_model.py --variant reg-small --layers 5 8 --out models/dinov2_ml.onnx
"""
import argparse
import re
import urllib.request
from pathlib import Path

import onnx

VARIANTS = {
    "small": "onnx-community/dinov2-small",
    "reg-small": "onnx-community/dinov2-with-registers-small",
    "base": "onnx-community/dinov2-base",
    "reg-base": "onnx-community/dinov2-with-registers-base",
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", default="reg-small", choices=sorted(VARIANTS))
    ap.add_argument("--source", help="use an already downloaded ONNX file instead of downloading")
    ap.add_argument("--layers", type=int, nargs="*", default=[5, 8],
                    help="0-based encoder layers to expose in addition to last_hidden_state")
    ap.add_argument("--out", default="models/dinov2_ml.onnx")
    args = ap.parse_args()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    src = Path(args.source) if args.source else out.with_name(f"dinov2-{args.variant}.onnx")
    if not src.exists():
        url = f"https://huggingface.co/{VARIANTS[args.variant]}/resolve/main/onnx/model.onnx"
        print(f"downloading {url}")
        urllib.request.urlretrieve(url, src)

    model = onnx.load(str(src))
    layer_out = {}
    for node in model.graph.node:
        m = re.search(r"/encoder/layer\.(\d+)/Add_1$", node.name)
        if m:
            layer_out[int(m.group(1))] = node.output[0]

    existing = {o.name for o in model.graph.output}
    for layer in args.layers:
        if layer not in layer_out:
            raise SystemExit(f"layer {layer} not found; available: {sorted(layer_out)}")
        name = layer_out[layer]
        if name in existing:
            continue
        # Rename the tensor to a stable output name by inserting an Identity node.
        stable = f"layer_{layer}"
        model.graph.node.append(onnx.helper.make_node("Identity", [name], [stable]))
        model.graph.output.append(onnx.helper.make_tensor_value_info(stable, onnx.TensorProto.FLOAT, None))

    onnx.save(model, str(out))
    print(f"wrote {out} with outputs {[o.name for o in model.graph.output]}")


if __name__ == "__main__":
    main()
