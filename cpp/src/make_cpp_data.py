#!/usr/bin/env python
"""Generate reference data for the C++ decoder benchmark (qec_bench).

Wraps NVIDIA's code/export/generate_test_data.py and additionally writes the
detector error model as text (model.dem), so the C++ program builds exactly the
same PyMatching decoder that the Python reference used.

Run from the Ising-Decoding repo root with PYTHONPATH=code, for example:

    CUDA_VISIBLE_DEVICES="" PYTHONPATH=code python make_cpp_data.py \
        --noise si1000 --p 0.002 --basis X --num-samples 20000 \
        --onnx-model predecoder.onnx --output-dir data/d9_T9_X_si1000_p002
"""
import argparse
from pathlib import Path

import export.generate_test_data as g


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--noise", choices=["si1000", "depolarizing"], required=True)
    ap.add_argument("--p", type=float, required=True, help="physical error scale p")
    ap.add_argument("--distance", type=int, default=9)
    ap.add_argument("--n-rounds", type=int, default=9)
    ap.add_argument("--basis", default="X", choices=["X", "Z"])
    ap.add_argument("--num-samples", type=int, default=20000)
    ap.add_argument("--onnx-model", default=None)
    ap.add_argument("--output-dir", required=True)
    a = ap.parse_args()

    if a.noise == "si1000":
        nm = g.NoiseModel.from_si1000(a.p)
    else:
        nm = g.NoiseModel.from_single_p(a.p)
    params = nm.to_config_dict()

    g.generate_test_data(
        distance=a.distance,
        n_rounds=a.n_rounds,
        basis=a.basis,
        p_error=a.p,
        code_rotation="XV",
        noise_model_params=params,
        num_samples=a.num_samples,
        onnx_model=a.onnx_model,
        output_dir=a.output_dir,
    )

    # Rebuild the identical circuit (construction is deterministic) and write
    # its detector error model, with the same options the reference decoder used.
    noise_model = g.NoiseModel.from_config_dict(params)
    pp = float(noise_model.get_max_probability())
    circ = g.MemoryCircuit(
        distance=a.distance,
        idle_error=pp,
        sqgate_error=pp,
        tqgate_error=pp,
        spam_error=(2.0 / 3.0) * pp,
        n_rounds=a.n_rounds,
        basis=a.basis.upper(),
        code_rotation=g.normalize_code_rotation("XV"),
        noise_model=noise_model,
        add_boundary_detectors=True,
    )
    circ.set_error_rates()
    dem = circ.stim_circuit.detector_error_model(
        decompose_errors=True,
        approximate_disjoint_errors=True,
    )
    out = Path(a.output_dir)
    dem.to_file(str(out / "model.dem"))
    with open(out / "metadata.txt", "a") as f:
        f.write(f"noise={a.noise}\nnoise_p={a.p}\n")
    print(f"wrote {out / 'model.dem'} ({dem.num_detectors} detectors, {dem.num_errors} error mechanisms)")


if __name__ == "__main__":
    main()
