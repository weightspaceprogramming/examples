#!/usr/bin/env python3
"""Inspect input-channel weight norms; reproduce the four Sparks policy maps."""
import argparse
import hashlib
import json
from pathlib import Path
import tempfile
import urllib.request

import numpy as np

UPSTREAM = "https://github.com/RikiyaT/lm_understand-logs"
REVISION = "4d09ba6d6a7e891dd8063974dfe81b3d78e3494f"
CHANNELS = ["Agent", "Moving objects", "Reward targets", "Obstacles", "Previous frame"]
NETWORKS = [
    ("Breakout", "a9f535cc", "12297581cd5c3dfa4f7494c7f302e578b5819c8b9302c65f59cea5fb2dbe82e2"),
    ("Pong", "dfc6cb65", "c3bbd09e8cfaadb89fee8c8092c1240794d132f87a69c2b0ae8b1fe4f424082c"),
    ("Pac-Man", "bdf99279", "74d3487acdda971f1c0e18024ac310ac4cb79db65beb5cffb8d4ed9e84bff395"),
    ("Freeway", "0bbcc3ea", "235276cf02cc7efb51dc6a031ccf2e827adbfbdbd8c860dad437fafaaae9209e"),
]
KERNEL = "actor/dense1/kernel"


def sha256(payload):
    return hashlib.sha256(payload).hexdigest()


def channel_norms(kernel, grid_shape):
    """L2 norms over hidden units, with HWC inputs; result has shape C,H,W."""
    if len(grid_shape) != 3 or any(type(n) is not int or n <= 0 for n in grid_shape):
        raise ValueError("grid_shape must contain three positive integers: height, width, channels")
    values = np.asarray(kernel)
    if values.ndim != 2 or values.shape[1] == 0:
        raise ValueError("Kernel must have shape (input features, hidden units)")
    if values.dtype.kind not in "fiu" or not np.isfinite(values).all():
        raise ValueError("Kernel must contain finite real numbers")
    if values.shape[0] != np.prod(grid_shape):
        raise ValueError("Kernel input dimension does not equal height * width * channels")
    try:
        with np.errstate(over="raise", invalid="raise"):
            values = values.astype(np.float64)
            shaped = values.reshape(*grid_shape, values.shape[1])
            return np.sqrt(np.sum(shaped * shaped, axis=-1)).transpose(2, 0, 1)
    except FloatingPointError as error:
        raise ValueError("Squared kernel magnitudes exceed float64 range") from error


def load_arrays(path):
    with np.load(path, allow_pickle=False) as archive:
        return {key: archive[key] for key in archive.files}


def download_network(network_id, expected_sha256, cache):
    """Read a pinned file, verifying cached and downloaded bytes before use."""
    cache = Path(cache) / REVISION
    cache.mkdir(parents=True, exist_ok=True)
    target = cache / (network_id + ".npz")
    url = f"https://raw.githubusercontent.com/RikiyaT/lm_understand-logs/{REVISION}/exp1_which_game/networks/{network_id}.npz"
    if target.exists():
        payload = target.read_bytes()
    else:
        with urllib.request.urlopen(url, timeout=30) as response:
            payload = response.read(1_000_001)
        if len(payload) > 1_000_000:
            raise ValueError("Network download exceeds the expected file-size bound")
    if sha256(payload) != expected_sha256:
        raise ValueError(f"SHA-256 mismatch for {network_id}; the file was not accepted")
    if not target.exists():
        with tempfile.NamedTemporaryFile(dir=cache, delete=False) as temporary:
            temporary.write(payload)
            name = Path(temporary.name)
        try:
            name.replace(target)
        finally:
            name.unlink(missing_ok=True)
    return target, url


def actor_logits(arrays, inputs):
    hidden = np.tanh(inputs @ arrays["actor/dense1/kernel"] + arrays["actor/dense1/bias"])
    hidden = np.tanh(hidden @ arrays["actor/dense2/kernel"] + arrays["actor/dense2/bias"])
    return hidden @ arrays["actor/logits/kernel"] + arrays["actor/logits/bias"]


def permutation_check(arrays, grid_shape, seed=1729):
    """Reorder hidden units consistently, then compare maps and full actor outputs."""
    values = {key: value.astype(np.float64) for key, value in arrays.items() if key.startswith("actor/")}
    rng = np.random.default_rng(seed)
    permutation = rng.permutation(values[KERNEL].shape[1])
    changed = {key: value.copy() for key, value in values.items()}
    changed[KERNEL] = values[KERNEL][:, permutation]
    changed["actor/dense1/bias"] = values["actor/dense1/bias"][permutation]
    changed["actor/dense2/kernel"] = values["actor/dense2/kernel"][permutation, :]
    original_maps = channel_norms(values[KERNEL], grid_shape)
    permuted_maps = channel_norms(changed[KERNEL], grid_shape)
    inputs = rng.integers(0, 2, size=(64, values[KERNEL].shape[0])).astype(np.float64)
    before, after = actor_logits(values, inputs), actor_logits(changed, inputs)
    map_error = float(np.max(np.abs(original_maps - permuted_maps)))
    output_error = float(np.max(np.abs(before - after)))
    passed = bool(np.allclose(original_maps, permuted_maps, atol=1e-10, rtol=0) and
                  np.allclose(before, after, atol=1e-10, rtol=0))
    return {"seed": seed, "binary_input_vectors": len(inputs), "atol": 1e-10,
            "max_map_difference": map_error, "max_actor_logit_difference": output_error,
            "passed": passed}


def map_record(title, arrays, key, shape, channels):
    if len(channels) != shape[2]:
        raise ValueError("Supply one channel label per input channel")
    if key not in arrays:
        raise ValueError(f"Kernel key {key!r} is absent; available keys: {', '.join(arrays)}")
    maps = channel_norms(arrays[key], shape)
    return {"title": title, "kernel": key, "kernel_shape": list(arrays[key].shape),
            "grid_shape_hwc": list(shape), "channel_labels": channels,
            "norm_maps_chw": maps.tolist(),
            "channel_maxima": [float(x) for x in maps.max(axis=(1, 2))]}


def plot_records(records, output):
    import matplotlib
    matplotlib.use("Agg")
    matplotlib.rcParams["svg.hashsalt"] = "weightspaceprogramming-weight-reading"
    import matplotlib.pyplot as plt
    columns = max(len(record["channel_labels"]) for record in records)
    figure, axes = plt.subplots(len(records), columns, squeeze=False,
                               figsize=(2.45 * columns, 2.7 * len(records)))
    for row, record in enumerate(records):
        maps = np.asarray(record["norm_maps_chw"])
        vmax = float(maps.max()) or 1.0
        for column, axis in enumerate(axes[row]):
            if column >= len(maps):
                axis.set_visible(False)
                continue
            image = axis.imshow(maps[column], vmin=0, vmax=vmax, cmap="viridis", interpolation="nearest")
            axis.set_title(record["channel_labels"][column], fontsize=10)
            axis.set_xticks([0, maps.shape[2] - 1])
            axis.set_yticks([0, maps.shape[1] - 1])
            if column == 0:
                axis.set_ylabel(record["title"], fontsize=12)
            if column == columns - 1:
                figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
    figure.suptitle("Input-channel weight norms", fontsize=16)
    figure.text(0.5, 0.015, "L2 norm across first-layer hidden units · one shared color scale per network",
                ha="center", fontsize=10)
    figure.tight_layout(rect=(0, 0.04, 1, 0.96))
    figure.savefig(output / "heatmaps.png", dpi=140, metadata={"Software": "weightspaceprogramming/examples"})
    svg = output / "heatmaps.svg"
    figure.savefig(svg, metadata={"Date": None})
    svg.write_text("\n".join(line.rstrip() for line in svg.read_text().splitlines()) + "\n")
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    sparks = sub.add_parser("sparks", help="Download and inspect the four pinned Sparks policies")
    sparks.add_argument("--cache", type=Path, default=Path.home() / ".cache/weightspaceprogramming/sparks")
    inspect = sub.add_parser("inspect", help="Inspect a local NPZ kernel with HWC input order")
    inspect.add_argument("checkpoint", type=Path)
    inspect.add_argument("--key", default=KERNEL)
    inspect.add_argument("--shape", type=int, nargs=3, required=True, metavar=("H", "W", "C"))
    inspect.add_argument("--channels", nargs="+", required=True)
    for command in (sparks, inspect):
        command.add_argument("--output", type=Path, default=Path("results/weight-reading"))
    args = parser.parse_args()
    try:
        records = []
        if args.command == "sparks":
            for game, network_id, digest in NETWORKS:
                path, url = download_network(network_id, digest, args.cache)
                arrays = load_arrays(path)
                record = map_record(game, arrays, KERNEL, (10, 10, 5), CHANNELS)
                record.update({"network_id": network_id, "source_url": url, "sha256": digest,
                               "actor_parameters": sum(value.size for key, value in arrays.items() if key.startswith("actor/")),
                               "permutation_check": permutation_check(arrays, (10, 10, 5))})
                if not record["permutation_check"]["passed"]:
                    raise ValueError(f"Hidden-unit permutation check failed for {game}")
                records.append(record)
            source = {"repository": UPSTREAM, "revision": REVISION, "networks_license": "CC-BY-4.0",
                      "authors": ["Rikiya Takehi", "Jayden Teoh", "Alex Zhang", "Omar Khattab"]}
        else:
            records = [map_record(args.checkpoint.stem, load_arrays(args.checkpoint), args.key,
                                  tuple(args.shape), args.channels)]
            source = {"filename": args.checkpoint.name, "sha256": sha256(args.checkpoint.read_bytes())}
        args.output.mkdir(parents=True, exist_ok=True)
        result = {"computation": "sqrt(sum_j W[(row*width+column)*channels+channel,j]^2)",
                  "accumulation_dtype": "float64", "source": source, "networks": records}
        (args.output / "norms.json").write_text(json.dumps(result, indent=2) + "\n")
        plot_records(records, args.output)
        print(json.dumps({"networks": len(records), "channel_maps": sum(len(r["channel_labels"]) for r in records),
                          "output": str(args.output), "permutation_checks": [r.get("permutation_check") for r in records]}))
    except (ValueError, KeyError, OSError) as error:
        parser.exit(1, f"{error}\n")


if __name__ == "__main__":
    main()
