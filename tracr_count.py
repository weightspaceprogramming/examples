#!/usr/bin/env python3
"""Compile a RASP token counter to transformer weights and check its outputs.

Install Tracr following https://github.com/google-deepmind/tracr.
This example uses a small CPU model, without training or hosted inference.
"""
import argparse
import itertools
import json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sequence", action="append", help="A nonempty sequence of a/b/c; repeat for several")
    parser.add_argument("--all-up-to", type=int, default=0, metavar="N", help="Also check every sequence through length N (1–4)")
    args = parser.parse_args()
    if not 0 <= args.all_up_to <= 4:
        parser.error("--all-up-to must be between 0 and 4")

    from tracr.rasp import rasp
    from tracr.compiler import compiling

    vocab = {"a", "b", "c"}
    bos = "BOS"
    max_seq_len = 8
    count_a = rasp.SelectorWidth(
        rasp.Select(rasp.tokens, rasp.tokens, lambda key, query: key == "a")
    ).named("count_a")
    model = compiling.compile_rasp_to_model(
        count_a, vocab=vocab, max_seq_len=max_seq_len, compiler_bos=bos,
    )
    print(json.dumps({"parameter_arrays": sum(len(group) for group in model.params.values()),
                      "max_sequence_length": max_seq_len, "counted_token": "a"}))
    sequences = args.sequence or ["abac", "bbb", "a", "caaab"]
    if args.all_up_to:
        sequences += ["".join(tokens) for n in range(1, args.all_up_to + 1)
                      for tokens in itertools.product(sorted(vocab), repeat=n)]
    failures = 0
    for sequence in dict.fromkeys(sequences):
        if not sequence or len(sequence) > max_seq_len or set(sequence) - vocab:
            parser.error("Sequences must contain 1–8 tokens from a, b, c.")
        tokens = list(sequence)
        expected = [tokens.count("a")] * len(tokens)
        interpreted = list(count_a(tokens))
        decoded = model.apply([bos, *tokens]).decoded[1:]
        actual = [float(value) for value in decoded]
        matches = interpreted == expected and all(abs(value - target) < 1e-5 for value, target in zip(actual, expected)) and len(actual) == len(expected)
        failures += not matches
        print(json.dumps({"input": tokens, "expected": expected, "rasp_output": interpreted,
                          "transformer_output": actual, "matches": matches}))
    if failures:
        raise SystemExit(f"{failures} sequences did not match")


if __name__ == "__main__":
    main()
