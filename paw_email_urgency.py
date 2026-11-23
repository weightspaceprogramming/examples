#!/usr/bin/env python3
"""Compile, save, reload, and test an email-urgency function.

Install: pip install programasweights --extra-index-url https://pypi.programasweights.com/simple/
Run with --help for commands. Compilation contacts the PAW service; run and
prompt execute locally. --offline requires previously downloaded model assets.
"""
import argparse
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
import json
from pathlib import Path
import time
from urllib.parse import quote
from urllib.request import urlopen

COMPILER = "paw-4b-qwen3-0.6b"
SPEC = (
    "Classify whether this email needs prompt action from its recipient. "
    "Return exactly URGENT or ROUTINE. An imminent deadline or an immediate "
    "blocker is URGENT. Newsletters, completed work, and FYI updates are ROUTINE."
)
DEADLINE_SPEC = (
    'Decide whether the recipient must act by an explicitly stated deadline today. '
    'Return exactly URGENT or ROUTINE. Return URGENT only if the email gives a '
    'deadline today, such as a clock time today or the end of today. Words such as '
    'now, urgently, immediately, or ASAP do not state a deadline today: return '
    'ROUTINE for those requests unless a deadline today is also stated. Blocked '
    'work without an explicit deadline today, completed work, newsletters, and FYI '
    'updates are ROUTINE.'
)
ROUTER_SPEC = (
    'Classify the topic of this email. Return exactly ACADEMIC, OPERATIONS, or '
    'OTHER. ACADEMIC includes all messages about submissions, submission approvals, '
    'submission access, seminars, teaching, classes, research, and university '
    'administration. A submission or seminar belongs to ACADEMIC even when no paper '
    'or university is named. OPERATIONS includes all messages about software '
    'releases, release approvals, deployments, servers, service failures, and '
    'computing infrastructure. A release belongs to OPERATIONS even when software '
    'is not mentioned. OTHER includes unrelated topics such as meals, travel, '
    'shopping, and social plans. Classify by topic, regardless of urgency. '
    'Examples: "Is the submission form ready?" -> ACADEMIC; "The lecture room has '
    'changed." -> ACADEMIC; "The deployment needs a sign-off." -> OPERATIONS; "Our '
    'server crashed." -> OPERATIONS; "Shall we go hiking?" -> OTHER.'
)
CASES = [
    ("Please approve the submission by 5 p.m. today so we can meet the deadline.", "URGENT"),
    ("Here is next month's seminar schedule for your information.", "ROUTINE"),
    ("The release is waiting on your approval. Please unblock it now.", "URGENT"),
    ("Forwarding the earlier ‘URGENT: signature needed’ email for the record. This was resolved yesterday.", "ROUTINE"),
    ("The submission is ready, but the team cannot send it until you grant access.", "URGENT"),
]


def reference(value):
    path = Path(value)
    if path.suffix == ".json" and path.is_file():
        metadata = json.loads(path.read_text(encoding="utf-8"))
        return metadata["program_id"], metadata
    return value, {}


def emit(value):
    print(json.dumps(value, ensure_ascii=False))


def download_program(program_id, output_path, timeout=120):
    """Wait for an artifact, then save it without replacing an existing file."""
    url = "https://programasweights.com/api/v1/programs/" + quote(program_id, safe="") + "/download"
    deadline = time.monotonic() + timeout
    while True:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError(f"Program artifact was not ready within {timeout} seconds.")
        with urlopen(url, timeout=remaining) as response:
            if response.status != 202:
                payload = response.read()
                break
            retry_after = response.headers.get("Retry-After", "3")
        try:
            delay = float(retry_after)
        except ValueError:
            try:
                ready_at = parsedate_to_datetime(retry_after)
                if ready_at.tzinfo is None:
                    ready_at = ready_at.replace(tzinfo=timezone.utc)
                delay = (ready_at - datetime.now(timezone.utc)).total_seconds()
            except (TypeError, ValueError, OverflowError):
                delay = 3
        remaining = deadline - time.monotonic()
        if remaining > 0:
            time.sleep(min(max(1, delay), remaining))
    with output_path.open("xb") as output:
        output.write(payload)
    return len(payload)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    compile_parser = commands.add_parser("compile", help="Compile a new function and save its ID")
    compile_parser.add_argument("--output", type=Path, required=True)
    compile_parser.add_argument("--policy", choices=["original", "deadline"], default="original")
    compile_parser.add_argument("--kind", choices=["urgency", "router"], default="urgency")
    run_parser = commands.add_parser("run", help="Print actual local outputs next to expected labels")
    run_parser.add_argument("reference", help="Saved JSON, program ID, or local .paw path")
    run_parser.add_argument("--offline", action="store_true")
    run_parser.add_argument("--policy", choices=["original", "deadline"], help="Expected-label policy; overrides saved JSON metadata")
    run_parser.add_argument("--input", help="Test one new email instead of the five examples")
    prompt_parser = commands.add_parser("prompt", help="Run the same task by prompting the unadapted local interpreter")
    prompt_parser.add_argument("--offline", action="store_true")
    prompt_parser.add_argument("--policy", choices=["original", "deadline"], default="original")
    prompt_parser.add_argument("--input")
    prepare_parser = commands.add_parser("prepare", help="Download assets for an existing Hub program")
    prepare_parser.add_argument("reference", help="Saved JSON or program ID")
    download_parser = commands.add_parser("download", help="Save an existing public program as a .paw file")
    download_parser.add_argument("reference", help="Saved JSON or program ID")
    download_parser.add_argument("--output", type=Path, required=True)
    compose_parser = commands.add_parser("compose", help="Combine saved urgency and routing functions")
    compose_parser.add_argument("urgency")
    compose_parser.add_argument("router")
    compose_parser.add_argument("--input", required=True)
    compose_parser.add_argument("--offline", action="store_true")
    args = parser.parse_args()

    if args.command == "download":
        if args.output.exists():
            parser.error("The output exists; choose a new filename.")
        program_id, _ = reference(args.reference)
        size = download_program(program_id, args.output)
        emit({"program_id": program_id, "saved_file": str(args.output), "bytes": size})
        return

    import programasweights as paw

    if args.command == "compile":
        if args.output.exists():
            parser.error("The output exists; choose a new filename to preserve the old program.")
        spec = ROUTER_SPEC if args.kind == "router" else (SPEC if args.policy == "original" else DEADLINE_SPEC)
        started = time.perf_counter()
        program = paw.compile(spec, compiler=COMPILER)
        record = {
            "program_id": program.id,
            "compiler": COMPILER,
            "compiler_snapshot": program.compiler_snapshot,
            "kind": args.kind,
            "policy": args.policy,
            "specification": spec,
            "compiled_at": datetime.now(timezone.utc).isoformat(),
            "compile_seconds": time.perf_counter() - started,
        }
        with args.output.open("x", encoding="utf-8") as output:
            json.dump(record, output, ensure_ascii=False, indent=2)
            output.write("\n")
        emit(record)
        return

    if args.command == "prepare":
        program_id, _ = reference(args.reference)
        paw.prepare_program(program_id)
        emit({"program_id": program_id, "offline_ready": paw.is_offline_ready(program_id)})
        return

    if args.command == "compose":
        urgency_id, _ = reference(args.urgency)
        router_id, _ = reference(args.router)
        destinations = {"ACADEMIC": "research-office", "OPERATIONS": "operations", "OTHER": "general-inbox"}
        with paw.function(urgency_id, offline=args.offline) as urgency:
            label = urgency(args.input).strip()
        if label not in {"URGENT", "ROUTINE"}:
            raise ValueError("Unexpected urgency label: " + repr(label))
        with paw.function(router_id, offline=args.offline) as route:
            destination = route(args.input).strip()
        if destination not in destinations:
            raise ValueError("Unexpected routing label: " + repr(destination))
        emit({"input": args.input, "urgency": label, "destination": destinations[destination], "queue": "priority" if label == "URGENT" else "ordinary"})
        return

    started = time.perf_counter()
    if args.command == "prompt":
        policy = args.policy
        fn = paw.function(None, interpreter="Qwen/Qwen3-0.6B", offline=args.offline)
        program_id = None
    else:
        program_id, metadata = reference(args.reference)
        if metadata.get("kind", "urgency") != "urgency":
            parser.error("Use an urgency program for these test cases.")
        policy = args.policy or metadata.get("policy")
        if policy is not None and policy not in {"original", "deadline"}:
            parser.error("Saved policy must be original or deadline; use --policy to select it.")
        if args.input is None and policy is None:
            parser.error("Built-in test cases require --policy original or --policy deadline when the reference has no saved policy.")
        fn = paw.function(program_id, offline=args.offline)
    emit({"mode": args.command, "program": program_id, "load_seconds": time.perf_counter() - started})
    cases = [(args.input, None)] if args.input is not None else CASES
    results = []
    try:
        for email, expected in cases:
            if expected is not None and policy == "deadline":
                expected = "URGENT" if email == CASES[0][0] else "ROUTINE"
            model_input = email
            if args.command == "prompt":
                spec = SPEC if policy == "original" else DEADLINE_SPEC
                model_input = spec + "\n\nEmail:\n" + email + "\n\nLabel:"
            started = time.perf_counter()
            actual = fn(model_input).strip()
            result = {"input": email, "expected_label": expected, "actual_output": actual,
                      "valid_label": actual in {"URGENT", "ROUTINE"},
                      "matches_expected": actual == expected if expected is not None else None,
                      "call_seconds": time.perf_counter() - started}
            results.append(result)
            emit(result)
    finally:
        fn.close()
    invalid = sum(not result["valid_label"] for result in results)
    mismatches = sum(result["matches_expected"] is False for result in results)
    emit({"summary": True, "cases": len(results), "policy": policy,
          "checked_cases": sum(result["expected_label"] is not None for result in results),
          "matches": sum(result["matches_expected"] is True for result in results),
          "mismatches": mismatches, "invalid_labels": invalid,
          "passed": invalid == 0 and mismatches == 0})
    if invalid or mismatches:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
