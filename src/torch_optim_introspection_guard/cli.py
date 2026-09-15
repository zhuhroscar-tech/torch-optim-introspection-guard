"""Command-line interface: run the from-scratch diagnosis of PyTorch's
get_optimizer_state_dict() step-counter mutation bug against the
currently installed torch build, using the shared semantic-color design
system."""
from __future__ import annotations

import argparse
import json
import sys

from .style import print_fields, resolve_style, section, status_headline


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        prog="torch-optim-introspection-guard",
        description=(
            "Diagnose whether the currently installed torch build's "
            "get_optimizer_state_dict() silently mutates optimizer state "
            "(pytorch/pytorch#164929) and verify a safe read-only wrapper "
            "neutralizes it, for every common optimizer family, on THIS "
            "host's actual installed torch version -- never trusts the "
            "upstream issue's reported version alone."
        ),
    )
    parser.add_argument("--json", action="store_true", help="emit machine-readable JSON instead of text")
    parser.add_argument("--no-color", action="store_true", help="disable ANSI color even on a TTY")
    parser.add_argument("--seed", type=int, default=20260915, help="RNG seed for the diagnosis (default 20260915)")
    parser.add_argument("--version", action="store_true", help="print version and exit")
    args = parser.parse_args(argv)

    if args.version:
        from . import __version__

        print(f"torch-optim-introspection-guard {__version__}")
        return 0

    from .core import TorchUnavailableError, diagnose

    try:
        report = diagnose(seed=args.seed)
    except TorchUnavailableError as exc:
        if args.json:
            print(json.dumps({"error": str(exc)}, indent=2))
        else:
            style = resolve_style(no_color_flag=args.no_color)
            print(status_headline(style, "fail", f"torch unavailable: {exc}"))
        return 2

    if args.json:
        print(json.dumps(report, indent=2))
        return 0 if report["guard_fully_effective"] else 1

    style = resolve_style(no_color_flag=args.no_color)
    print_fields([("torch version", report["torch_version"]), ("tracking issue", report["issue_url"])])

    if report["any_bug_present"]:
        print(status_headline(style, "warn", "bug reproduced on this host's installed torch build"))
    else:
        print(
            status_headline(
                style,
                "info",
                "bug NOT reproduced on this host's installed torch build (fixed upstream, or optimizer set unaffected)",
            )
        )

    if report["guard_fully_effective"]:
        print(status_headline(style, "ok", "safe_get_optimizer_state_dict() neutralizes it in every tested optimizer"))
    else:
        print(status_headline(style, "fail", "guard did NOT neutralize the bug for at least one optimizer"))

    section("per-optimizer results")
    for r in report["results"]:
        flag = "bug" if r["bug_present"] else "clean"
        guard_flag = "guard-ok" if r["guard_effective"] else "GUARD-FAILED"
        print_fields(
            [
                (
                    r["name"],
                    f"{flag:5s}  max|diff| raw={r['max_abs_diff_raw']:.3e}  "
                    f"guarded={r['max_abs_diff_guarded']:.3e}  {guard_flag}",
                )
            ]
        )

    return 0 if report["guard_fully_effective"] else 1


if __name__ == "__main__":
    sys.exit(main())
