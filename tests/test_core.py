"""Tests for torch_optim_introspection_guard.core.

These tests require torch to be installed (skipped otherwise, so this
package's own test suite -- and CI on both ubuntu-latest and
macos-latest -- degrades gracefully on a torch-less environment rather
than failing outright, matching how the CLI itself reports
TorchUnavailableError instead of crashing).

The regression test's independent oracle is NOT "trust core.py's own
diagnose() output" -- it is a from-scratch, hand-written repro identical
in spirit to the upstream issue's own minimal example but re-derived
here independently, run against three separate seeds, so a
future accidental revert of the guard implementation (e.g. someone
"simplifies" safe_get_optimizer_state_dict to only reset the step
counter instead of the full state) would be caught even if it happened
to still pass a single-seed check.
"""
from __future__ import annotations

import copy

import pytest

torch = pytest.importorskip("torch")
from torch import nn  # noqa: E402
from torch.distributed.checkpoint.state_dict import (  # noqa: E402
    StateDictOptions,
    get_optimizer_state_dict,
)

from torch_optim_introspection_guard.core import (  # noqa: E402
    PROBE_OPTIMIZERS,
    diagnose,
    safe_get_optimizer_state_dict,
)


def _independent_repro(optimizer_cls, optimizer_kwargs, seed, use_guard):
    """A hand-rolled repro, independent of core.py's internal helpers
    (_build_model_and_optimizer / _run_probe), so it serves as a real
    oracle rather than re-testing the same code path against itself."""
    torch.manual_seed(seed)
    weight = torch.randn(4, 4)

    def one_run(introspect):
        model = nn.Linear(4, 4, bias=False)
        model.weight.data.copy_(weight)
        opt = optimizer_cls(model.parameters(), **optimizer_kwargs)
        if introspect:
            if use_guard:
                safe_get_optimizer_state_dict(model, opt, options=StateDictOptions(full_state_dict=True))
            else:
                get_optimizer_state_dict(model, opt, options=StateDictOptions(full_state_dict=True))
        model.weight.grad = weight.clone()
        opt.step()
        return model.weight.detach().clone()

    baseline = one_run(introspect=False)
    introspected = one_run(introspect=True)
    return baseline, introspected


# --- Known-affected optimizer families (Adam-variant bias correction is
# step-count dependent) ------------------------------------------------

ADAM_FAMILY = [
    (torch.optim.AdamW, {"lr": 0.1}),
    (torch.optim.Adam, {"lr": 0.1}),
    (torch.optim.NAdam, {"lr": 0.1}),
    (torch.optim.RAdam, {"lr": 0.1}),
]

# --- Optimizer families NOT expected to depend on step count for this
# specific bug (no step-count-dependent bias correction term) ----------

NON_ADAM_FAMILY = [
    (torch.optim.SGD, {"lr": 0.1, "momentum": 0.9}),
    (torch.optim.Adagrad, {"lr": 0.1}),
]


@pytest.mark.parametrize("optimizer_cls,kwargs", ADAM_FAMILY)
@pytest.mark.parametrize("seed", [1, 2026, 999999])
def test_bug_reproduces_on_adam_family_unguarded(optimizer_cls, kwargs, seed):
    """Regression oracle: this test FAILS (bug not reproduced) if a future
    torch release fixes #164929 upstream, or if the optimizer's internals
    change such that lr=0 truly is a no-op. That is itself useful
    information (see the CLI's --json 'any_bug_present' field), so this
    test is intentionally allowed to start failing -- a failure here is a
    signal to re-run diagnose() and update the README's live-verification
    claim, not a code defect to silence."""
    baseline, unguarded = _independent_repro(optimizer_cls, kwargs, seed, use_guard=False)
    assert not torch.allclose(baseline, unguarded), (
        f"expected {optimizer_cls.__name__} to diverge when introspected "
        "unguarded (pytorch/pytorch#164929) -- if this now passes, the bug "
        "may have been fixed upstream; re-run `torch-optim-introspection-guard "
        "--json` and update README before treating this as a real failure"
    )


@pytest.mark.parametrize("optimizer_cls,kwargs", ADAM_FAMILY + NON_ADAM_FAMILY)
@pytest.mark.parametrize("seed", [1, 2026, 999999])
def test_guard_neutralizes_bug_for_every_optimizer_family(optimizer_cls, kwargs, seed):
    """The actual product guarantee: with the guard, introspection must be
    truly read-only for every optimizer family, whether or not that
    family is affected by the underlying bug in the first place."""
    baseline, guarded = _independent_repro(optimizer_cls, kwargs, seed, use_guard=True)
    assert torch.allclose(baseline, guarded, atol=1e-6), (
        f"safe_get_optimizer_state_dict failed to neutralize introspection "
        f"side effects for {optimizer_cls.__name__} (seed={seed})"
    )


def test_guard_preserves_prior_state_on_midtraining_introspection():
    """The guard must also work correctly on a NON-fresh optimizer (one
    that already has real accumulated state from prior real steps), not
    just a freshly constructed one -- this is the "had_state" branch in
    safe_get_optimizer_state_dict that restores via load_state_dict
    rather than clearing state."""
    torch.manual_seed(7)
    weight = torch.randn(4, 4)

    model = nn.Linear(4, 4, bias=False)
    model.weight.data.copy_(weight)
    opt = torch.optim.AdamW(model.parameters(), lr=0.1)

    # Two real steps to build up genuine (non-empty) Adam state.
    for _ in range(2):
        model.weight.grad = weight.clone()
        opt.step()

    state_before = copy.deepcopy(opt.state_dict())

    safe_get_optimizer_state_dict(model, opt, options=StateDictOptions(full_state_dict=True))

    state_after = opt.state_dict()
    for group_before, group_after in zip(state_before["param_groups"], state_after["param_groups"]):
        assert group_before == group_after
    for key in state_before["state"]:
        for field in state_before["state"][key]:
            before_val = state_before["state"][key][field]
            after_val = state_after["state"][key][field]
            if torch.is_tensor(before_val):
                assert torch.equal(before_val, after_val), f"field {field} mutated by introspection"
            else:
                assert before_val == after_val, f"field {field} mutated by introspection"


def test_guard_is_a_vacuous_check_catch():
    """Sanity check that the guard test above is not vacuously true: if we
    deliberately DON'T restore state (simulate what the raw/buggy path
    does), the step counter must actually differ, proving the assertions
    in the guard test are exercising real state."""
    torch.manual_seed(7)
    weight = torch.randn(4, 4)
    model = nn.Linear(4, 4, bias=False)
    model.weight.data.copy_(weight)
    opt = torch.optim.AdamW(model.parameters(), lr=0.1)

    step_before = None
    get_optimizer_state_dict(model, opt, options=StateDictOptions(full_state_dict=True))
    for p_state in opt.state.values():
        if "step" in p_state:
            step_before = p_state["step"].item() if torch.is_tensor(p_state["step"]) else p_state["step"]
            break
    assert step_before == 1, "expected the unguarded probe step to have advanced the step counter to 1"


def test_probe_optimizers_registry_is_nonempty_and_constructible():
    """PROBE_OPTIMIZERS must actually resolve to real torch.optim classes;
    this catches a typo'd factory name that diagnose() would otherwise
    silently AttributeError on deep inside a loop."""
    assert len(PROBE_OPTIMIZERS) >= 5
    for spec in PROBE_OPTIMIZERS:
        cls = getattr(torch.optim, spec.factory)
        model = nn.Linear(3, 3, bias=False)
        opt = cls(model.parameters(), **spec.kwargs)
        assert opt is not None


def test_diagnose_end_to_end_reports_guard_fully_effective():
    """End-to-end smoke test through the real public API (not the
    internal helpers), matching exactly what the CLI calls."""
    report = diagnose(seed=42)
    assert report["torch_version"] == torch.__version__
    assert report["issue_url"].endswith("164929")
    assert len(report["results"]) == len(PROBE_OPTIMIZERS)
    assert report["guard_fully_effective"] is True, (
        "guard must neutralize the bug for every probed optimizer; a False "
        "here means safe_get_optimizer_state_dict regressed"
    )
