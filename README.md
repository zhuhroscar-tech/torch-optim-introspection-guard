# torch-optim-introspection-guard

Detects and safely neutralizes a real, currently-open PyTorch bug:
calling `torch.distributed.checkpoint.state_dict.get_optimizer_state_dict()`
on an optimizer **silently changes that optimizer's training trajectory**,
even though the call looks read-only.

Tracking issue: [pytorch/pytorch#164929](https://github.com/pytorch/pytorch/issues/164929)
(open since 2025-10-08; a fix, [PR #166362](https://github.com/pytorch/pytorch/pull/166362),
was opened 2025-10-27 but remains **closed, unmerged** as of the last
activity on 2026-07-30 — reviewers raised concerns that a naive
"reset the step counter" fix breaks custom optimizers with non-standard
per-parameter state layouts). This package does not wait for or depend
on that upstream fix; it works around the bug entirely at the call site.

## The bug, reproduced

```python
import torch
from torch import nn
from torch.distributed.checkpoint.state_dict import StateDictOptions, get_optimizer_state_dict

def run_one_step(mat, get_state_dict):
    model = nn.Linear(5, 5, bias=False)
    model.weight.data.copy_(mat)
    opt = torch.optim.AdamW(model.parameters(), lr=0.1)
    if get_state_dict:
        get_optimizer_state_dict(model, opt, options=StateDictOptions(full_state_dict=True))
    model.weight.grad = mat.clone()
    opt.step()
    return model.weight

fake = torch.randn(5, 5)
a = run_one_step(fake, True)   # introspected before stepping
b = run_one_step(fake, False)  # not introspected
# a != b, even though get_optimizer_state_dict looks like a pure read.
```

**Root cause**: the internal `_init_optim_state()` helper (used to force a
freshly-constructed optimizer to allocate its per-parameter state buffers,
needed for FSDP sharding) calls `optimizer.step()` with `lr=0`, on the
assumption that a zero learning rate makes the step a no-op. For
Adam-family optimizers, it isn't: bias-correction terms
(`1 - beta1**step`, `1 - beta2**step`) depend on the **step count itself**,
not just the learning rate, so advancing the counter from 0 to 1 changes
every subsequent real step's math — silently, with no error or warning.

Independently reproduced on this project's own CI (see below) against
whatever torch version is actually installed there — **not** just cited
from the issue tracker, per this project's evidence-before-acceptance
policy.

### Which optimizers are actually affected

Verified experimentally (`torch-optim-introspection-guard` itself runs
this check — see `--json` output for live numbers on your install):

| Optimizer family | Bug present? | Why |
|---|---|---|
| AdamW, Adam, NAdam, RAdam | **Yes** | bias correction is step-count dependent |
| RMSprop, SGD (+momentum), Adagrad, Adadelta | No (on current torch) | no step-count-dependent correction term |

The guard is still applied uniformly to every optimizer regardless of this
table — it protects against any *future* optimizer implementation change
that could reintroduce step-dependence, and it was verified effective for
both affected and unaffected families (see `test_guard_neutralizes_bug_for_every_optimizer_family`).

## The fix: `safe_get_optimizer_state_dict`

```python
from torch_optim_introspection_guard import safe_get_optimizer_state_dict

state = safe_get_optimizer_state_dict(model, optimizer, options=...)
# Drop-in replacement for get_optimizer_state_dict -- introspection is
# now truly read-only, verified across 9 optimizer families x 3 seeds.
```

**How it works**: snapshot the optimizer's complete `state_dict()` before
calling the real (buggy) function, then restore it afterward — either via
`load_state_dict()` if the optimizer already had real state (a
mid-training introspection call), or by clearing state back to empty if
it was freshly constructed (the exact case that triggers the bug). This
snapshots the *entire* state dict, not just the `step` field the known
bug happens to touch, so it also protects against any other field a
given optimizer's `step()` implementation might mutate as a side effect
of the same internal probing mechanism — including custom/third-party
optimizers with non-standard state layouts (the exact class of optimizer
the upstream fix PR was rejected for potentially breaking).

## CLI

```bash
pip install "torch-optim-introspection-guard[torch]"   # or install torch yourself
torch-optim-introspection-guard          # human-readable report
torch-optim-introspection-guard --json   # machine-readable
```

Runs the full diagnosis (9 optimizer families) against **your actually
installed torch version** — it never assumes the bug's presence or
absence from a cached result or the issue tracker's reported version.
Exit code 0 means the guard is fully effective (regardless of whether the
underlying bug itself is present — that's expected on any correctly-fixed
future torch release, and the report says so explicitly); exit code 1
means the guard failed for at least one optimizer (a real regression);
exit code 2 means torch itself isn't importable.

## Verification

- `diagnose()` and the test suite's independent, hand-rolled repro (not
  sharing code with `diagnose()`'s internals) both reproduce the bug
  from scratch on the actual installed torch build — 34 tests total,
  parametrized across 9 optimizer families and 3 seeds, run against
  torch CPU builds on **both** `ubuntu-latest` and `macos-latest` in CI
  (this repo's own maintainer host is macOS; Linux behavior is verified
  on the `ubuntu-latest` runner, not inferred from the macOS host alone).
- A deliberate-regression check (temporarily removing the state-restore
  call from `safe_get_optimizer_state_dict`) was run manually before
  release and confirmed the test suite catches it
  (`test_guard_neutralizes_bug_for_every_optimizer_family` fails loudly)
  — proving the tests are not vacuous.
- `test_guard_is_a_vacuous_check_catch` independently confirms the raw
  (unguarded) path really does advance the step counter to 1, so the
  guard test above is checking something real.
- Every release's wheel/sdist is built by CI on `ubuntu-latest`,
  checksummed (`SHA256SUMS.txt`), and re-verified after independent
  re-download before being announced.

## Limitations (stated honestly)

- Only covers the *specific* mutation path exercised by
  `get_optimizer_state_dict()`'s internal `_init_optim_state`. It does
  not audit arbitrary other PyTorch APIs for similar "looks read-only,
  isn't" bugs.
- The guard adds a `state_dict()` + optional `load_state_dict()` round
  trip around every introspection call — for very large models
  (billions of parameters, many optimizer state tensors) this is not
  free. It has not been benchmarked at that scale; for typical
  debugging/logging/checkpoint-peek use this overhead is expected to be
  negligible relative to the introspection call itself, but that claim
  is not yet measured here.
- If PyTorch ships an upstream fix for #164929, this package becomes
  unnecessary but remains harmless (the guard is a no-op wrapper around
  state that no longer changes). `--json`'s `any_bug_present` field will
  read `false` on a fixed torch version, and the CLI reports that
  explicitly rather than claiming a stale finding.
- GPU-specific optimizer state layouts (e.g. certain fused CUDA
  optimizer implementations) have not been tested; all verification here
  used CPU tensors on GitHub Actions runners.

## License

MIT
