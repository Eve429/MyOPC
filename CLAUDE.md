# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this project is

MyOPC is a high-performance **hierarchical layout and geometry foundation for OPC** (Optical Proximity Correction, used in semiconductor lithography). The implemented layer today is the **MB-OPC frontend** (`opc/input/edge/`): it turns a GDS/OASIS layout into compact, batch-oriented numpy structures that a future model-based solver can iterate on. The actual optical model, optimizer, and lithography/evaluation layers are **not yet implemented** — those directories (`lithography/`, `evaluation/`, `opc/iteration/`) are intentionally empty placeholders.

Runtime: Python ≥ 3.12. Hard dependencies: `klayout`, `numpy`, `pillow`. The verified interpreter is the `myopc` conda env (`D:\app\miniforge\envs\myopc\python.exe`). **The project is not installed as a package** — it runs via direct Python file entries from the repo root (see `run_mbopc_frontend.py`, `run_layout_geometry.py`).

## Authoritative project rules (read these first)

`AGENTS.md` is the binding development manual. The rules below are non-obvious and frequently violated — follow them exactly:

- **All source comments and docstrings must be written in Chinese (中文).** Comments must *start* with a Chinese word, not an English API name (e.g. write "原生栅格接口" not "rasterize"); technical terms may appear mid-sentence. Comment the *why* — coordinate directions, data invariants, performance paths, memory bounds, boundary ownership, exception reasons — placed before compact logic blocks, not line-by-line syntax restatement.
- **Do not modify `layout/` or `geometry/` without explicit per-task user approval.** New OPC work must build only on their existing public interfaces. If that is impossible, stop and state the necessity, impact, and minimal change before touching them. This was escalated from user feedback and promoted into `AGENTS.md` (`.learnings/LEARNINGS.md` `LRN-20260809-005`).
- **No automatic formatting.** Preserve the hand-tuned compact layout. The `[tool.black]` block in `pyproject.toml` is *not* used — see `ERR-20260809-011`. Quality gates are Ruff rule-checks, `compileall`, and tests only.
- **Every bug fix ships a reproducible regression test**, and after fixing you must search call sites and delete any function/wrapper/branch/variable that existed only to serve the old bug.
- **No speculative abstractions** — new interfaces, registries, or empty directories must have a current caller; do not scaffold for hypothetical future methods.
- **Final-delivery audit** (before declaring done): full diff review, unused-function scan, duplicate-implementation check, exception-entry check, coverage-missed-branch check. Record the cleanup in the relevant report under `doc/`.
- **Each feature must update**: dev manual, test manual, the specialized dev/test report, and the planning files `task_plan.md` / `findings.md` / `progress.md`. Use the `planning-with-files` workflow for multi-step work.
- **Git**: local commits only at key stages; never push to remote without explicit authorization. Exclude user GDS files, images, and unrelated worktree changes from feature commits. Recent commits use conventional prefixes (`docs:`, `refactor:`, `perf:`).
- Track errors/learnings/feature-requests in `.learnings/` with the existing `ERR-`/`LRN-`/`FEAT-` id convention.

## Commands

The docs use PowerShell; this environment is bash on Windows, so use forward slashes. Substitute the project's `python` (the `myopc` env).

```bash
# Fastest functional smoke test (synthetic multi-polygon case; writes to .benchmarks/mbopc_frontend_demo/)
python run_mbopc_frontend.py

# Real GDS verification
python run_mbopc_frontend.py TestReticle/gcd_45nm.gds --layer 11/0 --grid 2 1 --skip-geometry-suite --json

# Tests
python -m pytest -q                              # full repo regression
python -m pytest -q tests/opc                    # OPC-only
python -m pytest -q tests/opc/test_fragment.py   # single file
python -m pytest -q tests/opc/test_fragment.py -k test_name   # single test
python -m pytest --cov=opc --cov-branch --cov-report=term-missing -q tests/opc   # coverage

# Lint — NOTE the explicit scope: never run `ruff check .` (the pre-existing Test/klayout.ipynb user notebook is out of scope)
python -m ruff check layout geometry opc tests benchmarks/benchmark_layout_geometry.py benchmarks/benchmark_mbopc_frontend.py run_layout_geometry.py run_mbopc_frontend.py

# Byte-compile gate
python -m compileall -q layout geometry opc tests run_layout_geometry.py run_mbopc_frontend.py

# Strict performance benchmark (returns non-zero if regressions; gates memory savings ≥40%, lookup/materialize timing, halo not degrading to dense matrix)
python benchmarks/benchmark_mbopc_frontend.py --strict
```

`pyproject.toml` configures pytest (`testpaths=["tests"]`, `addopts="-ra"`) and coverage (`branch=true`, `source=["layout","geometry","opc"]`).

## Architecture

### Layered dependency (one-directional, never reversed)

```
layout  ->  geometry  ->  opc.input  ->  opc.input.edge
                                     <- opc.iteration.<method>   (future)
                                     <- lithography, evaluation  (future)
```

- **`layout/`** — read-only hierarchical GDS/OASIS via `LayoutDB.open(...)`. ROI/layer queries (`ShapeQuery`) return native KLayout `Region`s. Never flattens hierarchy; materializes only shapes intersecting a local ROI.
- **`geometry/`** — method-agnostic Region↔contour↔edge conversion, Patch clip/stitch (`PatchSet`), rasterization.
- **`opc/input/`** — shared by MB-OPC and future ILT: `PhysicalMask`, `RectilinearCoreGrid`, sampling templates.
- **`opc/input/edge/`** — MB-OPC specific: fragmentation, ownership, update merge, contour reconstruction, sampling, artifacts, visualization, verification. Public entry point is `prepare_problem()`.
- **`opc/iteration/`, `lithography/`, `evaluation/`** — empty. The future solver loop belongs in `opc.iteration.<method>`; the optical model in `lithography`; metrics in `evaluation`. Do **not** put optical/optimizer code in `opc.input`.

`doc/function_call_architecture.md` is the authoritative call-graph and data-flow reference — read sections 2–4 and 10 before touching the frontend.

### `prepare_problem()` is the architecture center

`prepare_problem(batch, layer, config, cores)` runs **once per layer/ROI/config** and produces an `MBOPCProblem` (frozen dataclass) holding four *fixed reference* objects reused across the whole optimization:

| Object | Holds | Lifetime |
|---|---|---|
| `PhysicalMask` | merged Region + `ContourBatch` + `EdgeBatch` | fixed |
| `SegmentBatch` | edge-id + `t0/t1` params + normals + stable 128-bit keys (splitmix64) | fixed |
| `OwnershipBatch` | unique `owner` per segment + CSR halo `memberships` | fixed |
| `BoundarySampleTemplate` | line indices / tangential pos / normal offset (no coordinates) | fixed |

Per-iteration state is just a `displacements` numpy array. The intended hot path: `merge_owner_updates()` → `SegmentBatch.materialize()` → `sample_lines()` → (future) optical model → repeat. Reuse the sorted-key lookup index and output buffers; never rebuild per iteration. Section 10 of the architecture doc gives the solver skeleton.

### Native Region lifetime (critical invariant)

`ShapeQuery.materialize()` returns a KLayout `Region` that is still tied to the open database. Therefore **`materialize()` and `prepare_problem()` must both run inside the `with LayoutDB.open(...)` context.** After `prepare_problem` builds the independent numpy/Region data, the DB may close and all subsequent iteration/output never touches the source file. Violating this yields empty/zero-count Regions (see `.learnings/ERRORS.md` `ERR-...025`, `ERR-...009`).

### Reference vs. iteration state

`MBOPCProblem` is frozen but its numpy arrays are mutable memory. Treat `problem.*` reference arrays as **read-only**; keep all optimization state in a separate `displacements` array. Reconstruction (`reconstruct_region`) is only for final/diagnostic output — do not put it in the per-iteration hot path.

## Testing and fixtures

- `TestReticle/*.gds` (`simple.gds`, `gcd_45nm.gds`, `JustPoly.gds`, `test1.gds`) are **user-editable regression data**. Tests must **not** hardcode their exact coordinates/counts — use generated/deterministic GDS instead (`ERR-20260809-016`). These files must be preserved and excluded from feature commits.
- `Test/klayout.ipynb` is a pre-existing user notebook — out of scope, do not modify.
- New geometry logic must assert together: zero-displacement XOR == 0, segment keys unique, segment length ≤ config, normals are unit vectors, owner unique.
- Test dirs use package markers and relative helpers (`tests/geometry/helpers.py`, `tests/fixtures/`); `conftest.py` provides `project_root` and `reticle_dir` session fixtures.

## Where to look

| Need | Look in |
|---|---|
| Binding rules | `AGENTS.md` |
| Call graph / data flow / solver integration | `doc/function_call_architecture.md` |
| Dev & test commands, deliverable definitions | `doc/development_manual.md`, `doc/test_manual.md` |
| Per-feature reports | `doc/*_development_report.md`, `doc/*_test_report.md` |
| Error/learning/feature history | `.learnings/` |
| Active planning | `task_plan.md`, `findings.md`, `progress.md` |
