# Findings

## Repository
- Repository is on branch `master` and has an `origin`; no remote push is authorized.
- Existing implementation directories are empty; current tracked assets are GDS fixtures and a KLayout notebook.
- `TestReticle/gcd_45nm.gds` is untracked user data and must be preserved.

## Environment
- Canonical interpreter: `D:\app\miniforge\envs\myopc\python.exe` (Python 3.12.0).
- Installed: KLayout 0.30.10, NumPy 2.5.1, psutil 7.2.2.
- Missing at discovery: pytest and pytest-cov.
- Hardware: AMD Ryzen 7 4800H, 8 cores / 16 threads, 15.37 GB RAM.

## Existing GDS Baselines
- `simple.gds`: DBU 0.001 um, one top (`TOP`), one layer (1/0), 10 polygon-like shapes after text is ignored.
- `JustPoly.gds`: one top, two polygons, negative coordinates.
- `test1.gds`: two tops (`test`, `cell1`), layers 2/0, 1/0, 3/0; `cell1` materializes 22, 13, and 1 polygon-like shapes respectively.
- A KLayout prototype queried 25 shapes from a one-million-instance AREF in about 0.3 ms and clipped them in about 2.3 ms on this machine.

## Architecture Constraints
- Never flatten the full layout by default.
- Never expose all polygons as Python lists.
- Use one lazy query per cell/layer/ROI and materialize only local Region batches.
- KLayout Region converts boxes and paths to polygons and ignores text/edges.
- Convert Region to compact NumPy contour/edge arrays only at the repeated-compute boundary.
- Source layout remains immutable; patch ownership is resolved in global DBU coordinates.

## Final Benchmark Baseline
- Million logical-instance AREF: 25 ROI polygons, 0.116 ms median query+clip, 0.183 ms p95, 0.48 MB RSS delta, 12.03 ms file open.
- 100,000-edge local grid: 426.20 ms one-time build, 0.0214 ms median indexed query versus 0.4436 ms brute scan, 20.73x speedup, exact results.
