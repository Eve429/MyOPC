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
- `simple.gds`: user-editable example, DBU 0.001 um, one top (`TOP`), one layer (1/0); its initial 10-shape baseline is historical and no longer used by deterministic tests.
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
- Million logical-instance AREF: 25 ROI polygons, 0.1058 ms median query+clip, 0.1164 ms p95, 0.48 MB RSS delta, 10.01 ms file open.
- 100,000-edge local grid: 425.48 ms one-time build, 0.0208 ms median indexed query versus 0.3325 ms brute scan, 15.99x speedup, exact results.
- 2,048 × 2,048 native grayscale raster: 482.84 ms, 4.32 MB RSS delta, exact aligned-area coverage.

## Real Layout Smoke Test
- Read-only validation of user-owned, untracked `TestReticle/gcd_45nm.gds` succeeded; the file was not added to Git.
- Selected top Cell: `TOP`; DBU: 0.0001 um; Layer: 11/0.
- Full top bbox: `[11400, 13150, 317300, 308850]` DBU.
- Materialized 1,776 polygons with total area 28,594,652,500 DBU².
- NumPy boundary conversion produced 21,590 vertices, 1,776 rings, and 21,590 closed edges.
- Optional diagnostics agreed on 1,776 polygon-like objects and reported zero Text/Edge/other objects.
- End-to-end direct CLI wall time measured by PowerShell was 461.001 ms, including interpreter and imports.

## Raster Feature Design
- Before phase 8, the code could query and clip planner ROIs and write GDS/OASIS patches, but had no pixel renderer or PNG output.
- KLayout 0.30.10 `Region.rasterize(origin, pixel_size, nx, ny)` returns exact covered area for each pixel as a bottom-up Python float matrix.
- Native rasterization does not apply merged semantics; the local Region must be merged first so overlaps are not counted twice.
- A PNG array must be vertically flipped because image row zero is top while KLayout raster row zero starts at the lower-left origin.
- Pillow and Matplotlib are installed; Pillow is sufficient for grayscale PNG save and optional system-viewer display without a GUI framework.
- `gcd_45nm.gds` full bbox is 305,900 × 295,700 DBU. At DBU 0.1 nm and 5 nm/pixel it becomes exactly 6,118 × 5,914 = 36,181,852 pixels.
- `.vscode/` and `TestReticle/gcd_45nm.gds` are user-owned untracked items and must remain outside Git commits.
- User confirmed the tracked `TestReticle/simple.gds` worktree change is intentional; preserve it and exclude it from this feature commit.
- Exact automated baselines now use generated temporary GDS files, so user edits to example reticles do not invalidate deterministic tests.
- Implemented two-dimensional native raster tiles, so the one-million temporary-pixel bound also holds for images wider than one million pixels.
- Targeted raster and direct CLI verification passed: 10 tests in 1.23 seconds; Ruff checks passed.
- Full real-layout raster succeeded for `gcd_45nm.gds` Layer 11/0 at 5 nm/pixel: 6,118 × 5,914 pixels, 4.663 seconds end-to-end, 149.36 MB peak process RSS, and a 78,464-byte PNG.
- Visual inspection confirmed top-up orientation and a clear white-mask/black-background rendering across the complete layout.
- The generated real-layout preview is an ignored test artifact at `.benchmarks/gcd_45nm_full_5nm.png`; source GDS and preview remain outside Git.
