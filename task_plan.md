# Layout / Geometry Development Plan

## Goal
Build a high-performance, extensible layout and geometry foundation for multiple OPC methods, with complete tests, benchmarks, local Git milestone commits, and reports under `doc/`.

## User Requirements
- Preserve speed: load each layout once, retain hierarchy, query by ROI, batch all Python/native crossings.
- Keep RB-OPC, MB-OPC, ILT, SRAF, and future backends compatible.
- Use compact code formatting with file comments, function docstrings, and comments on critical paths.
- All source comments, module docstrings, function docstrings, and configuration comments must be Chinese; critical-path comments must explain performance/correctness intent in detail.
- Provide a repository-root Python entry point that works without installing this project as a package.
- Treat source layouts as immutable and export geometry patches.
- Test a single polygon crossing adjacent core boundaries.
- Commit key milestones locally; never push.

## Phases
| Phase | Status | Deliverable |
|---|---|---|
| 1. Project foundation | complete | Packaging, contracts, planning records, test setup |
| 2. Layout layer | complete | LayoutDB, hierarchy, layers, lazy ROI queries |
| 3. Geometry layer | complete | Region operations, contours, edges, validation, local index |
| 4. Patch/output | complete | Ownership clipping, conflicts, GDS/OASIS export |
| 5. Verification | complete | Unit/integration/regression/performance tests |
| 6. Reports | complete | Development and test reports under `doc/` |
| 7. Simplify and harden audit | complete | Removed overdesign, error-driven special cases, cycles, and dead abstractions; reran all verification |
| 8. Native raster foundation | complete | Single-layer grayscale coverage rasterization with bounded temporary memory |
| 9. PNG display and CLI | complete | Callable render function, atomic PNG save, optional viewer, direct CLI options |
| 10. Raster verification | complete | Unit/integration/performance/real-layout tests and report updates |
| 11. Raster simplify audit | complete | Reviewed abstraction, hot-path allocations and error branches; removed duplicate CLI clipping and passed all gates |

## Acceptance Gates
- Existing GDS regression counts and hierarchy transforms are correct.
- Generated hierarchy, multi-layer, path/text, holes, transforms, and boundary cases pass.
- A polygon crossing adjacent core boxes is clipped into complementary patches without loss or overlap.
- Patch GDS/OASIS round trips have XOR area zero.
- New `layout/` and `geometry/` code reaches at least 90% test coverage.
- Million-instance ROI benchmark does not flatten the full hierarchy and stays within agreed limits.
- Git working tree retains unrelated user files and all milestone commits remain local.
- A planner `DbuBox` can render one Layer to a top-up grayscale PNG at 5 nm/pixel.
- Raster values equal exact polygon area coverage, including holes and partial boundary pixels.
- Full `gcd_45nm.gds` Layer 11/0 bbox renders within the 64-million-pixel guard.
- Rasterization uses bounded horizontal stripes and never loops over Polygon coordinates in Python.

## Errors Encountered
| Error | Attempt | Resolution |
|---|---|---|
| Earlier Git probe ran before repository existed | 1 | Repository now exists; verify status before every commit |
| Python stdin mangled a Chinese HTML path | 1 | Pass Unicode paths through PowerShell environment variables |
| ROI Region materialized 11 shapes while diagnostics found 10 polygon-like + 1 text | 1 | Set native iterator shape flags to Box/Path/Polygon; avoids Python filtering and fixes count |
| Geometry tests failed collection due to relative imports | 1 | Added package markers for `tests` and `tests/geometry` |
| Combine test expected raw area sum from Region.area() | 1 | Assert raw count/non-merged state; Region area correctly follows set semantics |
| Generated transform fixture expected geometry-only layout bbox and two ROI shapes | 1 | Corrected full bbox to include Text and ROI count to include Box/Path/Polygon |
| Hierarchical ROI diagnostics counted zero Text objects | 1 | Diagnostic-only iterator now uses touching semantics for zero-area objects; materialization remains overlapping |
| CLI comma-separated negative Box was parsed as an option | 1 | Simplified CLI to four integer arguments; removed the comma parser instead of adding a special case |
| Ruff audit command was unavailable | 1 | Added Ruff as an optional development dependency; runtime remains unaffected |
| Ruff found 10 maintainability findings | 1 | Manually simplified imports, constants, casts, test defaults, and context managers; no auto-fix used |
| Ruff formatter would expand the requested compact style | 1 | Keep compact manual layout; use Ruff rules, compileall, and tests as quality gates instead |
| Tracked `simple.gds` differed after verification | 2 | User confirmed it is their intentional edit; preserve it and exclude all `TestReticle` files from this feature commit |
| Tests coupled exact baselines to mutable `simple.gds` | 1 | Move exact hierarchy/query/CLI assertions to deterministic generated GDS fixtures |
| Planning completeness script reported 0/0 phases | 1 | Script does not parse the existing status table; manually verified all 11 phases complete |

## Decisions
- Backend: KLayout 0.30.x C++ Region plus NumPy arrays.
- Source database: immutable; patch-only output in this phase.
- Core geometry units: integer DBU.
- OPC fragmentation remains outside geometry.
- `TestReticle/gcd_45nm.gds` is user-owned and excluded from commits unless explicitly requested.
- Direct execution: `python run_layout_geometry.py ...`; editable installation is optional and used only for development tooling.
- Final audit simplification: diagnostics allocate stats only when requested; PatchSet uses per-layer native ownership/result Regions instead of Python O(n²) scans and per-add sorting.
- Raster API: `render_layout_region(LayoutDB, DbuBox, LayerSpec, pixel_size_nm=5.0, output_path=None, show=False)` returns a top-up `uint8` array.
- Raster semantics: white means full mask coverage, black means empty, gray means exact partial area coverage; one Layer per image.
- Raster bounds: 64,000,000 output pixels and 1,000,000 temporary pixels per native stripe by default.
