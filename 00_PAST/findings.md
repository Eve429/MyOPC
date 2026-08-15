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

## MB-OPC Frontend Design
- OpenILT's useful ideas are corner-aware segmentation, fixed reference geometry, inner/outer sampling, maximum displacement, and core/halo separation; its list aliasing, repeated deep copies, pixel-probe normal inference, full-frame coupling and missing cross-tile identities are not suitable here.
- A GDS hole round trip can expose a ten-point keyhole hull with a zero-width bridge. KLayout `Region.merged()` converts it back to one valid hull plus one hole, so normalization must occur before contour extraction.
- `gcd_45nm.gds` currently contains 21,590 mathematical edges. The chosen 16/32 nm policy estimates 223,553 control segments.
- A read-only NumPy prototype measured 10.37 MiB for compact edge-ID/parameter/key storage versus 25.72 MiB for persistent expanded endpoints/normals/parents, a 59.7% reduction; index construction took about 4.2 ms.
- The largest Python-only speedups are one-time physical-boundary caching, edge-level metadata, lazy coordinate materialization, sorted stable-key lookup, sparse core membership and regular-grid point location.
- Physical-mask normalization, rectilinear core grids, line sampling and annotated boundary visualization are reusable by ILT evaluation and later OPC methods; displacement/reconstruction remain MB-OPC-specific.
- The common mask regression proved that minimum-coherence keeps corner-touch components separate, overlap cut-lines disappear, and a GDS keyhole becomes exactly one hull plus one hole with eight physical edges.
- The first real implementation run on `gcd_45nm.gds` produced 223,553 segments with 13.79 MiB persistent arrays, 290.01 ms preparation, 36.25 ms coordinate materialization, 960.54 ms full reconstruction, and zero XOR area.
- The 2x1 grid stored 226,813 sparse memberships for 223,553 segments, confirming that halo context scales with nearby cores instead of an S×C matrix.
- The first synthetic overlay was geometrically correct but only 276×266 pixels because DBU rasterization cannot use a sub-DBU pixel. At that size 71 segment labels obscured the mask, so diagnostic rendering needs a display-only nearest-neighbor enlargement plus independent label/sample decimation; this must not affect physical coordinates or reconstruction.
- Final storage audit found `fragment_indices` and `fragment_counts` duplicated information already encoded by `t0/t1` and stable keys. Removing them saves 8 bytes per segment (about 1.71 MiB on `gcd_45nm`) without adding iteration-time computation or reducing artifact reconstructability.
- The deterministic 110,000-segment benchmark measured 43.4% persistent-array savings versus a conservative fully expanded representation. The remaining sorted-key order/token arrays are intentional: removing them to chase the provisional 50% target would force every update round to rebuild lookup state, conflicting with iteration speed priority; the strict regression floor is therefore 40%.
- The corrected direct real-file flow prepares physical geometry while `LayoutDB` is open, then safely completes updates and artifacts after close. On `gcd_45nm` it used 12,675,300 persistent segment bytes, prepared in 282.89 ms and completed all artifacts in 2.951 s with zero reconstruction and stitch XOR.
- Visual inspection of the five-case atlas confirmed readable normals, owner colors, holes and four-core boundaries; the 1200-pixel `gcd_45nm` overlay clearly separates the two owner domains while decimated labels/samples preserve the full mask topology.

## Function Call Architecture Document
- The runtime has three distinct call phases: database-scoped preparation, database-independent iteration/reconstruction, and optional diagnostics/artifact output.
- `run_mbopc_frontend.run` is the orchestration root; `_prepare_input_problem` is only a CLI unit-conversion/grid adapter, while `prepare_problem` is the reusable library facade.
- `prepare_problem` fans out to `normalize_physical_mask`, `fragment_edges`, `OwnershipPolicy.assign`, and `build_sample_template`, then returns the immutable reference container `MBOPCProblem`.
- The iteration hot path is deliberately short: `merge_owner_updates` -> `SegmentBatch.lookup_keys`, followed by `SegmentBatch.materialize` -> `sample_lines`; no layout query, mask merge, contour extraction, edge fragmentation, hash construction or owner assignment repeats.
- Reconstruction is method-specific: `reconstruct_region` -> `reconstruct_contours` -> `SegmentBatch.materialize`/`validate_contours` -> `contours_to_region`.
- Output is a fan-out after reconstruction: `PatchSet.add/region`, `save_problem_npz`, `write_debug_gds`, `render_boundary_overlay`, and optional `run_geometry_suite`; none of these are required by a future solver loop.
- The shared-method boundary is `layout` + `geometry` + `opc.input`; edge-oriented construction lives in `opc.input.edge`, while future concrete iteration methods depend downward from `opc.iteration.<method>`.
- The delivered document contains 453 lines, nine Mermaid blocks and 19 verified relative source links; no link target is missing and all Markdown fences are balanced.
- Mermaid CLI is not installed in the current environment, so validation is structural rather than a local SVG render; the diagrams use only standard `flowchart` syntax, quoted labels and subgraphs/arrows supported by common Markdown Mermaid renderers.
- `.vscode/launch.json` has a concurrent user change adding an MB-OPC debug configuration; it is unrelated to this documentation task and must remain unstaged.

## OPC Directory Responsibility Split
- The repository has no concrete iterative solver yet: the former `opc.mbopc` files construct edge-oriented input, assign ownership, carry updates, reconstruct contours and emit diagnostics.
- The move-only target is therefore `opc.input` for method-neutral input data and `opc.input.edge` for edge-segment input mechanics; future solvers belong under `opc.iteration.<method>`.
- `lithography/`, `evaluation/`, `opc/iteration/` and `opc/iteration/mbopc/` may be created as empty physical directories, but Git cannot retain them until a real implementation file is added; adding placeholder files would violate the user's no-new-file constraint.
- The user-owned `.vscode/launch.json` change and untracked `gcd_45nm.png` must remain untouched and unstaged; `layout/` and `geometry/` must have no content diff.
- Post-move verification passed: 37 OPC tests at 93% combined statement/branch coverage, 81 full-repository tests, scoped Ruff and compileall.
- AST comparison of the 12 non-initializer production modules found zero logic mismatch after removing imports and module docstrings; the two initializers only changed exports and names.
- Both obsolete import specs (`opc.common`, `opc.mbopc`) now resolve to `None`; all 19 source links in the call-architecture document resolve after the move.

## Physical Tile-Size CLI
- Existing `--grid COLUMNS ROWS` divides the selected processing box, not unconditionally the full reticle; without `--box` that processing box is the selected top-cell bbox.
- A grid core is the tile's unique ownership/output region. Its `halo_nm` is not another output tile: the halo is read-only context used by nearby segment membership.
- The new mode should be `--tile-size-nm SIZE`, mutually exclusive with an explicitly supplied `--grid`; it creates square fixed-size cuts anchored at the selected box's left/bottom and clips the final column/row to the right/top boundary.
- Unit conversion should remain in the root CLI adapter because `RectilinearCoreGrid` intentionally stores integer DBU cuts and is shared by callers that may not use nanometers.
- The first 100 nm synthetic run generated exact cuts `x=[-20,80,180,250]`, `y=[-20,80,180,240]`, but nine per-core KLayout intersections did not reconstruct the diagonal-containing Region exactly.
- The discrepancy exists at zero displacement (35 DBU² XOR) and after demo updates (29 DBU²), so it is not caused by the new physical-size conversion or update policy.
- Direct manual per-box intersections and `PatchSet` produce identical loss; merging the fragments does not repair it. The failure boundary is the existing integer Region clipping of diagonal polygons across multiple ownership boxes.
- Completing a trustworthy whole-reticle tile mode therefore requires a focused `geometry/patch.py` correction plus diagonal/multi-row/multi-column regressions. Skipping the stitch check in the runner would hide data loss and is rejected.
- The user approved the corrected responsibility split: core boxes partition computation and ownership, while canonical vector output uses one global reconstruction. This removes the unnecessary diagonal clipping operation rather than weakening its XOR assertion, and requires no `layout/` or `geometry/` change.
- `run_mbopc_frontend.run()` and its physical-tile regression are the only existing functions explicitly approved for this semantic correction; subsequent solver work should use new modules unless separately authorized.

## ICCAD13 Lithography And Evaluation
- The four actually used OpenILT assets total about 474 KiB: focus/defocus kernels are 24×35×35 and each scale vector has 24 float weights. Asset hashes are locked by tests and the MIT license is copied beside the implementation.
- The migrated aerial-image calculation matches OpenILT pixel-for-pixel on the real 200×150 patch shape for nominal/max/min. OpenILT's full 256×256 path incorrectly returns the input mask from `unpad`; MyOPC deliberately fixes that and has a regression proving full-canvas output remains continuous.
- Direct `myopc/python.exe` on Windows needs `<env>/bin` both registered through `os.add_dll_directory()` and prepended to the process `PATH`; the latter is required because NVRTC performs its own builtins lookup. A direct CUDA subprocess test now covers this without `conda run`.
- Two 256×256 masks used about 64.45 MiB peak CUDA allocations in the first smoke test. The model remains a differentiable `torch.nn.Module`; MB-OPC will wrap calls in `torch.no_grad()`, while future ILT can reuse gradients.
- Evaluation keeps images on their existing device, excludes halo pixels with an ownership mask, and returns only scalar L2/PVBand plus compact per-segment EPE directions. Invalid narrow-feature/out-of-bounds/same-pixel probes never move an edge; simultaneous inner/outer violations are explicitly ambiguous with direction zero.

## Streaming Simple MB-OPC
- `opc.iteration.mbopc.optimize` keeps only the compact global displacement/contour state on CPU. It creates current/target/ownership images for one batch, returns only scalar metrics and compact owner directions, then releases the batch tensors.
- Every tile in a round reads `current`; owner updates accumulate in `next_values` and become visible only after all batches complete and global contour validation succeeds. Cross-core order therefore cannot update an edge early.
- Target tiles are cached as uint8 under an explicit LRU byte bound. A regression caught and fixed a cache-hit normalization error that would otherwise pass values up to 255 into the second round.
- Final vector output is reconstructed once from global best displacements. Core boxes control evaluation ownership and update authority but never clip final polygons.
- Candidate topology safety is deliberately conservative in v1: any predictable reconstruction failure rolls back the complete round. This avoids partial-polygon publication without adding an unproven per-polygon recovery layer.
- The 2 DBU hollow-wall/8 DBU probe case invalidates long-edge probes through target semantics. A few corner fragments can remain locally valid because their inward normal crosses the adjacent perpendicular wall; tests distinguish this geometry rather than incorrectly declaring every probe invalid.
- Direct `simple.gds` CUDA validation used 8 cores and 885 segments. Three rounds reduced EPE 338 → 203 → 113, used about 65 MiB peak allocated GPU memory, and produced valid GDS/NPZ/JSON/PNG artifacts.
- The new `run_mbopc.py` is directly executable from any working directory without installing MyOPC. Defaults target `TestReticle/simple.gds`; full-reticle and ROI runs share fixed-nm tile, halo, pixel-grid and batch controls.
- Final-code full `gcd_45nm` validation processed 870 cores, 223,553 segments and 880,801 memberships in three rounds. EPE fell 129,645 → 74,592 → 48,348 and L2 fell 1,038,629.522 → 563,335.522 → 440,251.431; PVBand rose and is reported separately rather than hidden.
- The final run took 84.708 seconds, peaked at 271,544,320 CUDA-allocated bytes on a 4 GiB GTX 1650, and wrote valid GDS/NPZ/JSON/PNG artifacts. This validates bounded streaming behavior but is not a maximum-batch claim for the target 24 GiB GPU.
- Topology regressions proved plain reconstruction accepted both a rectangle whose left edge crossed the right edge and a hull moved inside its hole. The solver now rejects ring-orientation changes and hole escape before the round barrier; legal `gcd_45nm` results remained bitwise-identical in metrics.

## OPC Architecture Subtraction Audit
- The complete repository baseline is healthy: 119 tests passed in 19.26 seconds before phase 25 changes.
- `problem.physical_mask.contours/edges` and `problem.segments.contours/edges` are immutable shared references rather than copied geometry arrays; retaining those aliases keeps the two domain objects self-contained without material memory cost.
- Stable `edge_keys`, `segment keys`, `_key_order` and `_sorted_tokens` consume 7,499,136 bytes on `gcd_45nm`, 59.16% of the reported 12,675,300-byte `SegmentBatch` persistent storage.
- The current streaming solver bypasses stable keys and `merge_owner_updates`; those paths are sustained only by the frontend demo, tests, benchmark and NPZ export. The user selected complete removal and current-process index identity.
- `edge_segment_offsets` has no production consumer outside diagnostic export, and persistent `edge_lengths` only feeds materialized diagnostic lengths. Both can be derived only when explicitly requested.
- `BoundarySampleTemplate` is stored on every problem although the solver computes EPE probes independently. Its default normal offset comes from `corner_length`, while the solver uses `epe_distance`; separate values make the preview disagree with actual evaluation.
- The user selected regular `RectilinearCoreGrid` ownership only and requested that `run_mbopc_frontend.py` remain as a separate direct verifier.
- NPZ remains only in the explicitly invoked frontend verifier, becomes index-aligned without keys, and is no longer emitted by the full solver entry point.
- `UniformGridIndex` and the replaceable-backend claims around `GeometryEngine` remain separate protected-directory audit findings; phases 25-27 must not change `layout/` or `geometry/`.
- `SegmentGeometry.lengths` and `segment_indices` were allocated on every full materialization although reconstruction consumes only starts/ends/normals; diagnostic maximum lengths can be derived from `np.linalg.norm(ends-starts)` only when requested.
- The three edge-input files `artifacts.py`, `visualize.py` and `verification.py` are all explicit diagnostic lifecycle code. Consolidating them under `opc/diagnostics.py` removes output/test responsibilities from input construction without inventing a new directory.
- Full `run_mbopc.py` tests currently assert a default NPZ; this must be deliberately updated so only its GDS/JSON remain mandatory, while the retained frontend verifier checks NPZ format version 2 without key arrays.
- After core source/test migration, the only remaining Python references to removed key/length APIs are in `benchmarks/benchmark_mbopc_frontend.py`; documentation references are intentionally deferred to the synchronized report phase.
- `opc.diagnostics` now owns every explicit artifact and geometry-atlas caller. `opc.input.edge` no longer imports or exports diagnostics, preserving one-way dependency and avoiding an import cycle.
- The post-migration Python call-site search finds no remaining use of stable keys, external update batches, template sampling, selective materialization, persistent edge offsets/lengths, or the old reconstruction signature.
- The source diff currently deletes four obsolete edge-input modules, adds one diagnostics module and leaves all protected `layout/`/`geometry/` files untouched.
- Post-refactor full regression passes 118 tests in 15.31 seconds; the count changed because tests dedicated only to deleted APIs were removed while new alias/probe regressions were added.
- The strict 5,000-shape/110,000-segment benchmark passes: 120.11 ms prepare, 11.84 ms materialize, 441.61 ms zero reconstruction, 2.441 MiB persistent segment arrays and 69.38% saving versus expanded storage.
- The historical same-scale benchmark was 168.41 ms prepare, 17.04 ms materialize, 477.95 ms reconstruction and 43.43% saving, so the subtraction improves every measured hot-path time while substantially lowering persistent storage.
- Real `gcd_45nm` frontend preparation remains exact with 223,553 segments: persistent arrays fell from 12,675,300 to 4,830,716 bytes (61.89% reduction), prepare took 161.12 ms, total diagnostic run 2.357 s and zero-displacement XOR remained zero.
- Visual inspection of the regenerated whole-layout overlay confirms complete red/cyan owner partitioning at the vertical core boundary, continuous cross-core geometry, readable labels and paired cyan-inner/red-outer probe markers.
- Focused coverage verification passes 74 tests with 92% combined statement/branch coverage for `opc`, `lithography` and `evaluation`.
- Three-round full `gcd_45nm` CUDA results are numerically identical to the pre-refactor baseline: EPE 129,645→74,592→48,348, L2 1,038,629.522→563,335.522→440,251.431 and peak allocation 271,544,320 bytes.
- Full-flow total time is 85.473 seconds versus the historical 84.708 seconds (0.90% difference, normal run variance); frontend preparation is 0.249 seconds and persistent segment arrays are 4,830,716 bytes.
- The full runner artifact schema now contains only summary JSON, result GDS and optional preview PNG; no NPZ path or file is produced.
- 当前开发手册、测试手册、函数调用文档和两组专项报告仍保留稳定键、外部更新批次、可替换 owner 策略及全流程 NPZ 等旧描述，必须按实际代码整体同步，不能通过局部术语替换掩盖数据流变化。
- `doc/design_review.md` 和 `CLAUDE.md` 均为已跟踪的历史/用户文档；本阶段不改写设计审查历史，也不让旧兼容说明扩大本次 OPC 实现范围。
- 开发手册与测试手册已按当前数组下标身份、单一规则网格、固定参考同步迭代和诊断输出边界重写；其中明确记录 PVBand 上升，避免只展示改善指标。
- 函数调用文档已按实际入口、输入构造、浅引用、同步轮次、探针、全局重建、诊断边界和扩展位置重建；旧专项报告仍是剩余的主要陈旧引用来源。
- 五份既有 MB-OPC 专项文档已全部按当前实现重写，并新增架构精简开发/测试报告；稳定 key、外部更新和全流程 NPZ 只作为“已删除/不支持”历史对照出现。
- 当前 `run_geometry_suite` 重新生成 5 个图形案例，全部零位移 XOR 为 0；segment 数分别为 56、70、39、78、48，图与 JSON 均来自精简后的同一诊断实现。
- 人工查看孔洞与整张 `gcd_45nm` 图：孔洞内外 probe 方向、四 core 边界和左右 owner 分区正确，跨 core 轮廓连续；无显示层面的回归。
- 最终静态审计覆盖 76 个第一方 Python 文件：中文模块/函数 docstring 零缺失；13 份 `doc/*.md` 无断链或不平衡代码围栏；生产模块没有零引用函数。
- 删除符号搜索只命中两个刻意断言旧字段不存在的回归测试；Ruff、compileall 和全仓库 118 项测试最终复跑通过（19.29 s）。
- 最终覆盖率复核为 74 项通过（15.94 s）、OPC/光刻/评价综合语句/分支覆盖率 92%，`opc.iteration.mbopc.solver` 本身为 92%。
- 用户现已授权修改 `layout/`、`geometry/` 以继续精简；此前保留这些目录不是遗漏，而是遵守逐次确认边界。当前调用搜索显示 `UniformGridIndex` 只有测试/基准调用，`RegionBatch.backend` 永远固定为 `klayout`，`GeometryEngine` 是固定后端上的无状态包装，三者需要进一步逐项审计。
- 历史设计评审同样把 `UniformGridIndex` 判为零生产调用方，把 `GeometryEngine` 的多后端理由判为不成立；当前完整流程已经稳定，满足其“真实工作流稳定后再退役”的前提。
- `ShapeQuery.materialize()` 当前只用 ROI 限定候选，完整 runner 额外调用 `GeometryEngine.clip`，但 MB-OPC 直接消费者没有同一步骤；把精确 Region 相交统一到物化边界可以同时删除门面唯一生产调用并修复不同入口的 ROI 语义差异。
- 阶段 28 已删除 `geometry/region.py`、`geometry/spatial.py`、固定 backend 字段/异常、edge bbox 属性及仅服务这些 API 的两份测试；direct runner 与层级基准改为消费精确物化结果。
- 新回归用 100×100 DBU 图形查询内部 50×60 DBU ROI，要求物化 bbox 精确等于查询框且面积为 3,000 DBU²，直接覆盖此前不同入口裁剪语义不一致的问题。
- KLayout 普通 Region `&` 会丢弃属性；本地 API 实验证实 `and_(clip, NoPropertyConstraint)` 保留左侧 Polygon 属性。属性模式现使用该原生批量约束，普通模式继续使用更直接的 `&`。
- 属性回归现在让 ROI 同时截断普通/带属性两个图形，证明精确裁剪后数量一致且 tagged 属性仍准确；修复后 39 项 layout/geometry/OPC 聚焦测试与 Ruff 通过。
- 精简后的严格 Layout/Geometry 基准通过：百万实例 ROI 精确物化中位数 0.10435 ms、P95 0.16723 ms、RSS 增量 0.484 MiB；2048² 栅格 499.59 ms、6.379 MiB、覆盖完全一致。
- 活跃源码已无 `GeometryEngine`、`UniformGridIndex`、backend 标记或 edge bbox 调用；剩余陈旧引用仅在两份 Layout/Geometry 报告中，历史 `design_review.md` 作为当时审计证据保留不改。
- Layout/Geometry 专项 38 项通过（1.49 s），综合 statement/branch coverage 91%；删除 5 项仅服务旧 API 的测试、新增 1 项生产 ROI 回归，不降低现有路径覆盖。
- 全仓库最终静态/语法/回归初次复跑通过：Ruff、compileall、114 tests passed in 15.74 s。测试数由 118 净减 4，恰好对应删除 5 个旧门面/索引用例并新增 1 个 ROI 回归。
- 阶段 28 真实 `gcd_45nm` 前端复跑保持 1,776 polygons、21,590 edges、223,553 segments、4,830,716 persistent bytes 和全部零面积差异；prepare 152.82 ms，总计 2.308 s。
- 三轮 CUDA 完整复跑逐项指标再次完全一致：EPE 129645→74592→48348，L2 1038629.522→563335.522→440251.431，PVBand 115626.751→134540.869→147186.806，GPU 峰值 271,544,320 bytes；总计 85.892 s，结果合法且无 NPZ。
- 阶段 28 最终 MB 前端严格基准仍全部通过：prepare 125.64 ms、materialize 12.45 ms、zero reconstruct 427.83 ms、persistent 2.441 MiB、内存节省 69.38%、XOR/unowned/strict failures 均为 0。
- 最终引用审计另发现 `DbuBox.overlaps` 只是 `intersection(...) is not None` 的无调用方包装，已删除并让原测试直接断言 intersection 语义。
- `LayoutDB.hierarchy_summary` 虽无内部生产调用，但它是早期明确交付的只读层级检查公共能力，且没有第二套实现或热路径常驻成本；保留它不是为假设后端建立的抽象。
- 最终代码复跑通过：Ruff、compileall、全仓库 114 tests（16.03 s）；Layout/Geometry 38 tests（1.50 s）、综合 statement/branch coverage 91%。
- 最终静态审计覆盖 72 个第一方 Python 文件和 13 份 Markdown：中文 docstring 零缺失、链接/围栏零错误、删除符号零生产引用、diff whitespace 零错误；用户 `CLAUDE.md`、历史 `design_review.md` 与 `TestReticle` 无差异。
- 零内部生产引用只剩 `hierarchy_summary` 与 `DbuBox.intersection` 两个明确公共基础能力；前者提供只读层级快照，后者提供 planner DBU 框精确关系，均实现紧凑、无并列抽象且已有直接测试，因此有意保留。
- 阶段 29 用户只授权两项优化：ownership 构建避免完整 `SegmentGeometry` 临时物化；零位移初态和局部未变化 core 跳过轮廓/Region 重建。稀疏活跃 core 方案暂缓，不得增加 `core_at`、改变 core 数量或修改分块 JSON 语义。
- 11 万 segment 的 ownership 等价对照中，owner/core offsets/memberships 逐项一致；中位耗时 `40.20 -> 37.15 ms`，tracemalloc 峰值 `24.56 -> 17.85 MiB`。端点中间表必须在 CSR 展开前显式释放，否则 Python 局部引用会抵消省掉 normals 的内存收益。
- target LRU 使用 uint8 常驻，而历史 current mask 保留浮点像素覆盖率；零位移 core 因此不能直接返回 target。当前快路直接栅格化参考 Region，只跳过轮廓子集与 Region 差分，保持原评价数值语义。
- 零位移局部等价微基准（500 polygons/11,000 segments）像素逐项相同，中位耗时 `44.35 -> 11.06 ms`；`gcd_45nm` 三轮 CUDA 的 EPE/L2/PVBand 等全部指标与阶段 28 逐项一致，总耗时 `85.892 -> 79.117 s`，GPU 峰值仍为 271,544,320 bytes。
- 阶段 29 最终审计覆盖 72 个第一方 Python 文件和 20 份 Markdown：中文模块/函数 docstring、链接、代码围栏、diff whitespace 均无问题；没有新增生产抽象、死函数、重复字段或异常兼容分支，`layout/`、`geometry/` 无差异。
- 离线光刻输入必须复用 `opc.input.raster.rasterize_region_canvas`，其数组原点在左下；现有 `geometry.render_region_batch` 是顶部向下的显示图，不能作为模型输入归档。
- 现有 `opc.diagnostics.save_problem_npz` 是不可恢复的 version 2 诊断快照，缺少 DBU、Layer、query box、分段配置和完整 `EdgeBatch.is_hole`，不应兼容性扩张为离线问题格式。
- `MBOPCProblem` 的可恢复最小状态是 `PhysicalMask + FragmentationConfig + SegmentBatch + OwnershipBatch`；原始 Region 可由 contour 批量重建，因此加载后不需要源 GDS。
- 物化前只能对原始层级图形复杂度做保守估算；KLayout 布尔合并产生的新交点无法在不构造 Region 的情况下完全精确预测，所以还需要物化后的实际数组上限复核。
- 测试脚本位于 `tests/workbench` 时，直接运行的 `sys.path[0]` 不是仓库根；入口需用文件位置一次性加入项目根路径，才能满足不安装包且从外部工作目录运行。
- `simple.gds` 完整 bbox 在 8 nm/256 canvas 下需要 225×413 像素，像素准备函数按设计在物化前拒绝；真实光刻验证必须给出明确小 ROI 或使用更大像素，不能绕过 canvas 契约。
- `simple.gds` 以 512 nm core/256 nm halo 生成的离线问题包含 10 polygons、14 rings、107 edges、885 segments、28 cores、3305 memberships；关闭源版图后加载并零位移重建 XOR 面积为 0。
- `simple.gds` 显式 ROI `[-2000,-1100,-200,948]` 在 8 nm 下生成 225×256 有效像素，CUDA 光刻前向 0.477 s、峰值 34,026,496 bytes，三工艺角 PNG/NPZ 均成功保存。
- 512/256 nm core/halo 的完整离线迭代加载耗时 0.023 s，三轮 EPE 339→212→112，总计 1.578 s，GPU 峰值 271,531,520 bytes，结果 Region 合法。
- 人工查看 nominal PNG：坐标方向正确，孔洞未被填平；查看 OPC overlay：跨 core 外轮廓连续，斜边端点相接，owner 色与 inner/outer 探针方向可辨。全图标签密集但不影响诊断数据。
- 默认 1024/512 nm core/halo 离线复跑恢复为 8 cores、2658 memberships，三轮 EPE 338→203→113，与既有 `run_mbopc.py` 历史基准逐项一致；归档加载 0.021 s，总流程 1.513 s。
- 全仓库最终功能基线为 Ruff/compileall 通过、127 tests passed in 26.84 s；新增 10 项工作台测试后原有 117 项无回归。
- 工作台自身 statement/branch 综合覆盖率为 74%；主要未命中是三个可直接运行 CLI 的错误退出、低概率损坏归档分支和外部工作目录 bootstrap。核心成功路径、物化前保护、三类损坏输入及两个真实模型入口均已命中。
- 最终异常入口审计发现 `metadata.counts` 缺键时原实现会在统一 try 块外泄漏 KeyError；已把计数结构/类型转换纳入同一校验并增加损坏 metadata 回归，没有保留兼容分支。
- 从 `C:\Windows\Temp` 使用绝对脚本路径实际完成 raster 准备、CUDA 光刻和一轮 MB-OPC，三个入口退出码均为 0；深层脚本无需安装项目的 bootstrap 已由真实工作流验证。

## MB-OPC Contract Subtraction
- `PhysicalMask.contours/edges` and `SegmentBatch.contours/edges` currently share object references, so the duplicate fields do not copy arrays, but they create two apparent owners and make the preparation boundary unclear.
- `EdgeBatch` is fully derived from `ContourBatch`: starts copy vertices, ends index the next ring vertex, and ring/polygon/hole vectors repeat contour topology. On `gcd_45nm`, its five arrays occupy 1,057,910 bytes.
- The current `ContourBatch.layer` repeats the `LayerSpec` already carried by `PhysicalMask` or the mapping key. `ring_polygon_ids` and `ring_is_hole` can be replaced by `polygon_ring_offsets`, because extraction already emits each polygon hull followed by its holes.
- A compact endpoint benchmark showed cached `edge_next_ids:int32` materialization at 15.826 ms median versus 15.718 ms for the current EdgeBatch; deriving closure every call rose to 19.515 ms. Retain the small cache to avoid a simplification-induced hot-path regression.
- The solver repeatedly maps member segments to polygons, so retain `edge_polygon_ids:int32` as the second measured/useful edge cache. Ring IDs and hole flags are only needed during one-time preparation and can remain temporary.
- `OwnershipBatch.cores` expands a 488-byte grid into roughly 200,380 bytes of Python objects for 870 cores. Store `RectilinearCoreGrid` in the problem and generate CoreSpec objects once locally only when iteration or diagnostics requests them.
- Current unique NumPy storage on `gcd_45nm` is 10,688,650 bytes. Removing EdgeBatch, replacing it with two int32 caches, and removing persistent expanded cores is expected to save about 1.08 MB without endpoint-materialization regression.
- The offline v1 archive currently persists all repeated contour/edge fields plus derivable core IDs/core/context boxes. Version 2 will store nested contour CSR, two edge caches, segment arrays, grid cuts and membership CSR only.
- Final `gcd_45nm` run with 1024 nm tiles/512 nm halo produced 1,776 polygons, 21,590 edges, 223,553 segments, 870 cores and 880,801 memberships; all XOR/coverage/overlap checks were zero.
- New full-problem NumPy storage is 9,802,180 bytes versus the pre-change 10,688,650-byte same-scope baseline, an 886,470-byte/8.29% reduction. SegmentBatch-owned arrays are 5,003,436 bytes because the two measured caches are now counted explicitly.
- A same-process 30-run comparison is the valid materialize gate: old EdgeBatch-style arrays 28.229 ms median, nested-next 28.205 ms median. The earlier 15.7 ms microbenchmark measured a narrower endpoint expression and should not be compared directly to full `SegmentGeometry` materialization.
- Zero-displacement reconstruction on the full problem was 234.115 ms median over five runs; prepare through the direct frontend was 233.339 ms versus the current-task pre-change read-only baseline of roughly 266 ms.
- Final delivery audit covers 76 first-party Python files and 20 Markdown reports with zero missing Chinese docstrings, broken relative links or unbalanced fences. The only names without another production reference are three tested public APIs and PyTorch's `forward` callback, so none are accidental dead wrappers.

## 本轮代码优化审计

- 百万逻辑实例 ROI 查询中位数为 0.10375 ms、P95 为 0.13689 ms、RSS 增量约 0.48 MB；2048×2048 栅格化为 502.317 ms、RSS 增量约 7.14 MB。这两项作为 Layout/Geometry 不退化基线。
- 20,000 个矩形轮廓提取中，逐 ring NumPy 小数组再拼接约 563.74 ms、峰值 6,679,859 bytes；单次 `array('q')` 连续缓冲约 456.30 ms、峰值 1,845,632 bytes，峰值下降约 72%。
- CUDA 256×256、batch=8 下，当前三次 mask FFT 路径约 25.615 ms；共享一次频谱并按 dose 平方缩放约 16.750 ms，提升 1.529 倍，最大逐像素差约 2.95e-6，GPU 峰值略降。
- CUDA batch=64 的目标 tile 路径中，逐 tile 转 float32 后堆叠约 20.565 ms；先堆叠 uint8、一次传 GPU 并原位归一化约 9.582 ms，提升 2.146 倍，CPU 临时批内存由 32 MiB 降至 4 MiB。
- owner 索引利用严格递增且唯一的 membership CSR 过滤，可避免每个 core 全局扫描 segment owner；223,553 segment/870 core 合成基准由 45.908 ms 降至 13.178 ms。
- `layout/writer.py` 反向依赖 `geometry.patch.PatchSet`，`layout/layer.py` 只有一个单调用函数；用户已确认直接移除并把 `PatchWriter` 归入 `geometry`。
- `benchmarks/benchmark_mbopc_frontend.py` 仍读取已删除的 `segments.edges` 和 `problem.ownership`，小规模直跑会抛 `AttributeError`；需要回归覆盖。
- 本轮不实现稀疏大 reticle 数据结构。密集 core、全 ROI 物化和 macro 间 segment 身份问题将在独立方案中说明。
- 正式 CUDA batch=8、256²、20 次中位数对照中，独立 FFT 路径为 25.0241 ms，共享频谱路径为 16.4560 ms，提升 1.5207 倍；两者峰值均为 277,296,128 bytes，最大逐像素误差为 5.82e-7。
- 修复后的 5,000 图形前端严格基准为 prepare 122.358 ms、materialize 13.097 ms、零位移重建 398.755 ms，110,000 segments 的 XOR 与无 owner 数均为零。
- `simple.gds` 三轮 CUDA 正式验证保持 EPE 338→203→113，L2 2822.4661→1766.5405→1309.4220，PVBand 388.9287→415.5956→436.1449，owner 更新计数和停止原因未改变。
- `gcd_45nm` 三轮 CUDA 保持 1,776 polygons、223,553 segments、870 cores、880,801 memberships；EPE 129,645→74,592→48,348，所有更新/歧义/停止统计一致，总耗时 79.834 s，GPU 峰值 267,334,656 bytes。
- 漂移的 MB-OPC 前端基准已修复并由仓库外直接 CLI 回归覆盖；100 图形严格模式通过，避免 10 图形/64 core 的非代表性 halo 比例。
- 最终生产零内部引用仅为已明确保留且有直接测试的 `hierarchy_summary` 和 `DbuBox.intersection`；本轮没有新增生产函数、结构体或模块，源码净删除两个文件。

## 阶段 44 前置审计

- `geometry.raster` 服务显示：可变图幅、顶部向上、`uint8` 与分条内存上界；`opc.input.raster` 服务模型：固定方形画布、左下原点、padding 与 `float32`。两层接口不能合并，但底层 Region 覆盖率计算可以共用。
- 通用 `types.py` 分散在四个包中；包边界本身合理，问题是名称不表达职责且若干类型只被单一操作使用。采用选择性归位，不建立新的万能类型目录。
- `opc.input.types` 与 `opc.input.edge.types` 存在完全重复的 `_vector`；数组形状/类型校验应集中到输入层私有 `_arrays.py`，已有多个现实调用方。
- 当前前端在预检前构造完整 Region 和 `MBOPCProblem`，且每个 core 全局扫描 owner、全局物化参考/移动端点与 probes、默认输出全量 NPZ/GDS/PNG；百亿边段会在 `int32` 索引与总体近 TiB 状态下不可运行。
- 阶段 1 只实现物化前容量预检与安全拒绝，不实现 CPU macro 或磁盘 shard；默认内存预算使用启动时可用内存的 70%。
- `gcd_45nm` 预检 segment 与实际值同为 223,553；membership 采用数学边 bbox 上界得到 1,167,992，实际为 880,801，符合“拒绝保护可高估、不能低估”的目标。
- `gcd_45nm` 跳过产物的完整前端 peak working set 为 148,467,712 bytes，显著低于 73,809,488-byte prepare 估算与进程基础内存之和所给出的安全量级；估算不作为精确计费值。
- 2048² raster 公共底层复用后为 416.94 ms、RSS 增量 7.62 MiB，覆盖率精确；相比最近 483–502 ms 项目基线无退化。
- 当前结构已无第一方旧深层 types 导入；`_arrays.py` 仅含三种有多个现实调用方的数组校验，不承担无边界的通用工具职责。
- 最终重复实现审计发现固定 tile cuts 在三个当前入口完全相同；已归入 `grid.axis_cuts_by_size` 并删除三个本地版本。数量均分 `_axis_cuts` 只服务 frontend 的另一种 CLI 语义，继续保留。

## 阶段 48 大 Reticle 方案结论

- `ShapeQuery.materialize()` 的精确 ROI 裁剪适合显示、栅格化和局部画布，但会产生 ROI 框轮廓；当前 `prepare_problem()` 不区分裁剪边和物理边，因此它不能直接用于阶段 2 的跨 macro 提边。
- macro 必须同时具有唯一写入的 `ownership_box` 和只读上下文的 `context_box`。未来取得 Layout 修改授权后，应增加未裁剪的完整候选 occurrence 批量入口，同时保持现有 `materialize()` 语义不变。
- 阶段 2 采用“每一轮逐 macro 展开、计算、释放，轮末统一发布”，不采用“一个 macro 完成全部轮次再处理下一个”。数学边的分段相位锚定完整真实边，避免斜边被独立裁剪后出现 33/34 DBU 分歧。
- 百亿 segment 不能依赖全局 `int32` 或全部 RAM 常驻；采用 shard-local `int32`、全局 `int64` offset，并根据预检在 RAM 紧凑状态与 memmap 双代状态之间选择内部路径。

## 阶段 49 前置审计

- OpenILT `simple/exact` 的 Hopkins forward 基本一致，差别集中在 backward：simple 使用单个 combo/CT 近似核，exact 使用完整 CT 核；MyOPC 当前普通 PyTorch 算子已经能从实际 forward 自动构造精确 VJP。
- TorchLitho Hopkins 值得借鉴独立工艺条件、batch 和梯度验证；运行时 TCC/SVD、Abbe 光源循环和只以全一上游梯度展示的自定义 VJP 暂不迁移。
- 当前 `LithographyResult` 固定绑定 nominal/maximum/minimum，和用户要求冲突；改为单条件 `forward` 与任意条件 `forward_many`，后者仍共享 FFT/相同 kernel bank 传播以避免性能倒退。
- OpenILT evaluation 的实际能力是二值 L2、二值 PVBand、EPE 和 Shot。现有矢量 EPE 更适合跨 core MB-OPC；Basic 会原地修改 mask，ShotCounter 有逐像素 Python 循环、随机多进程和额外 adabox 依赖，均只迁移语义、不照抄实现。
- 仓库当前共有五个 `run_*.py`；本阶段连同新增 `run_simpleilt.py` 全部集中到 `main/`，并把离线入口共享实现移出 tests，避免正式入口反向依赖测试包。

## 阶段 49–54 实施结论

- 固定 `LithographyResult` 已删除；独立条件字典只存在于一个 batch，不形成新的常驻数据结构。相同 kernel bank 的 unit aerial 复用，模型仍只执行一次 mask FFT。
- 非均匀正负上游权重的中心有限差分与原生 autograd 一致，证明当前反向不依赖“损失梯度全一”的特殊假设。
- OpenILT 二值 L2/PVBand 已按 ownership 流式累计；EPE 相同但诊断 L2 改善的回归仍选择首轮，确认普通 MB-OPC 不受诊断指标控制。
- Shot 使用确定性 horizontal-run 矩形合并，避免 OpenILT 的逐像素连通域收集、随机搜索、多进程和额外依赖；固定 512² 仅是评价分辨率，不是物理最少 shot 证明。
- SimpleILT 只复用 raster、lithography 与 evaluation，不构造 edge problem。唯一实现把配置/记录/结果/算法同置 `simple.py`，不存在为了未来方法预建的空层。
- 全部七个可执行/离线脚本集中到 `main/`；旧根路径和 `tests/workbench` 脚本均删除。JSON/NPZ/PNG 原子输出收敛到 `main/offline_inputs.py`，没有新增仅含小工具的额外文件。
- 全仓库 152 项通过；专项 39 项、四目标模块综合 statement/branch coverage 92%。本阶段不修改 `layout/`、`geometry/`，也未重新建立 `gcd_45nm` 的新二值指标整图基线。
- 最终 AST 审计覆盖 80 个第一方 Python 文件，中文模块/函数 docstring 缺失为 0；低引用项均为有真实调用方的私有函数、被测试的公共 API 或 PyTorch `forward` 回调。
- 重复函数名复核后只保留不同对象的自然同名方法和两个算法命名空间各自的 `optimize`；runner 的重复 JSON/PNG/DBU helper 已收敛，frontend 中语义不同但同名的选择函数改为 `_select_layout_scope`。
- 文件拆分复核确认 ILT 只有实现文件和包导出，evaluation 仍为单实现文件，`main/` 每个脚本对应一个可独立运行工作流；继续拆分会增加跳转而没有第二个现实调用方。

## 阶段 55 前置审计

- `run_lithography.py` 与 `run_simpleilt.py` 当前都只接受 raster NPZ；版图到 mask 的安全预检、精确 ROI 物化和固定画布栅格逻辑已经完整存在于 `main/offline_inputs.py`，不应在两个入口复制。
- 最小设计是把现有 raster 准备拆成“内存物化”和“可选归档”两层：直接 GDS/OASIS 只返回 `float32` mask 与 metadata，NPZ 工作流继续调用相同内存层后原子保存。
- 输入自动分派只需要一个已有模块内的共享函数，并已有光刻、SimpleILT 两个现实调用方；不新增注册器、输入类或独立文件。非 `.npz` 路径交给 KLayout 读取，版图参数只影响该分支。
- 整版栅格仍受文件大小、层级 occurrence、源顶点、输出画布和预计内存上限约束；直接输入不会绕过保护，也不会在源版图旁生成隐式 NPZ。

## 阶段 55–57 实施结论

- `materialize_raster_input` 成为版图到内存 mask 的唯一实现，`prepare_raster_input` 只增加原子归档；直接输入与 NPZ round-trip 的 mask 和 metadata 完全一致。
- `resolve_raster_input` 是唯一新增分派函数，已有光刻和 SimpleILT 两个调用方；未新增输入数据类、注册器、包装模块或第二套模型路径。
- `run_mbopc_iteration.py` 继续只接受边段 NPZ，因为直接 GDS 的完整 MB-OPC 已由 `run_mbopc.py` 提供；避免同一前端存在两个实现。
- `simple.gds` 直接 CPU 光刻与一轮 SimpleILT 均退出 0，光刻 shape 为 256²，SimpleILT binary L2 为 1900；两条命令均未准备 raster NPZ。
- 函数体审计只发现 `run_mbopc.py` 中一份与公共 `parse_layer` 完全相同的实现；删除后重复函数体为 0，30 项相关入口回归通过。

## 阶段 59 接口审查约束

- 接口文档以当前源码为唯一依据，区分包级公共导出、模块级可调用接口、内部辅助函数和命令行入口，不把未来大 reticle 方案写成当前能力。
- 每项核心接口需要明确数据类型、数组 shape/dtype、坐标系与单位、对象生命周期、可变性、异常、内存/性能语义和主要调用方；本轮不通过修改生产代码来“配合文档”。
- 当前生产代码共 45 个 Python 模块：`layout` 6、`geometry` 6、`opc` 21、`lithography` 2、`evaluation` 2、`main` 8（含各包 `__init__`）；接口文档将逐一覆盖，并单列 7 个直接运行入口。
- 包级公共入口由 `layout/__init__.py`、`geometry/__init__.py`、`opc/input/__init__.py`、`opc/input/edge/__init__.py`、`lithography/__init__.py`、`evaluation/__init__.py` 和两个迭代子包定义；`_arrays.py`、ownership 及 runner 私有辅助函数只记录为内部契约，不宣传为稳定 API。
- Layout 的核心生命周期是 `LayoutDB.open -> query -> ShapeQuery.materialize -> RegionBatch`；`ShapeQuery` 持有数据库引用，数据库关闭后不可物化，而已经返回的 `RegionBatch` 持有原生 Region。ROI 的候选筛选和精确裁剪均在 KLayout 侧完成，`preserve_properties=True` 只导入并继承属性，不过滤无属性图形。
- Geometry 有三组独立输出契约：`ContourBatch` 是 Polygon→Ring→Vertex 两级 CSR；`PatchSet` 是按 ownership 裁剪、同层无正面积重叠的结果集合；Geometry 与 OPC raster 公共返回数组现已统一为左下原点，只有图片输出边界翻转。
- OPC 公共输入分为三层：`PhysicalMask` 保存合并后的单层物理 Region；`RectilinearCoreGrid` 定义唯一写入 core 与只读 halo；`MBOPCProblem` 再加入固定参考轮廓、segment 参数区间、owner 向量和 core→segment CSR。`prepare_problem` 是这三层的组合边界，一次准备后供多轮迭代复用。
- `SegmentBatch` 不常驻端点；它以 `edge_ids + t0/t1` 引用数学边，`materialize(displacements)` 才输出三个 `float64[S,2]` 数组。owner 由参考边段中点唯一决定；membership 由参考边段 bbox 扩 halo 后形成 CSR，同一 segment 可属于多个 core context 但只能有一个 owner。
- 栅格接口方向已统一：`opc.input.raster.rasterize_region_canvas` 返回左下原点、固定 `canvas×canvas` 的 `float32`；`geometry.render_*` 返回同方向可变图幅 `uint8`。PNG、查看器和诊断标注只在图片 I/O 边界翻转。
- `preflight_layout` 会单独读取版图并扫描层级 ROI，在完整 Region/Segment 分配前返回字节估算和接受判断；提前超预算时计数是下界且 `scan_complete=False`，当前拒绝结果指向尚未实现的 `sharded_required`，不能写成自动切换流式求解。
- 光刻公共接口是 `ICCAD13Lithography.forward(mask, condition)` 和 `forward_many(mask, conditions)`；输入为 `[H,W]` 或 `[B,H,W]`，输出形状与输入一致、值为连续光刻胶概率。模型内部居中 padding 到 canvas、必要时最近邻缩放到 resolution，并在一次 `forward_many` 内共享 mask FFT 和同 kernel bank 的单位剂量强度；原生 PyTorch autograd 提供 backward。
- Evaluation 的 L2/PVBand 返回 Python `int` 像素计数；EPE 返回五个逐探针 Tensor，其中方向 +1 为沿外法向外移、-1 为内移、同时冲突为 0 且 `ambiguous=True`。这些接口接受的像素图必须 shape/device 一致，ownership mask 只控制计分，不裁剪卷积上下文。
- SimpleILT 的输入是目标像素 Tensor 和光刻模型，不依赖 `MBOPCProblem`；返回最优参数、软 mask、二值 mask 和逐轮连续损失。SimpleMBOPC 的输入是固定边段问题和光刻模型；每轮全部 core 从同一 `current` 读取，owner 更新暂存到 `next_values`，全局拓扑检查通过后才发布。
- MB-OPC 当前是“按 GPU batch 流式像素张量、CPU 常驻完整问题”的实现：target tile 可按字节上限 LRU 缓存，GPU batch 结束即释放；但完整 `SegmentBatch`、owner/membership CSR 和全局位移仍在内存中。接口文档必须把这一点和未来 macro shard 方案严格分开。
- `main/offline_inputs.py` 是当前文件级数据契约边界：raster NPZ v1 保存 `float32[canvas,canvas]` 与 JSON metadata；segment NPZ v2 保存轮廓 CSR、边/段数组、owner/membership CSR 和 grid cuts，可恢复完整 `MBOPCProblem`。两个 loader 都先限制归档及解压尺寸、禁止 pickle，并执行结构和跨数组校验。
- `resolve_raster_input` 只按 `.npz` 与非 `.npz` 分派：NPZ 中的 Layer/ROI/pixel 已由 metadata 固定，调用时传入的版图参数不会覆盖；GDS/OASIS 分支执行同一 preflight→materialize→raster 流程且不写隐式中间文件。
- `main/run_mbopc.py` 是直接 GDS→完整前端→迭代→最佳 GDS/JSON/可选 PNG 的生产验证入口；`run_mbopc_frontend.py` 是几何前端/归属/重建/诊断入口，不执行光刻优化；`run_layout_geometry.py` 是单 ROI 查询、轮廓计数、PNG/Patch 验证入口。
- `run_lithography_test` 返回仍在模型设备上的三工艺条件 Tensor 字典，输出目录只控制额外 NPZ/JSON/PNG；`run_simpleilt` 返回 `(SimpleILTResult, summary)` 且总会写结果 NPZ/JSON；`run_mbopc_iteration_test` 只接受 segment NPZ，返回 `SimpleMBOPCResult` 并写 GDS/result NPZ/JSON/可选 PNG。
- 七个可直接运行脚本是：`offline_inputs.py`、`run_layout_geometry.py`、`run_lithography.py`、`run_mbopc_frontend.py`、`run_mbopc.py`、`run_mbopc_iteration.py`、`run_simpleilt.py`。所有入口把仓库根按脚本位置加入 `sys.path`，所以无需安装项目包；正常成功返回 0，可预期输入/领域错误返回 2。
- `run_mbopc_frontend` 的诊断 NPZ（`save_problem_npz`）与可恢复离线 segment NPZ（`prepare_segment_input`）用途不同：前者用于带位移的可视化诊断，后者是严格版本化的 `MBOPCProblem` 输入。接口文档必须明确禁止把两者互换。
- `opc.diagnostics` 的四类输出均是显式副作用接口：按 segment 下标对齐的诊断 NPZ v3、含 `REFERENCE/RECONSTRUCTED` 两个 top cell 的 GDS、顶部原点标注 PNG、五类确定性几何用例 JSON/PNG。它不被输入或求解热路径反向依赖。
- 现有 `function_call_architecture.md` 已较好解释 MB-OPC/ILT 数据流，但不是逐模块 API 参考：缺少 dataclass 全字段、返回 shape/dtype、对象生命周期、完整异常和七个 runner 的输入输出差异。本轮应新增独立 `module_interface_reference.md`，避免把调用关系文档膨胀成难以查询的混合手册。
- 当前运行时 Python 要求 `>=3.12`；核心依赖 KLayout/NumPy/Pillow/PyTorch，`psutil` 实际被生产 preflight 导入，虽然 `pyproject.toml` 把它列在 dev 可选组、`requirements.txt` 已包含。此次只记录现状，不顺带修改依赖配置。
- 测试目录按 Layout、Geometry、OPC、Lithography、Evaluation、离线工作台分层，公开入口均存在生产或测试调用证据；文档可用这些真实调用边界判断“公共”与“内部”，不需要为说明目的新增接口。
- `opc.iteration.ilt.optimize` 与 `opc.iteration.mbopc.optimize` 同名但属于不同算法命名空间，调用方必须从明确子包导入或使用别名；顶层 `opc.iteration` 有意不做聚合导出，避免含义冲突。
- 最终接口参考为 763 行，逐路径覆盖 45 个生产模块和全部包级 `__all__` 符号；49 个相对链接全部存在，代码围栏平衡，Python 示例均可编译。
- 本轮没有修改任何生产 Python、`layout/`、`geometry/` 或用户版图；只新增接口文档并更新手册导航、计划记录和一次已解决的模块计数审计学习。

- 最终光刻输出采用共享原子 I/O；MB-OPC 只按 batch 保留当前 core context，manifest 记录 ownership-only 裁剪，避免整张 reticle tensor 常驻。

- OpenILT 的 LevelSet、CurvMulti、Multilevel 依赖旧 LithoSim/配置体系；本项目仅迁移参数化和调度思想，统一复用当前 ICCAD13 模型。DiffOPC 采用独立解析软边段 rasterizer，不能把不可微 KLayout rasterizer 改造成 autograd 路径。
- LevelSet 的硬前向必须以 `phi < 0` 为唯一权威；`sigmoid(-phi)` 只是连续诊断图，不能再用 `>=0.5` 生成硬结果，否则 `phi==0` 会从关闭错误翻转为开启。
- 精确欧氏 SDF 可用两遍一维下包络在 `O(HW)` 时间和内存内完成；本项目只在初始化执行一次并通过 1×1 至 8×8 随机图暴力对照，不在光刻迭代热路径重复计算。
- 历史第一阶段只验收 LevelSetILT；此后 CurvMulti、Multilevel 和 DiffOPC 已分别在第二、第三、第四阶段完成专项验收。
- OpenILT CurvMulti 不是 LevelSet 的多尺度包装：它对连续像素参数先做 7×7 平均池化，再以带 offset 的 sigmoid 生成软 mask，使用 SGD，并把曲率正则施加到标称曝光图。当前 `multiscale.py` 调用 LevelSetILT，算法身份错误，第二阶段必须替换而不是继续包装。
- OpenILT CurvMulti 源码把 nominal L2 实际写成 `printedMax`，与 process L2 重复，且把 mask 直接乘中央 filter 令窗口外归零；这两处属于历史实现问题，不应照搬。项目版应使用具名 nominal condition，并让优化窗口外保持固定初值语义。
- 当前 ICCAD13 `forward_many` 对小于 canvas 的 mask 只做居中补零，不会按尺度放大；因此 CurvMulti 的 coarse mask 必须先恢复到完整 target 网格再做光刻。粗尺度只能减少控制变量自由度，不能改变 Hopkins 核对应的像素物理尺度。
- CurvMulti 不需要新的阶段记录数据类：固定 `iterations_per_stage` 和 `scales` 可由全局 record 下标推导 stage，runner 只在 JSON 边界附加三个 stage 字段，求解热路径继续复用 `ILTIterationRecord/SimpleILTResult`。
- 第三阶段现有生产树没有 Multilevel 实现或导出；唯一 `main/run_ilt.py` 入口已具备 Simple/LevelSet/CurvMulti 的直接 GDS/NPZ 输入、统一产物和资源统计，可直接扩展方法分派，无需新 runner。
- OpenILT 的 Multilevel 参考实现位于 `pyilt/multilevel.py`，配有 256/512 两套配置；应先区分它与 CurvMulti 的“同一完整物理网格上的粗控制参数”语义，避免再次把张量尺寸变化误当作不改变光刻像素尺度。
- OpenILT Multilevel 的实际身份是两次独立 CurvILT：Low 256²/20 轮后把最优参数近邻放大两倍，Mid 512²/100 轮继续；每一级重建 Adam，实际学习率为配置 StepSize 的 0.2 倍。它不是单个 optimizer 在多个尺度连续运行。
- 参考代码在 Low/Mid 上直接使用同一 35×35 Hopkins 核，且评估时再分别放大 8/4 倍；在当前明确 `canvas/resolution` 的模型中照搬会改变核的物理像素语义。项目版采用“级别参数/监督网格 + 完整物理仿真网格”：低级软 mask 先放大到完整 target 网格做 forward，wafer 再 area 缩到级别网格算损失。
- Multilevel 与 CurvMulti 的现实差异保留为：前者每级可配置独立轮数/Adam 步长并在级别监督网格算损失；后者所有尺度使用相同轮数/SGD，始终在完整 target 网格算损失。两者可共享现有结果/逐轮记录，但不应建立统一求解器基类或新阶段记录类。
- OpenILT Multilevel 内嵌 CurvILT 同样把 `printedMax` 当 L2、窗口外清零并每轮重新分配曲率核；项目版分别改为具名 nominal、窗口外固定初值、复用现有曲率实现且权重为零时不计算。

## 阶段 74 DiffOPC 前置审查

- NVlabs DiffOPC 的公开实现用二值前向加自定义直通反向：每条 edge 从 mask 梯度取平均，再乘移动方向；它适合固定小画布，但 backward 含逐 edge Python 循环，不能直接满足本项目整张 reticle 的流式资源目标。
- 参考实现的训练损失是 nominal L2、maximum/minimum 对 target 的工艺角 L2、maximum/minimum 之间的连续平方差和可选 EPE；真正的二值 PVBand 只作为诊断，不能把返回 Python `int` 的评价函数放进 autograd 损失。
- 参考仓库的 `mrc` 模块是优化后连通域矩形分解与最小面积/宽高过滤，不是边段迭代中的可微 MRC。当前阶段采用已有 `max_displacement_dbu` 投影和 `reconstruct_region` 全局拓扑校验作为已定义的几何约束，不虚构工艺厂规则。
- 当前原型把每个 context 的 halo 像素和全部 member segment 都计入损失，导致同一物理像素/边段因 tile membership 数不同而重复贡献梯度；必须以 `ownership_canvas` 限定像素，以 `owner_indices` 限定 EPE，每个全局对象恰好计分一次。
- 当前原型在所有 batch 结束后一次 backward，会让整轮光刻计算图常驻；等价且有界的做法是在只读同一参数快照期间逐 batch 缩放损失并立即 backward，仅累积全局位移梯度，全部 batch 完成后才执行一次 optimizer step。
- 当前软栅格的 `base + displacement * sign * Gaussian` 不是边界平移的 occupancy 差，且 probe 像素中心少了 `0.5 pixel` 修正。修正版应使用 `sigmoid((d-q)/T)-sigmoid(-q/T)` 的局部占据变化，保证零位移严格等于参考 mask，并按有限边段切向窗口限制影响。
- 当前最佳损失来自 step 前状态，却保存 step 后位移，记录与产物不对应；最佳位移必须与被实际评价的同一快照绑定。最后一次未评价的 step 不得冒充最佳状态。
- 第四阶段不把 SRAF 混入 DiffOPC：SRAF 会新增图形、segment 身份和归属，属于显式 remesh/输入构造方法，必须在其独立阶段重建问题和优化器状态。

## 阶段 75–77 DiffOPC 实施结论

- 软栅格使用法向 signed-distance 占据差，零位移严格复现参考 coverage；segment chunk 配合 checkpoint 后，反向中间量峰值由 `canvas²×chunk` 而非当前 tile 全部 segment 决定。
- L2/连续 PV 只统计 ownership 像素，EPE 只统计 owner segment；1-core/2-core 与 batch=1/2 专项结果一致。逐 batch backward 只累积梯度，Adam step 位于全轮屏障后。
- `reconstruct_contours` 统一拒绝 ring 翻转和 hole 越出 hull，simple MB-OPC 删除重复拓扑实现；共享 `ArrayTileCache` 取代旧私有 `_TargetCache`，生产函数体重复扫描为 0。
- 直接版图入口使用 `materialize_segment_input` 内存层，显式离线归档仍由 `prepare_segment_input` 完成；没有大问题临时 NPZ 写读开销，也没有第二套前端。
- 真实 `simple.gds` 4-core CPU/CUDA 两轮二值指标一致，L2/PVBand/EPE=`773→687/350→247/2→0`；CUDA 峰值分配 133,264,384 bytes。全仓 208 项通过，DiffOPC 专项核心覆盖率 80%。
- 当前仍为 CPU 常驻完整问题、GPU 流式 batch；macro shard、SRAF、多 GPU 和未定义规则 deck MRC 均未虚报为完成能力。

## 阶段 78 FAQ 初步审查

- `geometry.raster` 与 `opc.input.raster` 已共用 `iter_region_coverage_tiles`，裁剪、合并、面积覆盖率和分块逻辑没有重复；差异只在最终数组方向、dtype 和固定 canvas padding。
- OPC/ILT、ownership 与探针坐标统一依赖“第 0 行为低 Y”的模型数组；人眼 PNG 才需要“第 0 行为顶部”。可把展示函数返回值也统一为模型方向，只在 PNG/显示/标注边界执行 `flipud`，从而消除公共 Python 数组存在两种方向的陷阱；这会修改 `geometry/` 的既有返回契约，实施前必须得到用户逐次确认。
- simple MB-OPC 的旧 `iterations=N` 至多提交 N−1 次更新问题已修复：当前最多提交 N 次，初态和每次更新后的状态均评价，完整执行记录数为 N+1。
- FAQ 第 1 条已按实测和正式回归修正：惰性 `ShapeQuery` 在数据库关闭后失败，已经物化的 `RegionBatch` 独立持有 Region 并可继续准备问题。
- 当前 KLayout 实测证明 `RegionBatch` 在 `LayoutDB.close()` 前后保持 25 个 Polygon、面积 136000，并可在关闭后继续执行 `normalize_physical_mask` 和 `prepare_problem` 得到 208 个 segment；FAQ 第 1 条及手册 §1.3/§3.4/§6.3 的“已物化 Region 绑定 DB”说法错误，应改为“只要求惰性查询在关闭前物化”。
- FFT 条目属于必要的模型校准不变量，已有四资产 SHA-256、OpenILT 三工艺角绝对和、共享/独立 FFT 逐像素对照和 autograd 有限差分测试，不应改变实现；可把 FAQ 改成明确指向这些现有回归。
- nm→DBU 精确换算、owner 有效范围、重复 owner 写检测、零位移 XOR、确定性测试夹具和大 reticle 预检均是应保留的安全约束，不是待修 bug。第 7 条文档把当前每轮新建 `written bool[S]` 写成 epoch/bitset，属于实现描述漂移，应只改文档。
- 整轮拓扑回滚能保证正确性但粒度保守；它不是当前 bug。若未来要提升收敛，可在不破坏轮次屏障的前提下采用候选步长回退或按 polygon 隔离非法更新，但必须先定义冲突语义并加入孔洞/对边穿越/跨 core 回归。
- 定向基线 `tests/geometry/test_raster.py`、`tests/opc/test_iteration_raster.py`、`tests/layout/test_database.py`、`tests/opc/test_simple_mbopc.py` 共 40 项全部通过（5.19 s）。
- FAQ 修复最终全仓 210 项通过；2048×2048 raster 为 471.94 ms、7.90 MiB 且 coverage exact。Ruff/compileall、95 个 Python 文件中文 docstring 与重复函数体、36 份文档链接/围栏及 diff whitespace 审计均通过。
- `pytest-cov` 在当前 Windows/KLayout 进程的收集阶段与 NumPy 扩展加载冲突；测试体未执行，因此不作为代码失败，也不报告新增覆盖率数值。无插桩全量与专项分支测试是本轮发布门禁。

## 阶段 82 设计结论

- 当前 KLayout 0.30.10 不直接读取 `.glp`；第一版 GLP 必须严格解析 `EQUIV/LEVEL/CNAME/CELL/RECT/PGON/ENDMSG`，直接构建内存 `Layout`，不得用临时 GDS 绕行。
- `EQUIV 1 1000 MICRON` 表示 `dbu=0.001um`；符号层默认仅允许末尾数字映射为 GDS layer，无法确定或冲突时要求显式 `NAME=LAYER/DATATYPE` 映射。
- `PhysicalMask.region` 继续保存源多边形而非处理框补集。`clear` 时多边形覆盖率即透光率；`opaque` 时仅在显式处理框内计算 `1-覆盖率`，处理框外固定为 0。
- opaque 边段法向需要反向，使公共不变量保持为“法向从透光侧指向不透光侧、正位移扩大透光区域”；这样 EPE、Simple MB-OPC 与 DiffOPC 无需拥有两套方向语义。
- 动态 SRAF 会改变 segment 数量和优化器状态，必须在轮次屏障追加并原子发布；旧 segment 保持前缀稳定，新状态初始化为零，接触导致拓扑合并时只能显式全量 remesh。
- TOML 配置采用“默认 common→默认 entry→自定义 common→自定义 entry→显式 CLI”优先级；配置路径只在进程启动时读取一次，配置内相对路径相对于配置文件目录解析。
- OpenILT ICCAD2013 GLP 的 `EQUIV` 实际包含第五个方向字段 `+X,+Y`，且可能声明未承载图形的 `E1TARGET`；严格解析器接受唯一已定义方向和未使用辅助 LEVEL，只在某符号层真正承载图形时要求可确定映射。
- 当前生产 raster 的直接调用已经全部收口到 `rasterize_mask_canvas`；`rasterize_region_canvas` 只作为其 coverage 底层保留。这样普通几何覆盖与光学极性没有两份裁剪实现。
# 2026-08-12：阶段 88 全项目结构与精简性评审（进行中）

- 当前工作树在评审开始前干净，生产代码共 57 个 Python 文件：`layout` 7、`geometry` 6、`opc` 29、`lithography` 2、`evaluation` 2、`main` 11。
- 文件数量本身暂未显示失控：基础层均较小，算法按 ILT/MBOPC/DiffOPC 隔离；需要重点审查的高复杂度文件是 `main/offline_inputs.py`（858 行）、`main/run_mbopc_frontend.py`（444 行）以及三个约 275–312 行 runner/solver。
- 规划表存在一处状态漂移：总表中的阶段 64 仍写 `in progress`，但后续阶段 66–77 和结论明确四阶段均已完成；这是项目状态文档错误，不是生产架构错误。
- `pyproject.toml` 的运行依赖没有列出生产预检代码使用的 `psutil`，却把它放在 dev 可选依赖；`requirements.txt` 虽可能补足，标准项目安装契约仍不一致，需要进一步核实实际导入与直接运行路径。
- 本轮不因一个文件行数大就预判拆分；后续以职责数量、数据所有权、重复调用链和变更原因是否一致为判断依据。
- 依赖静态扫描未发现基础层反向依赖迭代算法：`layout` 独立，`geometry -> layout`，`opc.input -> geometry/layout`，各 `opc.iteration` 才依赖 `lithography/evaluation`；`main` 负责组合。现有核心依赖方向符合 AGENTS.md。
- `ContourBatch -> SegmentBatch -> MBOPCProblem` 是逐层增加信息，不是字段复制：Contour 保存拓扑，Segment 保存边缓存/分段参数，Problem 保存 mask/config/grid/owner/membership。`SegmentBatch.contours` 是对象引用，不是数组副本；当前没有重复保存 layer/ring_id/is_hole。
- 生产函数 AST 规范化后没有完全相同的函数体；粗略未调用扫描只命中 `__enter__/__exit__/__len__/__iter__` 协议方法，没有确定的死函数证据。
- `main/offline_inputs.py` 同时承担四类职责：原子产物 I/O、最终光刻结果保存、版图→raster/segment 物化、NPZ 归档校验/版本迁移，以及自己的 CLI。虽然都围绕文件级边界，但 858 行已明显增加导航和变更耦合；是否拆分需结合跨 runner 重复代码判断。
- `psutil` 已确认在 `opc/input/preflight.py` 顶层生产导入，且多个生产 runner 调用；`pyproject.toml` 却只把它列入 `[project.optional-dependencies].dev`，`requirements.txt` 也把它放在“开发、测试与性能基准”注释下。这是确定的依赖契约错误：按 `pip install .`（尽管用户日常不要求安装）或只按“运行依赖”理解 requirements 都会缺生产依赖。
- 多个 runner 各自重复完成“输入解析/物化→模型→优化→重建/评价→原子产物→summary→资源统计”。这不是函数体复制，但存在流程骨架重复；不宜立刻建立通用 runner 框架，因为 ILT 与边段 OPC 数据流差异大，应只抽取稳定且已重复的 I/O/summary 小块。
- 入口层存在确定的私有接口泄漏：6 个 runner 从 `main.offline_inputs` 导入 `_atomic_json/_atomic_npz/_atomic_png/_exact_dbu` 等私有函数。下划线表明内部实现，但它们实际上是跨模块共享 API；这使 858 行的 offline 模块兼任工具箱，也让拆分困难。建议把“原子产物 I/O”提为一个小型公开模块，而不是建立 runner 基类。
- Layer CLI 解析存在三份真正重复实现：`main.configuration.parse_layer_spec`、`run_layout_geometry.parse_layer`、`run_mbopc_frontend.parse_layer`；`offline_inputs.parse_layer` 只是薄包装，`run_mbopc` 又引用这个包装。应统一直接使用 `parse_layer_spec`，删除三份重复/转发函数。
- `run_simpleilt.py` 与 `run_ilt.py --method simple` 是确定的功能重复：两者都完成相同输入、SimpleILT 优化、三工艺评价、NPZ/PNG/final lithography/summary；但输出格式名、默认值、时间/内存字段不同。保留两个独立实现会产生行为漂移，最佳方向是让兼容入口只委托统一 `run_ilt(method="simple")`，或明确废弃一个入口。
- `main/offline_inputs.py` 作为“归档契约 + 输入物化 + 最终产物 I/O + CLI”已违反单一变更原因；可按现有调用方最小拆为公开 `main/artifacts.py`（原子 JSON/NPZ/PNG 与最终光刻保存）和仍保留归档/物化的 `offline_inputs.py`。不建议继续细拆 raster/segment 两个文件，因为校验、预检和版本 metadata 有大量共用不变量。
- `PhysicalMask.region`（KLayout Region）与 `SegmentBatch.contours`（NumPy CSR）同时常驻是有意的速度/内存权衡：target/current tile 原生栅格化需要 Region，边段重建和可微 raster 需要 CSR；两者没有字段复制但几何信息重复。对当前完整内存 problem 合理，对十亿级边段不成立；真正解决方案是既定 macro shard/按块物化，不是删除其中一个表示后让热路径反复转换。
- `layout/source.py` 的 `read_layout` 用宽泛 `except Exception` 统一包装第三方 KLayout 读取错误。这里位于文件 I/O 边界且保留 cause，不属于明显错误；不应为了静态洁癖缩窄到不完整的异常列表。
- Git 跟踪了 `.gitignore` 已排除的 `output/mbopc` 四个生成产物，另有根目录 `gcd_45nm.png`、`result.gds`。这会污染仓库和让示例结果陈旧；由于它们可能是用户有意保留的基线，本轮只报告，后续清理必须逐项确认，不能直接删除。
- 两个边段求解器重复 `_owner_indices/_owner_segments` 和 `_target_tile` 的语义及实现，名称不同但代码骨架相同。前者属于稳定的 Problem 查询，适合成为 `MBOPCProblem.owner_segments_for_core()` 或一次性公共函数；后者还绑定各自 config，不必为消除几行重复强行抽象，可仅复用“参考 tile uint8 缓存”的小函数。
- ILT 三种扩展算法从 `ilt.simple` 导入 `_image_batch/_resize_image/_curvature_loss/_smooth_sigmoid_mask` 私有符号。与 offline I/O 一样，这是模块边界表达错误：这些已经是当前多算法共享的公共实现，却留在“simple 算法”的私有命名空间。建议将四个 helper 迁到一个紧凑的 `ilt/common.py` 或 `ilt/_common.py`；若用 `_common`，只能包内导入并明确它是包内稳定接口。
- ILT 各配置的数据字段有合理重复（每个算法可独立替换且默认/约束不同），不建议建立继承基类。将 config 合并会把算法参数耦合，并不能减少运行期内存；当前 dataclass `slots/frozen` 已足够紧凑。
- `MBOPCProblem` 的 owner/membership 直接数组字段虽然较多，但所有字段都有热路径、归档或诊断调用；将 ownership 再包成独立类只会恢复此前已经删除的结构层，不建议。
- 所有求解器的类型标注直接绑定 `ICCAD13Lithography`，但测试中已有至少三类结构兼容的替身模型，证明当前真实边界其实是 `device + config.canvas/print_threshold + condition() + forward_many()`。项目目标又明确会替换光刻模型，因此应在顶层 `lithography` 定义一个最小 `Protocol`（以及最小 config view），让迭代层依赖能力契约而非 ICCAD13 具体类。现在已有多个生产/测试调用方，这不是空抽象；但 runner 仍可明确实例化 ICCAD13，不需要模型注册器或工厂。
- 当前 `ProcessCondition` 仍是 ICCAD13 的具体 dataclass，并被算法用于类型检查；若替换模型仍采用同一 condition 结构可保留。若新模型条件不同，Protocol 应把 condition token 作为泛型/不透明值，而不应让 OPC 求解器解析 kernel/dose 字段。
- 测试覆盖很强（226 项，收集成功），尤其核心几何、跨 core、孔洞、斜边、极性、流式屏障和真实模型；但缺少“依赖方向/公共接口不私有导入”的架构测试，因此私有 helper 泄漏和模型具体耦合不会被现有行为测试捕获。
- 文档数量达到 39 份，其中许多阶段性开发/测试报告具有审计价值，不建议机械合并；但主手册同时存在 `development_manual.md` 与超长 `项目开发手册.md`，再叠加接口参考与调用图，导航成本偏高。可保留历史报告，只明确一份主开发手册、一份测试手册、一份接口参考、一份架构图作为当前事实源，其余标注为阶段归档。
- `MBOPCProblem` 已同时服务 Simple MB-OPC 与 DiffOPC，名称和归档格式 `myopc.mbopc-input` 已落后于实际职责。建议直接重命名为 `EdgeOPCProblem` / `myopc.edge-opc-input`，并为旧 v2/v3 归档保留读取迁移；不要新增包裹类或双份字段。该项是清晰性改进但影响公共 API/文档/归档，优先级低于无行为变化的精简项。
- `pyproject.toml` 项目名和描述仍是早期 `myopc-layout-geometry`，`task_plan.md` 顶部也仍以 Layout/Geometry 为总目标；当前项目已经包含完整 lithography/evaluation/ILT/MBOPC/DiffOPC。这是项目身份与现状漂移，会误导依赖管理和新人导航，应更新为整个 MyOPC 的描述。
- 9 个可直接运行脚本各自保留同一段 `PROJECT_ROOT/sys.path` 注入。它是用户“直接 python 文件、不安装”的明确要求所致，虽然文本重复但不宜抽函数：在导入项目模块之前必须执行，抽到项目模块反而无法导入。可接受为入口样板。
- `preflight.py` 的 `_fragment_counts` 与生产 `fragment_edges` 保留两份切分公式；这是一个潜在正确性漂移点，但预检必须在完整边段物化前执行，直接调用生产函数会违背内存保护。最佳最小改法是把纯向量“每边计数公式”放到输入层共享 helper，而不是让 preflight 依赖/分配完整 SegmentBatch。
- `opc.input.preflight` 跨包访问 `LayoutDB._native_layout/_native_cell/_native_layer_index`，是确定的封装泄漏。预检需要低层递归迭代器以避免 Region 物化，这个需求合理；错误在于 layout 没有公开“只读原生层级扫描”能力。修复必须最小修改受保护 `layout/`（例如公开受控 iterator/context 方法），需用户逐次授权，不能在 OPC 侧复制更多 KLayout 细节。
- 包内 `builder -> ownership._build_ownership` 的私有导入是同一 `opc.input.edge` 包内部实现协作，可接受；全项目模块依赖图没有循环。
- `prepare_problem()` 是公共 API，也可绕过 runner/preflight 直接构造完整 membership；`_build_ownership` 在 `np.repeat` 前没有显式 memory/count 上限。因此安全保证目前是“真实版图入口必须先 preflight”，而不是 Problem builder 自身有界。对已经物化的小 ROI 公共 API 这可接受，但文档/类型名应明确它是 in-memory builder；未来 shard builder 不能复用该函数偷偷分配全局 CSR。
- 当前接口参考 `module_interface_reference.md` 仍把 DiffOPC 描述为“原型、尚未完成连续 EPE/完整产物验收”，与阶段 74–77、测试和 runner 现实冲突；`task_plan.md` 总表阶段 64 也仍为 in progress。两者属于确定的事实源漂移，会误导架构理解，应立即只改文档。
- 静态 fan-in 显示 `main.offline_inputs` 被 6 个 runner 依赖，是入口层最明显的耦合中心；`main.run_mbopc` fan-out 12 个模块，但作为组合根可接受。没有循环依赖。
- `OwnershipError` 只有定义和包级导出，没有任何抛出点、捕获点或测试；当前 ownership 不变量实际抛 `ValueError`。这是确定的无实现公共抽象，应删除，或将真实 ownership 错误统一改抛它；从精简角度更建议删除，除非用户明确要稳定领域异常 API。
- `GeometryError`/`OPCError` 基类本身不直接抛出但由 runner 统一捕获子类，属于有效抽象；其他具体异常均有真实抛出点，不应删除。
- 阶段 88 最终验证：Ruff、compileall、226 项全量测试（60.08 秒）、文档链接/围栏、diff whitespace 和保护目录差异全部通过；评审结论记录于 `doc/current_architecture_review.md`，验证记录位于 `doc/current_architecture_review_test_report.md`。

# 2026-08-12：阶段 89–93 P1/P2 实施决策

- P1/P2 以 `doc/current_architecture_review.md` 第 4、5 节为唯一范围；P3 明确不在本轮实施。
- `layout` 修复应公开“构造受生命周期约束的原生递归图形迭代器”，而不是公开整个原生 Layout/Cell 或让 preflight 继续调用 `_native_*`。
- 光刻抽象只使用静态 `Protocol` 表达当前 solver 已实际消费的能力；runner 继续直接构造 ICCAD13，不新增运行期分派。
- SimpleILT 兼容入口需要保留结果对象返回值，因此统一 `run_ilt` 的内部结果应提供一个不破坏现有 `run_ilt -> summary` API 的私有/可选返回通道，或抽取共享执行函数；优先选择最少包装和单一产物路径的方案。
- `LayoutDB.recursive_polygon_shapes` 是本轮唯一新增的 layout 公共能力：返回仍绑定数据库生命周期的原生 iterator，不暴露 Layout/Cell/layer index，也不物化 Region。
- preflight 从预算按 32 B/membership 推导构造硬上限；`prepare_problem(max_memberships=...)` 保持旧小 ROI 调用兼容，真实输入路径显式传入该上限。
- P2 实施后 `offline_inputs.py` 只保留共享输入物化、归档版本/损坏校验和 CLI；所有 runner 改从 `main.artifacts` 使用公共原子产物函数，不再把下划线私有函数当 API。
- SimpleILT 的兼容要求不需要第二个执行对象或 runner 基类：`run_ilt(return_result=True)` 返回同次执行的 result/summary，默认仍只返回 summary；`run_simpleilt` 只负责固定 `method="simple"` 和历史默认值。
- 共享边段计数 helper 必须使用不会与 `fragment_edges` 内部展开数组冲突的动词名 `count_edge_fragments`；固定随机种子回归已证明其逐边计数与真实 SegmentBatch 相同。

# 2026-08-13：Macro 物化现状与实施决策

- 当前 `ShapeQuery.materialize()` 先用原生 ROI 迭代器取得相交候选，随后与查询框精确求交；把该结果逐 macro 送入 `prepare_problem()` 会把裁剪框线错误识别成可移动物理边。
- 当前 core 路径没有虚假边，是因为完整处理 ROI 只提边一次，core 仅负责 segment owner/context；本次新增的是更外层 CPU macro 调度边界。
- 用户确认物化规则为“只加载与 `macro ownership + roi_halo` 相交的完整图形”，完全不相交的图形不加载；栅格化时才裁到 tile context。
- 用户选择首个可用 Macro 前端阶段：`run_mbopc_frontend.py` 验证逐 macro 物化、提边、ownership 和栅格拼接，不提前锁定磁盘 shard 格式，也不宣称现有求解器已支持 out-of-core 多轮。
- 参数显式迁移为 `roi_halo_nm` 和 `tile_halo_nm`；用户授权 `layout/` 增加独立未裁剪批量物化入口，现有精确 `materialize()` 语义保持不变，`geometry/` 不修改。
- ILT 不提边，精确裁剪不会产生可移动假边；未来大版图 ILT 仍需 tile 光学上下文和 ownership-only 像素提交，但不复用 SegmentBatch。
- ICCAD13 资产的 35×35 数据是频域 Hopkins 核，不表示空间域只影响 17 pixel；不能据此承诺有限光学半径。`tile_halo` 是用户按精度收敛选取的有效截断范围，`roi_halo` 至少再覆盖最大允许边位移。

# 2026-08-13：Layout 层级接口轻量化决策

- `layout/hierarchy.py` 只有 `LayoutDB.hierarchy_summary()` 和一项直接测试消费；OPC、runner 和当前 planner 均不读取这些 dataclass。
- 用户确认删除 bbox、实例记录数和阵列展开数，只保留完整 Cell 层级；返回文件内全部 Cell，不限于当前选中 top。
- GDS Cell 层级是 DAG。共享 Cell 采用邻接字典保存：每个 Cell 只定义一次，可同时出现在多个父 Cell 的子列表中；不展开 SREF/AREF occurrence，避免阵列导致内存膨胀。
- KLayout `Cell.each_child_cell()` 原生返回去重后的直接子 Cell 索引，适合一次遍历生成 `dict[str, tuple[str, ...]]`，无需逐实例统计。
- 实现回归确认 100×100 AREF 与同父 Cell 重复 SREF 均只生成一个直接子名称；完整字典同时包含当前选择 top 之外的独立 Cell，叶节点明确为 `()`。
- 最终调用点审计确认旧层级模块/类型/方法在生产代码和测试中零匹配；历史报告只保留修改前证据并标注不再代表当前 API。
- 本轮没有新增缓存、兼容包装或 planner 专用对象；低引用扫描未发现因修复测试而遗留的辅助函数。
- 小版图实测确认旧“完整 ROI 精确裁剪”与新“完整 occurrence”在处理框最外边缘的长边分段相位不同：旧路径从裁剪端点起算，新路径从真实图形端点起算。正确验收应比较同一完整 occurrence 的单 macro 与多 macro 分区，不得以带处理框假边的旧分段坐标为真值。

# 2026-08-14：当前规则符合性修正发现

- simple MB-OPC 直接把 DBU probe 除以像素尺寸，遗漏像素中心的 `-0.5`；8 DBU 单像素矩形回归中旧公式有效探针为 0，修正后为 10。
- DiffOPC 软栅格器对每个 tile 重复执行 7 次设备标量 value 检查，`.item()` 强制 CUDA 同步；这些值已由内部 problem/优化状态保证，保留 shape/dtype/device 检查即可。
- 同进程 256² 空 tile、预生成像素中心、200 次 CUDA 对照：旧同步检查等效路径 1.844 ms/次，当前路径 0.288 ms/次，约 6.4 倍。
- 三个边段输入入口对 nm→DBU 的规则不一致：分段端点长度必须整数 DBU，最大位移属于连续优化量，应允许小数 DBU。共享换算仅有三个当前调用方。
- `MacroPreparation.__post_init__` 只重复校验 `prepare_macro` 自己刚构造的数组，没有第二构造入口；删除后不改变数据所有权。macro segment 下标不是跨 macro 全局 ID，只有 tile ID 全局。
- `OwnershipError` 没有抛出、捕获或测试调用方；删除比把现有 `ValueError` 强行改成新领域异常更符合最小代码原则。
