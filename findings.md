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
