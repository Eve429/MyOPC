# 配置系统重构开发报告

日期：2026-08-18。规格：`D:\浏览器下载\MyOPC_config_system_refactor_spec.md`
（用户 2026-08-18 批准，含 7 项按仓库现状调整，见下）。

## 提交链

```text
94cd621  refactor(main): MB-OPC 合并为单入口 run_mbopc（批 0，上轮已批方案）
1db593d  feat(main): 建立统一配置体系 configuration.py（批 1）
8f71b5a  refactor(config): 全流程迁移统一配置体系（批 2+3）
（批 4）docs(config): 报告与手册同步
```

## 1. 修改文件

生产：`main/configuration.py`（新）、`_macro_pipeline.py`、`_mbopc_workflow.py`、
`run_macro_pipeline.py`、`run_single_pass.py`、`run_mbopc.py`（批 0 新）；
配置：5 个 `config/*.toml`；测试：`test_configuration.py`（新）+ 四个流程
测试文件。删除：`run_mbopc_single_macro.py`、`run_mbopc_multi_macro.py`。

## 2. 原 Config / loader 清单 → 最终 Config

| 原结构 | 去向 |
|---|---|
| MacroCommonConfig + MacroPipelineConfig（P1-3 引入的基类+子类） | 删除；字段拆入 Layout/Partition/Lithography/Edge/Output |
| MBOPCRunConfig + load_config（simple） | 删除；run_mbopc 直接 load_config 六 Config |
| GradientMBOPCRunConfig + load_gradient_config | 删除；同上 |
| run_single_pass 的 SinglePassConfig(MacroCommon) + load_config | 删除；SinglePassConfig（configuration.py，[single_pass] 段） |
| load_macro_common_config / load_macro_config | 删除 |
| load_validation_deltas + [iteration] 手工解析 | 删除；ValidationConfig（__post_init__ 冻结 ±2nm） |
| SimpleMBOPCConfig / GradientMBOPCConfig（opc/iteration/mbopc/） | **保留**（§25 豁免，见 §5） |

最终 Config（main/configuration.py，全部 frozen+slots）：
LayoutConfig[layout] / PartitionConfig[partition]（互斥在 post_init）/
LithographyConfig[lithography]（canvas=256 冻结 + device 枚举）/
EdgeConfig[edge] / MBOPCConfig[mbopc] / GradientConfig[gradient] /
SinglePassConfig[single_pass] / ValidationConfig[iteration] /
OutputConfig[output]（work_dir: None=不适用，消费方检查）。
CONFIG_SECTIONS 声明式映射，load_config 无算法分支。

## 3. 字段迁移表（TOML 键名全部保持）

| 旧段.键 | 新段.键 | Config | 备注 |
|---|---|---|---|
| input.*（layout/top_cell/layer/datatype/polarity） | layout.* | Layout | 段更名 |
| grid.*（macro_grid/macro_size_nm/core_size_nm/context_nm） | partition.* | Partition | 段更名 |
| lithography.pixel_nm/canvas_pixels | 不变 | Lithography | |
| 算法段.device | lithography.device | Lithography | 键挪段（device 有默认 auto） |
| edge.*（corner/segment/max_displacement/miter） | 不变 | Edge | 新增 Config（两轮遗漏补） |
| mbopc.*（iterations/initial_step/decay/epe/batch/cache） | 不变 | MBOPC | 去 device/save/show |
| gradient_mbopc.* → gradient.*（iterations/lr/weights/epe/batch/cache） | gradient.* | Gradient | 段更名 |
| iteration.displacement_nm（single_pass.toml） | single_pass.displacement_nm | SinglePass | 段更名（避开与验证 [iteration] 冲突） |
| iteration.round_deltas_nm（macro_pipeline.toml） | 不变 | Validation | |
| 算法段.save_final_lithography/show_progress | output.* | Output | 键挪段（默认 false） |
| output.work_dir/final_layout/final_cell_mode | 不变 | Output | work_dir 变可选 |

运行时派生值（不进 Config，workflow 内联）：step_dbu/epe_dbu/
target_cache_bytes/pixel_dbu/canvas 校验/lr_dbu（Decimal 相除转 float）。

## 4. load_config 最终接口与行为

`load_config(path, *config_types) -> tuple`（顺序一致）；TOML 单次读；
未知 section 报错（含路径）；**出现的每个 section**（含未请求）查未知字段；
required=dataclass 无默认字段；default=dataclass 默认；Path 三态
（相对 TOML 目录/绝对/~/expanduser）；类型集 str/bool/int（拒 bool 冒充）/
float/Decimal/Path/MaskPolarity/tuple[int,int]/Literal/X|None。

## 5. 保留 Config 豁免说明（§25 条款）

`SimpleMBOPCConfig`/`GradientMBOPCConfig`（opc/iteration/mbopc/）：DBU 域
求解器输入包（nm→DBU 换算产物），消费者是 optimize_macro/
optimize_gradient_macro；字段事实源是各算法 Config + 运行时 dbu 换算；
不是 main 层中转（不再从任何 RunConfig 复制——workflow 直接从算法 Config
+dbu 构造）。等价于规格 §6 允许的 Runtime 语义，保留原名（solver 层稳定
契约，tests/opc/iteration 大量直接构造）。

## 6. 与规格的偏差（全部经用户批准的计划或在案）

1. GradientConfig 解读 = gradient 方法专属段（规格 §4.4 的 probe/
   sampling 字段不存在，不虚构）；
2. ILTConfig 未建（ILT 未迁移，反投机）；
3. final_cell_mode 保留（macro_cells 是真实第二模式）；
4. 新增 EdgeConfig（规格与计划首批均遗漏的算法无关段）；
5. 新增 ValidationConfig + SinglePassConfig 段名 [single_pass]（规格
   未考虑验证管线与单遍的 [iteration] 同名冲突）；
6. 4 个 config 的 layout 切 bench_30um_clear.gds（gcd_45nm.gds 已被
   用户删除，非本重构引入）；
7. 数值一致验收口径：旧 gcd_45nm 基线数字（7264/5893/5640/4892 等）
   的对照版图已不存在，改为 bench_30um_clear.gds 全量测试 + 三 smoke
   端到端（配置层重构不改算法路径，全量 444 passed 为等价证据）。
