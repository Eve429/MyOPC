# 最终光刻结果输出开发报告

本阶段只增加结果保存，不改变 layout、geometry、边段切分、owner、迭代更新或光刻模型数值路径。

- SimpleILT 保存 `final_lithography.npz`，字段为 `mask`、`nominal`、`dose_max`、`defocus_min`，并按请求保存 PNG。
- 在线和离线 MB-OPC 按 batch 构造 core context mask，模型输出立即转 CPU、裁 ownership、写盘并释放 tensor。
- `manifest.json` 记录格式版本、方向、pixel、core/context box、ownership-only 原点和尺寸。
- halo 只参与模型输入，不进入最终 tile；共享既有原子 I/O，没有新增空工具层或缓存层。
