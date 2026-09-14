# PHM 数据目录

本目录只存放 NASA C-MAPSS FD001 的原始数据与预处理结果，不再承载项目教程或简历
说明。完整文档已经迁移到：

- [FD001 数据、训练与模型服务教程](../docs/FD001_PHM_GUIDE.md)
- [Doro Coding + PHM Agent 项目经历与面试指南](../docs/DORO_PROJECT_EXPERIENCE.md)
- [Agent 在线训练快速教程](../README.md)

## 目录结构

```text
extensions/phm/data/
├── README.md
├── raw/                     # 下载并校验后的 FD001 原始文件，Git 忽略
│   ├── train_FD001.txt
│   ├── test_FD001.txt
│   └── RUL_FD001.txt
└── processed/               # 预处理结果，Git 忽略
    ├── fd001.npz
    └── fd001_manifest.json
```

## 生成数据

所有命令从仓库根目录执行：

```powershell
uv sync --project extensions/phm --python 3.11 --extra test
uv run --project extensions/phm phm download
uv run --project extensions/phm phm prepare
```

`phm download` 优先使用 NASA 数据源，并对三个 FD001 文件执行仓库固定的 SHA-256
校验；官方源不可用时会使用固定提交的只读镜像，仍然执行相同哈希校验。

`phm prepare` 会按发动机 ID 划分训练集和验证集，只使用训练发动机拟合特征选择和
标准化参数，再生成时序窗口，从而避免同一发动机的相邻窗口跨集合造成数据泄漏。

## 两类产物

### `raw/`

保存 NASA 原始文本文件。它们可以重新下载、体积较大且不属于源代码，因此由 Git
忽略。不要手工编辑这些文件；若哈希不一致，应删除异常文件后重新运行下载命令。

### `processed/`

`fd001.npz` 保存训练、验证和官方测试所需的 NumPy 数组；
`fd001_manifest.json` 保存数据来源、原始文件哈希、随机种子、发动机切分、窗口长度
和特征清单，用于复现实验与排查数据版本。

如果改变随机种子、验证比例、窗口长度或预处理逻辑，应重新运行 `phm prepare`，并
将这次数据配置与后续模型版本一起记录。

## 检查数据是否准备完成

```powershell
Test-Path extensions/phm/data/raw/train_FD001.txt
Test-Path extensions/phm/data/processed/fd001.npz
Test-Path extensions/phm/data/processed/fd001_manifest.json
```

三项均为 `True` 后，才能进行直接训练或通过 Agent 提交异步训练任务。

> FD001 是航空发动机退化仿真数据，仅用于算法学习、工程验证和 Agent 工具调用演示，
> 不能直接作为真实航空器维修、适航或放行依据。
