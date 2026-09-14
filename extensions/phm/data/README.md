# PHM 数据目录与上手教程

这里是 Doro PHM 扩展的数据入口，保存 NASA C-MAPSS FD001 的原始文件和预处理结果。
本文从数据准备讲到训练、Agent 调用、测试、面试和简历写法，可以独立作为快速教程；
更完整的原理和系统设计见：

- [FD001 技术实现与复现指南](../docs/FD001_PHM_GUIDE.md)
- [Doro Coding + PHM Agent 项目经历与面试指南](../docs/DORO_PROJECT_EXPERIENCE.md)
- [PHM 扩展与 Agent 在线训练教程](../README.md)

## 一、先理解当前项目边界

FD001 是 NASA C-MAPSS 生成的航空发动机退化仿真数据，包含 100 台训练发动机和
100 台截断测试发动机，只有一种运行工况和一种故障模式。每行包含发动机编号、循环
序号、3 个工况参数和 21 个传感器值；测试集真实剩余寿命单独保存在
`RUL_FD001.txt`。

Doro 当前实现的是“静态数据上的按需训练与在线任务控制”：用户可在 Agent 中提交
LSTM/Transformer 训练、查看 epoch 进度、检查候选、发布和回滚，但系统不接收实时
传感器流，也不做样本到达后立即更新权重的持续学习。模型数值由本地 PyTorch 代码
计算，LLM 只负责任务编排和带边界的结果解释。

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

模型和任务状态不在本目录：

```text
extensions/phm/
├── artifacts/                   # 正式模型、候选模型、活动版本注册表
└── runtime/                     # SQLite 任务队列、Worker 状态与日志
```

`raw/`、`processed/`、`artifacts/` 和 `runtime/` 都是可再生成或运行时数据，默认不作为
普通源代码提交。需要复现实验时，应保留数据哈希、manifest、模型配置和指标记录。

## 二、准备环境

所有命令从仓库根目录执行：

```powershell
uv sync --project extensions/phm --python 3.11 --extra test
```

这会创建或更新 PHM 自己的 `extensions/phm/.venv`。它与 Doro 主项目根目录的
`.venv` 相互独立，避免 PyTorch 等领域依赖污染通用 Agent 环境。

## 三、下载并校验原始数据

```powershell
uv run --project extensions/phm phm download
```

`phm download` 优先使用 NASA 数据源，并对三个 FD001 文件执行仓库固定的 SHA-256
校验；官方源不可用时会使用固定提交的只读镜像，仍然执行相同哈希校验。

下载成功后应存在：

```powershell
Test-Path extensions/phm/data/raw/train_FD001.txt
Test-Path extensions/phm/data/raw/test_FD001.txt
Test-Path extensions/phm/data/raw/RUL_FD001.txt
```

三项应均为 `True`。若哈希校验失败，不要继续使用该文件；删除异常的单个原始文件并
重新下载。下载测试使用 Mock 网络验证主源/镜像回退和哈希逻辑，不会在运行测试时
真实访问 NASA。

## 四、无泄漏预处理

```powershell
uv run --project extensions/phm phm prepare --seed 42 --validation-fraction 0.2
```

`phm prepare` 会按发动机 ID 划分训练集和验证集，只使用训练发动机拟合特征选择和
标准化参数，再生成时序窗口，从而避免同一发动机的相邻窗口跨集合造成数据泄漏。

当前固定窗口长度为 30 个周期，训练 RUL 标签上限为 125。处理过程是：

```text
原始 26 列
  → 按发动机 ID 划分 80% 训练、20% 验证
  → 只在训练发动机上剔除近常量特征并拟合均值/标准差
  → 用同一组统计量变换训练、验证和测试数据
  → 每台训练发动机生成长度 30 的滑动窗口
  → 每台测试发动机取最后 30 个周期作为一个测试样本
```

这里最重要的面试点是“按发动机切分，而不是按窗口随机切分”。如果先生成全部窗口再
随机划分，同一发动机的高度重叠窗口可能同时进入训练集和验证集，造成指标虚高。

## 五、理解数据产物

### `raw/`

保存 NASA 原始文本文件。它们可以重新下载、体积较大且不属于源代码，因此由 Git
忽略。不要手工编辑这些文件；若哈希不一致，应删除异常文件后重新运行下载命令。

### `processed/`

`fd001.npz` 保存训练、验证和官方测试所需的 NumPy 数组；
`fd001_manifest.json` 保存数据来源、原始文件哈希、随机种子、发动机切分、窗口长度
和特征清单，用于复现实验与排查数据版本。

`fd001.npz` 主要字段如下：

| 字段 | 含义 |
|---|---|
| `train_x` / `train_y` | 训练窗口及其 RUL 标签 |
| `validation_x` / `validation_y` | 按发动机隔离的验证样本 |
| `test_x` / `test_y` | 官方测试发动机末端窗口及真实 RUL |
| `train_unit_ids` / `validation_unit_ids` | 可审计的发动机划分 |
| `selected_feature_indices` | 保留下来的特征索引 |
| `feature_mean` / `feature_scale` | 仅由训练发动机拟合的标准化参数 |
| `window_size` | 模型输入的时间窗口长度 |

如果改变随机种子、验证比例、窗口长度或预处理逻辑，应重新运行 `phm prepare`，并
将这次数据配置与后续模型版本一起记录。

## 六、直接训练和测试模型

不经过 Agent 时，可以用 CLI 直接训练。下面显式写出 `standard` 等价参数，避免依赖
CLI 默认值：

```powershell
uv run --project extensions/phm phm train --model lstm --version v2 `
  --epochs 20 --batch-size 128 --learning-rate 0.001 --patience 5 `
  --seed 42 --device auto

uv run --project extensions/phm phm train --model transformer --version v2 `
  --epochs 20 --batch-size 128 --learning-rate 0.001 --patience 5 `
  --seed 42 --device auto
```

`device auto` 会依次选择 CUDA、Apple MPS、CPU；本机若安装的是 CPU 版 PyTorch，
就只会使用 CPU。CUDA 训练需要安装与显卡驱动匹配的 PyTorch CUDA wheel，不能只靠
把参数改成 `cuda`。AMP 仅用于 CUDA，可在合适环境中增加 `--amp`。

查看保存指标和单台发动机预测：

```powershell
uv run --project extensions/phm phm evaluate
uv run --project extensions/phm phm predict --unit-id 42 --model lstm --version active
uv run --project extensions/phm phm predict --unit-id 42
```

最后一条未指定模型，会比较 LSTM 和 Transformer。模型选择主要依据验证集，不应反复
使用测试集挑版本。项目使用 RMSE、MAE 和 NASA Score；NASA Score 对高估 RUL 的
惩罚更重，因为“以为设备还能运行很久”通常比过早维护风险更高。

## 七、在 Doro Agent 中按需异步训练

根目录 `.mcp.json` 应同时配置：

- `phm`：6 个只读工具，负责质检、预测、比较、指标和趋势证据；
- `phm-admin`：10 个管理工具，负责训练队列、Worker、候选发布与回滚。

启动 Doro 后可这样操作：

```text
/phm-training 查看 Worker、队列和最近任务，不要改变状态。
/phm-training 训练一个 LSTM 候选模型，版本 lstm-v2。不要发布。
```

如果没有指定任何超参数，Skill 应先展示 `standard` 预设（20 epochs、batch 128、
lr 0.001、patience 5、seed 42、device auto、AMP off），询问是否同意，而不是立即
提交。用户确认后任务写入 SQLite，管理服务会自动启动或复用后台 Worker，一般无需
单独保留 PowerShell 窗口。关闭 Doro 不会删除排队任务，后台 Worker 可继续训练；
队列空闲约 5 分钟后 Worker 自动退出。

继续查询和发布：

```text
/phm-training 查询任务 JOB_ID 的状态、epoch 进度、Worker 心跳和错误。不要发布。
/phm-training 读取任务 JOB_ID 的配置和候选指标，判断是否值得发布，但不要发布。
/phm-training 我确认发布任务 JOB_ID 对应的候选模型。
/phm 使用活动 LSTM 模型预测 FD001 测试发动机 42，并报告模型版本和指标边界。
```

训练成功只产生候选，不会自动替换活动模型。发布/回滚的显式确认目前主要由
`/phm-training` Skill 和 Doro 权限策略约束；`phm-admin` CLI 与管理服务本身尚没有
不可绕过的审批令牌，因此管理 CLI 只应交给受信任操作者。

## 八、检查数据是否准备完成

```powershell
Test-Path extensions/phm/data/raw/train_FD001.txt
Test-Path extensions/phm/data/processed/fd001.npz
Test-Path extensions/phm/data/processed/fd001_manifest.json
```

三项均为 `True` 后，才能进行直接训练或通过 Agent 提交异步训练任务。

进一步检查 manifest：

```powershell
Get-Content extensions/phm/data/processed/fd001_manifest.json
```

重点核对 `dataset`、`seed`、`validation_fraction`、`window_size`、`rul_cap`、发动机
ID 列表、特征清单和三个原始文件哈希。

## 九、测试是怎样运行的

PHM 测试与主项目回归必须分开运行：

```powershell
# PHM 独立环境
uv run --project extensions/phm pytest -q extensions/phm/tests

# Doro 主项目环境
uv run pytest -q
```

当前验证结果是 PHM `32 passed`、主项目 `234 passed`。两组数字来自不同环境，不是
一次运行得到的 266 项总测试。

PHM 测试覆盖合成数据预处理、下载回退与哈希、模型前向、指标公式、临时 SQLite
任务状态、MCP 工具 Schema、Worker Supervisor，以及在微型合成数据上执行一轮真实
CPU LSTM 训练、候选发布、推理和回滚。它不会完整复训 FD001，不验证文档中的 v1
指标，也不覆盖真实网络、CUDA/MPS、长期后台进程或生产压力。

主项目测试覆盖 Agent 循环、CLI/命令、模型配置、MCP stdio/HTTP、权限、会话、技能、
状态 UI、用量和价格计算等；外部模型 Provider 使用 Mock，不消耗真实 API Key。

## 十、常见问题

### 下载失败或哈希不一致

先检查网络和代理。程序会从 NASA 主源切换到固定镜像，但两个来源都必须通过同一组
SHA-256。哈希错误通常说明下载不完整或文件被修改，不应跳过校验。

### 任务一直停在 `queued`

让 Agent 查询 `get_training_worker_status`。若 `alive=false`，调用
`ensure_training_worker`；CLI 等价命令为：

```powershell
uv run --project extensions/phm phm-admin worker-start
```

尚未领取的 queued 任务会留在 SQLite 中，稍后启动 Worker 后会继续按队列顺序执行，
无需重复提交。排错日志位于 `extensions/phm/runtime/worker.stderr.log`。

### 修改窗口或预处理后模型加载失败

重新运行 `phm prepare`，并训练一个新版本。不要让旧模型权重与新的输入维度或
标准化统计量混用，也不要覆盖可用于回滚的正式版本。

### Transformer 出现 nested tensor 警告

当前 `norm_first=True` 可能让 PyTorch 提示 nested tensor 优化未启用。这是性能警告，
不是预测正确性失败；测试仍应正常通过。

## 十一、面试讲解指南

### 30 秒版本

> 我把 Doro 从 Coding Agent 扩展为 Coding + PHM 统一智能体。在 NASA C-MAPSS
> FD001 上，我按发动机 ID 做无泄漏切分，只用训练设备拟合标准化器，并构造 30 周期
> 窗口训练两层 LSTM 和轻量 Transformer。推理侧封装成 6 个只读 MCP 工具，训练侧
> 用 10 个管理工具、SQLite 队列和带租约/心跳的独立 Worker 实现按需异步训练；模型
> 先进入候选区，经检查后再原子发布并支持回滚。LLM 负责编排，确定性模型负责数值。

### 面试官可能追问

**为什么按发动机划分？** 为防止同一设备的重叠窗口跨训练/验证集造成数据泄漏。

**为什么限制 RUL 为 125？** 早期退化不明显，截断标签可减少大 RUL 区间对损失的
支配；这是常见建模假设，不代表物理寿命上限。

**为什么同时做 LSTM 和 Transformer？** LSTM 是稳定、轻量的序列基线；Transformer
便于比较全局时序建模。项目不预设 Transformer 必然更好，最终按验证指标和工程成本
评估。

**为什么拆两个 MCP？** 普通推理获得明确的只读身份，可安全并行；训练、发布和回滚
单独进入写权限与审计边界。分离不是 PyTorch 的要求，而是 Agent 工程的权限设计。

**“在线训练”是什么意思？** 是在线提交、进度查询和发布后立即切换活动版本，不是
实时遥测或持续学习。

**目前最大的不足？** 数据仍是单工况单故障仿真；只有单机单 Worker；缺少自动发布
门禁、服务端审批令牌、Web 仪表盘、真实 GPU/长期负载验证和生产数据域适配。

## 十二、如何写入简历

推荐把它写成一个统一项目，而不是把 Doro 和 PHM 拆成两个互不相关的小项目。项目名
可用“Doro：面向软件工程与工业健康管理的可扩展智能体”。下面是一版基于当前代码、
不借用旧项目指标的写法：

> **Doro：Coding + 工业装备健康管理智能体｜Python、MCP、PyTorch、SQLite**
> - 在本地 Coding Agent 底座上扩展 PHM 工作流，以 Skill 编排 6 个只读推理工具和
>   10 个管理工具，实现代码任务、数据质检、RUL 预测、模型比较及训练生命周期管理；
> - 基于 NASA C-MAPSS FD001 构建按发动机隔离的无泄漏数据管线，仅使用训练设备拟合
>   特征与标准化统计量，生成 30 周期窗口并训练两层 LSTM、轻量 Transformer；
> - 设计 SQLite 持久化队列与带租约/心跳的单实例 Worker，实现训练自动拉起、epoch
>   进度查询、取消、空闲退出和异常状态可追踪，避免耗时训练阻塞 Agent 对话；
> - 实现候选区、五项产物校验、活动版本注册、原子发布和历史回滚，将只读推理与写
>   管理权限分离；以 32 项 PHM 测试和 234 项主项目回归验证两个独立测试域。

如果需要写模型结果，可补充“LSTM v1 测试 RMSE 15.16、MAE 11.42；Transformer v1
NASA Score 432.39”，但必须说明这是仓库现有 FD001 产物记录，不是每次单元测试重新
训练得到的结果。不要写“准确率 95%”、真实生产部署、维修收益、并发多 Worker 或尚未
测量的效率提升。

## 十三、适合继续新增的项目功能

可按以下优先级增强，项目叙事会比单纯堆更深模型更完整：

1. 训练与模型版本 Web 仪表盘，显示队列、loss 曲线、设备、耗时和活动版本；
2. 发布门禁，按验证指标、数据哈希、回归测试和 smoke 推理自动阻止劣化候选；
3. MLflow 或自建实验追踪，关联参数、数据 manifest、Git SHA、曲线和产物；
4. 静态 CSV/NPZ 批量上传、字段映射、质检和异步预测，不引入实时传感器；
5. FD002～FD004 多工况适配、工况归一化和跨子集泛化报告；
6. split conformal prediction 与覆盖率监控，替代仅展示经验残差区间；
7. checkpoint 续训、任务重试、优先级、超时、磁盘/显存预算和候选清理；
8. 多 Worker 任务级租约、资源调度、孤儿恢复，再演进到分布式队列；
9. 服务端审批令牌、操作者身份、不可变审计事件和模型签名；
10. 传感器消融、Integrated Gradients 等解释功能，并坚持“统计证据不等于物理因果”；
11. 建立 Doro 自己的 Coding + PHM 任务评测集，测工具选择、越权率、结果完整性、
    端到端耗时和故障恢复；
12. 增加 XGBoost/TCN 等基线，比较精度、参数量、CPU 推理延迟和维护成本。

> FD001 是航空发动机退化仿真数据，仅用于算法学习、工程验证和 Agent 工具调用演示，
> 不能直接作为真实航空器维修、适航或放行依据。
