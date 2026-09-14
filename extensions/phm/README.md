# Doro 的工业健康管理（PHM）能力

这是 Doro 统一智能体中的工业垂直能力模块，不是另一个独立 Agent。它使用 NASA
C-MAPSS FD001 仿真数据训练 LSTM 和轻量 Transformer，预测航空发动机剩余使用
寿命（RUL），并通过 Doro 共用的 Skill、MCP、权限和工具编排系统完成数据质检、
推理、异步训练、候选模型发布与回滚。当前兼容 CLI 仍叫 `mini-claude`，PHM 独立
包仍叫 `mini-claude-phm`。

本 README 重点说明如何在 Agent 对话中按需提交训练和测试。更完整的技术说明见
[FD001 PHM 技术指南](docs/FD001_PHM_GUIDE.md)，统一项目经历见
[Doro Coding + PHM Agent 项目指南](docs/DORO_PROJECT_EXPERIENCE.md)。

## 一、先理解这里的“Agent 在线训练”

本项目的在线训练不是实时传感器流，也不是每收到一条数据就更新一次模型权重。
它指的是：

- 数据仍然使用已经下载并预处理好的静态 FD001；
- 用户可以在 Agent 对话中提交 LSTM/Transformer 训练任务；
- Agent 通过写权限管理 MCP 把任务写入 SQLite 队列；
- 管理服务按需隐藏启动或复用单实例 `phm-worker`；
- Worker 通过 SQLite 租约与心跳报告真实存活状态，空闲 5 分钟后退出；
- Agent 可以随时查询 epoch 进度、指标和错误；
- 训练成功后先形成候选模型，不会自动替换当前模型；
- 用户明确确认后，Agent 才能发布候选模型；
- 发布后的新请求立即使用新的 `active` 版本；
- 如果新模型异常，可以显式回滚上一活动版本。

因此，Agent 是训练任务的控制面，PyTorch Worker 才是实际计算进程。这样不会让一次
MCP 请求等待几十分钟，也能在训练期间继续使用原有模型预测。

```text
用户对话
  ↓ /phm-training
Doro Agent
  ↓ 写管理操作
phm-admin MCP
  ↓ SQLite 持久化队列
Worker Supervisor → phm-worker（租约、心跳、空闲退出）
  ↓ CPU / CUDA / MPS 训练
候选模型 artifacts/candidates/<job_id>
  ↓ 用户确认后原子发布
活动模型 registry.json
  ↓ /phm
phm 只读 MCP → 数据质检、RUL 预测、模型比较
```

## 二、系统组成

| 组件 | 权限 | 作用 |
|---|---:|---|
| `phm` MCP | 只读 | 数据质检、RUL 预测、模型比较、指标和退化趋势 |
| `phm-admin` MCP | 可写 | 提交/查询/取消训练、Worker 管理、发布、回滚 |
| `phm-worker` | 本地进程 | 按需隐藏启动，从 SQLite 队列领取任务并运行训练 |
| SQLite | 本地状态 | 持久化任务、预设、进度、Worker 租约/心跳和错误 |
| 模型注册表 | 本地状态 | 记录 LSTM/Transformer 的活动版本和回滚历史 |
| `/phm` skill | 只读编排 | 约束先质检、再推理、最后解释 |
| `/phm-training` skill | 管理编排 | 约束预设确认、训练、Worker、发布和显式回滚 |

把推理和管理 MCP 分开不是训练算法的必要条件，但能让客户端准确区分只读与写操作。
日常查询可以只启用 `phm`；只有需要改变任务或活动模型状态时才使用 `phm-admin`。

## 三、首次安装与数据准备

所有命令默认从仓库根目录执行：

```text
F:\LLM\mini-cc\claude-code-from-scratch-main
```

安装独立 PHM 环境：

```powershell
uv sync --project extensions/phm --python 3.11 --extra test
```

下载并校验 FD001：

```powershell
uv run --project extensions/phm phm download
```

预处理数据：

```powershell
uv run --project extensions/phm phm prepare
```

确认预处理文件存在：

```powershell
Test-Path extensions/phm/data/processed/fd001.npz
```

应该返回 `True`。如果已经完成过以上步骤，不需要重复执行。

## 四、配置两个 MCP Server

将 [mcp.example.json](mcp.example.json) 中的 `phm` 和 `phm-admin` 两个节点合并到
仓库根目录 `.mcp.json` 的 `mcpServers` 中。不要覆盖其中已有的其他 MCP 或密钥配置。

完整示例：

```json
{
  "mcpServers": {
    "phm": {
      "command": "uv",
      "args": ["run", "--project", "extensions/phm", "phm-mcp"],
      "cwd": ".",
      "readOnly": true,
      "connectTimeout": 30,
      "toolTimeout": 60
    },
    "phm-admin": {
      "command": "uv",
      "args": ["run", "--project", "extensions/phm", "phm-admin-mcp"],
      "cwd": ".",
      "readOnly": false,
      "connectTimeout": 30,
      "toolTimeout": 15
    }
  }
}
```

权限值不要写反：

- `phm` 必须为 `readOnly: true`；
- `phm-admin` 必须为 `readOnly: false`，因为它会写队列、候选目录和模型注册表；
- 训练不在 MCP 调用内部执行，所以管理 MCP 的 `toolTimeout` 不需要设置成训练时长。

`cwd` 是 MCP 子进程的工作目录。相对值按配置所属项目解析，因此这里写 `.` 后，
即使从仓库之外启动 `mini-claude`，`extensions/phm` 仍会按 Doro 仓库根目录定位。
不要把仓库的 `F:\\...` 绝对路径写进可共享配置；只有临时排错时才建议使用绝对路径。

修改 `.mcp.json` 后，如果 Agent 已经运行，执行：

```text
/mcp reconnect
```

也可以重新启动 Agent。

## 五、在 Agent 中在线训练：完整教程

### 5.1 默认不需要手工启动 Worker

通过 Agent 或新版 `phm-admin submit` 提交任务时，管理服务默认会隐藏启动或复用
单实例 Worker。Worker 使用 SQLite 租约和心跳防止重复启动，队列空闲 5 分钟后自动
退出；因此不需要长期保留独立 PowerShell 窗口。

仍可在调试时前台启动，系统发现已有健康 Worker 时会拒绝第二个实例：

```powershell
uv run --project extensions/phm phm-worker
```

- 排队任务保存在 SQLite 中，关闭 Agent 不会丢失任务；
- Worker 是独立后台进程，关闭 Agent 后训练仍会继续；
- 如果强制关闭正在训练的 Worker，当前任务不会断点续训；
- 下次 Worker 启动时会把遗留的 `running/cancelling` 任务标记为 `failed`；
- 尚未领取的 `queued` 任务会保留，Worker 重启后可以继续领取。

### 5.2 启动 Agent

从仓库根目录启动：

```powershell
uv run mini-claude
```

进入交互界面后先检查技能：

```text
/skills
```

应该能看到：

```text
/phm
/phm-training
```

检查 MCP：

```text
/mcp
/mcp tools
```

应该看到 `phm` 和 `phm-admin` 均已连接。`phm-admin` 下应有 10 个工具：

1. `submit_training_job`
2. `ensure_training_worker`
3. `get_training_worker_status`
4. `stop_training_worker`
5. `get_training_job`
6. `list_training_jobs`
7. `cancel_training_job`
8. `promote_candidate_model`
9. `rollback_phm_model`
10. `get_training_control_status`

如果没有连接，先执行：

```text
/doctor
/mcp reconnect
```

### 5.3 第一步：查看训练控制状态

在 Agent 中输入：

```text
/phm-training 查看当前 LSTM 和 Transformer 的活动版本、最近任务以及是否有待处理任务。不要修改任何状态。
```

Agent 会先调用管理 MCP 的控制状态工具。初始状态通常类似：

```text
lstm active: v1
transformer active: v1
worker: stopped
queue depth: 0
```

Worker 状态来自真实租约和心跳，包含 `alive`、PID、当前任务、最近心跳时间、队列
深度和停止请求。超过租约期限没有心跳的进程会显示为 `stale`，不会被误报为在线。

### 5.4 第二步：通过 Agent 提交训练

如果只说明模型而没有给出任何超参数，Agent 不会立即提交，而会先展示完整预设：

```text
/phm-training 训练一个 LSTM 候选模型，版本 lstm-v2。不要发布。
```

Agent 默认建议 `standard`：20 epochs、batch size 128、learning rate 0.001、
patience 5、seed 42、device auto、AMP 关闭。只有你明确回复同意后，Agent 才能以
`preset_confirmed=true` 提交。也可选择：

| 预设 | epochs | batch | lr | patience | 用途 |
|---|---:|---:|---:|---:|---|
| `smoke` | 1 | 256 | 0.001 | 1 | 快速验证完整链路 |
| `standard` | 20 | 128 | 0.001 | 5 | 默认候选实验 |
| `thorough` | 50 | 64 | 0.001 | 8 | 更高训练预算 |

预设只代表计算预算，不承诺精度。也可以直接给出全部或部分参数：

```text
/phm-training 提交一个 Transformer 候选训练任务：版本 transformer-v2，最多 20 个 epoch，batch size 128，学习率 0.001，patience 5，seed 42，device auto。只提交，不要发布。
```

Agent 应返回类似：

```text
job_id: train-20260914050530-d903f45c
status: running
queue_position: null
worker: busy, pid=12345, heartbeat=2s ago
parameter_source: confirmed_preset
published: false
```

提交接口返回后，Agent 会再查询一次状态。回复必须区分“已经写入队列”和“Worker 已
领取”，并列出实际参数及其来源；`succeeded` 仍只是候选完成，不代表已经发布。

如果任务长时间停在 `queued`：

1. 用 `/phm-training` 查询真实 Worker 心跳和错误；
2. 让 Agent 调用 `ensure_training_worker` 幂等重试；
3. 查看 `extensions/phm/runtime/worker.stderr.log`；
4. 确认管理 MCP 和 Worker 使用相同的 `PHM_RUNTIME_DIR`。

### 5.5 第三步：在 Agent 中查看进度

使用刚才返回的 job ID：

```text
/phm-training 查询任务 train-20260914050530-d903f45c 的最新状态和 epoch 进度。不要发布模型。
```

任务进入 `running` 后，`progress` 会包含：

- `epoch`：当前完成轮次；
- `epochs_requested`：请求的最大轮次；
- `train_mse`：本轮训练 MSE；
- `validation_mse`：本轮验证 MSE；
- `best_validation_mse`：当前最佳验证 MSE；
- `stale_epochs`：连续未改善轮数；
- `device`：实际使用的训练设备。

可以隔一段时间再次询问。Agent 本身不会自动持续刷新，除非你再次发起查询。

查看全部最近任务：

```text
/phm-training 列出最近 10 个训练任务，按状态说明哪些正在排队、运行、成功、失败或取消。
```

### 5.6 可选：取消任务

取消会改变任务状态，因此应明确写出 job ID：

```text
/phm-training 取消训练任务 train-20260914050530-d903f45c，并返回取消后的状态。
```

- `queued` 任务会立即变成 `cancelled`；
- `running` 任务先进入 `cancelling`；
- Worker 通常在下一个批次检查点或 epoch 边界停止；
- 已经 `succeeded/failed/cancelled` 的任务不能再次取消。

### 5.7 第四步：检查候选指标，但不要自动发布

任务成功后输入：

```text
/phm-training 读取任务 JOB_ID 的训练配置、验证指标和测试参考指标，判断是否值得进入发布评审，但不要发布。
```

`succeeded` 只代表训练和产物保存成功，不代表模型质量超过 v1。模型选择应该主要依据
验证集指标；测试集只用于最终报告，不能反复用测试集挑版本。

正式发布前至少确认：

- 模型名称与候选版本正确；
- 训练使用的数据和 seed 符合预期；
- 验证 RMSE/MAE/NASA Score 没有明显退化；
- 训练轮数不是仅用于连通性验证的 1 轮 smoke；
- 候选目录包含完整产物；
- 你已经决定接受该版本成为活动模型。

### 5.8 第五步：用户明确确认后发布

发布会改变活动模型，技能不会在没有明确同意时自动执行。确认发布时输入：

```text
/phm-training 我确认发布训练任务 JOB_ID 对应的候选模型。发布后返回新的活动版本和可用版本列表。
```

发布流程会：

1. 校验候选路径位于受管的 `artifacts/candidates` 目录；
2. 校验 `model.pt`、`config.json`、`metrics.json`、`history.json`、
   `residual_quantiles.json` 五项产物；
3. 校验候选配置中的模型名和版本；
4. 拒绝覆盖已有正式版本；
5. 原子复制并更新 `registry.json`；
6. 将原活动版本压入回滚历史。

### 5.9 第六步：用只读 Agent 技能测试活动模型

发布后不需要重启推理 MCP。使用 `/phm`：

```text
/phm 检查 FD001 测试集中 42 号发动机的数据质量，并使用活动 LSTM 模型预测 RUL，报告版本、预测值、经验区间和验证 RMSE。
```

比较两个活动模型：

```text
/phm 比较 42 号发动机的活动 LSTM 和 Transformer RUL 预测，并按验证 RMSE 给出推荐，同时说明这只是 FD001 仿真结果。
```

推理默认使用 `version=active`。如果需要复现旧实验，可以要求显式使用固定版本：

```text
/phm 使用 LSTM v1 预测 FD001 测试发动机 42，不要使用活动版本别名。
```

### 5.10 第七步：必要时显式回滚

如果发布后的模型存在问题：

```text
/phm-training 我确认将 LSTM 回滚到上一个活动版本。完成后报告当前活动版本。
```

回滚只改变活动指针，不删除当前版本或历史模型文件。没有历史版本、或历史产物已经
丢失时，系统会拒绝回滚。

## 六、无需 Agent 的管理命令

Agent 不可用时，可以用同一服务的 CLI 完成管理。默认提交会自动启动后台 Worker；
也可显式管理 Worker：

```powershell
uv run --project extensions/phm phm-admin worker-status
uv run --project extensions/phm phm-admin worker-start
uv run --project extensions/phm phm-admin worker-stop
```

使用完整自定义参数提交：

```powershell
uv run --project extensions/phm phm-admin submit `
  --model lstm `
  --version lstm-v2 `
  --epochs 20 `
  --batch-size 128 `
  --learning-rate 0.001 `
  --patience 5 `
  --seed 42 `
  --device auto
```

只使用预设时，CLI 也要求显式确认：

```powershell
uv run --project extensions/phm phm-admin submit `
  --model lstm `
  --version lstm-v2 `
  --preset standard `
  --confirm-preset
```

若只想入队而不自动启动 Worker，增加 `--no-auto-start-worker`。

查询、列表、取消、发布和回滚：

```powershell
uv run --project extensions/phm phm-admin status JOB_ID
uv run --project extensions/phm phm-admin list --limit 20
uv run --project extensions/phm phm-admin cancel JOB_ID
uv run --project extensions/phm phm-admin promote JOB_ID
uv run --project extensions/phm phm-admin registry
uv run --project extensions/phm phm-admin rollback --model lstm
```

## 七、GPU 训练

设备参数支持：

| 参数 | 说明 |
|---|---|
| `auto` | 优先 CUDA，其次 Apple MPS，最后 CPU |
| `cpu` | 强制 CPU |
| `cuda` | 使用默认 NVIDIA GPU，不可用时直接报错 |
| `cuda:0` | 使用指定 GPU 编号 |
| `mps` | 使用 Apple Silicon MPS，不可用时直接报错 |

Agent 中提交 CUDA 任务：

```text
/phm-training 提交 Transformer 训练：版本 transformer-gpu-v2，30 epochs，batch size 256，patience 5，device cuda，启用 AMP。只提交，不要发布。
```

CLI 等价命令：

```powershell
uv run --project extensions/phm phm-admin submit `
  --model transformer `
  --version transformer-gpu-v2 `
  --epochs 30 `
  --batch-size 256 `
  --patience 5 `
  --device cuda `
  --amp
```

AMP 只支持 CUDA。当前环境是否可用 GPU 必须以 PyTorch 检测结果为准：

```powershell
uv run --project extensions/phm python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.device_count())"
```

如果输出类似：

```text
2.5.1+cpu False 0
```

说明当前安装的是 CPU 版 PyTorch。代码支持 CUDA，但需要另行安装与显卡驱动兼容的
CUDA PyTorch wheel。不要仅凭电脑中存在 NVIDIA 显卡就把 `device` 写成 `cuda`。

## 八、原始离线命令

仍然保留直接训练功能，适合本地调试或基线复现：

```powershell
uv run --project extensions/phm phm train --model lstm --version v1 --device auto
uv run --project extensions/phm phm train --model transformer --version v1 --device auto
uv run --project extensions/phm phm evaluate
uv run --project extensions/phm phm predict --unit-id 42
```

直接 `phm train` 会把结果写入正式模型目录，不经过异步队列和候选发布流程。因此，
面向 Agent 的日常实验更推荐使用 `phm-admin submit + 托管 Worker + promote`。

## 九、目录和运行产物

```text
extensions/phm/
├── mcp.example.json
├── pyproject.toml
├── uv.lock
├── src/phm_mcp/
│   ├── data.py               # 下载、校验、预处理
│   ├── models.py             # LSTM、Transformer
│   ├── training.py           # 训练、设备、AMP、进度、取消
│   ├── inference.py          # 数据质检与 RUL 推理
│   ├── jobs.py               # SQLite 任务队列
│   ├── worker.py             # 异步训练 Worker
│   ├── supervisor.py         # 单实例租约、隐藏启动和停止请求
│   ├── registry.py           # 发布、活动版本、回滚
│   ├── server.py             # 只读推理 MCP
│   ├── admin_server.py       # 写管理 MCP
│   ├── cli.py                # 数据与直接训练 CLI
│   └── admin_cli.py          # 异步训练管理 CLI
├── data/
│   ├── raw/                  # Git 忽略
│   └── processed/            # Git 忽略
├── runtime/
│   └── jobs.sqlite           # Git 忽略
└── artifacts/
    ├── registry.json         # Git 忽略
    ├── candidates/<job_id>/  # Git 忽略
    ├── lstm/<version>/       # Git 忽略
    └── transformer/<version>/# Git 忽略
```

每个完整模型版本包含：

- `model.pt`：权重；
- `config.json`：模型结构、训练参数、设备、seed 和 Git SHA；
- `metrics.json`：验证与测试指标；
- `history.json`：epoch 级训练历史；
- `residual_quantiles.json`：经验预测区间所需的验证残差分位数。

## 十、测试与验证

PHM 扩展测试：

```powershell
uv run --project extensions/phm --extra test pytest -q extensions/phm/tests
```

静态检查：

```powershell
uvx ruff check extensions/phm/src extensions/phm/tests
uvx ruff format --check extensions/phm/src extensions/phm/tests
```

主项目回归：

```powershell
uv run pytest -q
```

当前已验证：

```text
PHM tests: 32 passed
Main project tests: 232 passed
Package build: mini-claude-phm 0.2.0 succeeded
```

## 十一、当前 v1 模型结果

| 模型 | 测试 RMSE | 测试 MAE | 测试 NASA Score |
|---|---:|---:|---:|
| LSTM v1 | 15.16 | 11.42 | 446.10 |
| Transformer v1 | 15.44 | 11.89 | 432.39 |

两种模型没有在所有指标上形成全面优势：LSTM 的测试 RMSE/MAE 略好，Transformer
的 NASA Score 略好。自动比较按验证 RMSE 推荐模型，不能使用测试集反复选择版本。

## 十二、常见问题

### `/skills` 看不到 `/phm-training`

新版 Doro 会从安装根目录发现内置 Skill，不要求当前目录是源码仓库。先确认文件存在：

```powershell
Test-Path F:\LLM\mini-cc\claude-code-from-scratch-main\.claude\skills\phm-training\SKILL.md
```

再退出并重新启动 `mini-claude`，然后执行 `/skills`；技能列表应包含
`/phm-training (bundled)`。Skill 在进程内有缓存，升级前已经运行的旧进程必须重启。
如果仍然看不到，确认当前 `mini-claude` 是从这份源码进行的可编辑安装；普通 wheel
安装需要在构建包中显式包含 `.claude/skills` 资源。

### `/mcp tools` 没有 `phm-admin`

检查根目录 `.mcp.json` 是否合并了 `phm-admin`，然后运行：

```text
/mcp reconnect
/doctor
```

### 任务一直是 `queued`

先让 Agent 查询 `get_training_worker_status`。若 `alive=false` 或状态为 `stale`，调用
`ensure_training_worker`；CLI 等价命令是：

```powershell
uv run --project extensions/phm phm-admin worker-start
```

若启动失败，查看 `extensions/phm/runtime/worker.stderr.log`。任务已经持久化，不需要
重复提交。只有调试 Worker 本身时才建议前台运行 `phm-worker`。

### 为什么只说“训练 LSTM”时没有立即提交

这是预期的安全交互。未给任何超参数时，Agent 必须先列出建议的 `standard` 完整
参数并询问是否同意；你明确同意后才提交。这样不会在未告知训练预算时自动占用
CPU/GPU。若已经给出部分参数，Agent 会明确说明其余字段由哪个预设补齐。

### 任务变成 `failed`

使用 Agent：

```text
/phm-training 读取任务 JOB_ID 的完整错误、训练参数和最后进度，并给出排错建议。不要重新提交或发布。
```

或者使用 CLI：

```powershell
uv run --project extensions/phm phm-admin status JOB_ID
```

常见原因包括预处理文件缺失、强制请求不可用的 CUDA、AMP 与 CPU/MPS 混用、Worker
中断、版本参数非法或磁盘写入失败。

### 为什么训练成功后预测仍然显示 v1

训练成功只产生候选。必须在检查指标并明确确认后执行发布。可以先查询：

```text
/phm-training 查看当前活动版本和可用正式版本，不要改变状态。
```

### 能否同时训练多个任务

可以提交多个任务，它们会排队；当前单 Worker 按创建时间依次执行，不会并行占用多张
显卡。需要真正多 Worker 时，要增加租约/心跳和孤儿任务恢复机制。

## 十三、安全与适用边界

- FD001 是仿真数据，只有单工况和单故障模式；
- 本项目用于算法学习、工程验证和 Agent 工具化演示；
- 经验区间来自验证残差分位数，不是严格概率置信区间；
- 传感器趋势是描述性关联，不代表物理因果；
- 预测不能用于真实航空器维修放行、适航判断或安全决策；
- “在线训练”表示按需任务控制和发布后立即切换，不表示实时遥测或持续学习。

进一步阅读：

- [NASA C-MAPSS FD001 技术实现与复现指南](docs/FD001_PHM_GUIDE.md)
- [Doro Coding + PHM Agent 项目经历与面试指南](docs/DORO_PROJECT_EXPERIENCE.md)
