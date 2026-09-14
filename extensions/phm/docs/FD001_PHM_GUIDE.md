# Doro Coding + PHM Agent：FD001 技术实现与复现指南

本文是 Doro 统一智能体中 PHM 能力的技术手册，重点说明 FD001 数据、时序模型、
MCP 推理、异步训练和模型生命周期。Doro 的统一项目经历与面试叙事见
[DORO_PROJECT_EXPERIENCE.md](DORO_PROJECT_EXPERIENCE.md)。

所有命令默认从仓库根目录执行：

```text
F:\LLM\mini-cc\claude-code-from-scratch-main
```

> 项目边界：C-MAPSS FD001 是 NASA 生成的航空发动机退化仿真数据。本项目用于
> 算法学习、工程验证和 Agent 工具调用演示，不能直接用于真实航空器的维修放行、
> 适航判断或安全决策。

## 目录

1. [项目要解决什么问题](#一项目要解决什么问题)
2. [认识 FD001 数据](#二认识-fd001-数据)
3. [工程目录与产物](#三工程目录与产物)
4. [环境安装](#四环境安装)
5. [下载并校验数据](#五下载并校验数据)
6. [无泄漏预处理](#六无泄漏预处理)
7. [训练 LSTM 与 Transformer](#七训练-lstm-与-transformer)
8. [评估指标与当前结果](#八评估指标与当前结果)
9. [单机预测与结果解释](#九单机预测与结果解释)
10. [接入 MCP 与 Agent](#十接入-mcp-与-agent)
11. [测试、排错与复现检查表](#十一测试排错与复现检查表)
12. [PHM 技术面试指南](#十二面试讲解指南)
13. [Doro 统一项目定位](#十三doro-统一项目定位)
14. [统一项目演示方式](#十四统一项目演示方式)
15. [按需在线训练与测试](#十五按需在线训练与测试)
16. [GPU、CPU 与混合精度](#十六gpucpu-与混合精度)
17. [只读与管理 MCP 分离](#十七为什么分离只读-mcp-与管理-mcp)
18. [异步训练面试说明](#十八异步训练面试说明)
19. [后续新增功能建议](#十九后续新增项目功能建议)

## 一、项目要解决什么问题

工业装备健康管理通常包含三层能力：

1. **数据层**：把多传感器时间序列清洗、对齐并转换成模型可用的样本；
2. **模型层**：根据最近一段运行状态预测剩余使用寿命，即 RUL（Remaining Useful
   Life）；
3. **应用层**：让使用者能可靠地查询设备、比较模型、查看指标和退化证据，而不是
   直接让大模型“猜”一个寿命数字。

本项目用 LSTM 和轻量 Transformer 完成 RUL 回归，再将确定性推理能力封装成只读
MCP 工具。LLM/Agent 负责理解问题、组织调用与解释结果，数值计算始终由本地模型
完成。这种分工与“LLM 解析需求 + 确定性工具执行”的企业 Agent 模式一致，可降低
幻觉和越权风险。

完整链路如下：

```text
NASA FD001 原始文件
        ↓ SHA-256 校验
按发动机 ID 切分训练/验证集
        ↓ 仅用训练发动机拟合标准化器
30 周期滑动窗口
        ↓
LSTM / Transformer 离线训练
        ↓ 保存权重、配置、指标、残差分位数
只读 MCP Server
        ↓
Agent：质检 → 预测/比较 → 证据 → 带边界的解释
```

## 二、认识 FD001 数据

### 2.1 数据集特点

FD001 是 C-MAPSS 的四个子集之一，特点如下：

| 项目 | FD001 |
|---|---:|
| 训练发动机 | 100 台 |
| 测试发动机 | 100 台 |
| 运行工况 | 1 种 |
| 故障模式 | 1 种，高压压气机退化 |
| 每行总列数 | 26 |
| 传感器数量 | 21 |
| 官方任务 | 根据截断测试轨迹预测终点 RUL |

每台发动机对应一条独立的多变量时间序列。训练轨迹一直运行到失效；测试轨迹在
失效前提前截断，真实剩余寿命由单独的 `RUL_FD001.txt` 提供。

### 2.2 三个原始文件

下载后应得到：

```text
extensions/phm/data/raw/
├── train_FD001.txt   # 完整的训练退化轨迹
├── test_FD001.txt    # 在失效前截断的测试轨迹
└── RUL_FD001.txt     # 100 台测试发动机在末周期的真实 RUL
```

原始数据和训练产物体积较大，且可以重复生成，因此都已在 `.gitignore` 中排除。
仓库只提交代码、依赖锁文件、测试和说明文档。

### 2.3 每行 26 列的含义

```text
unit_id, cycle,
setting_1, setting_2, setting_3,
sensor_1, sensor_2, ..., sensor_21
```

- `unit_id`：发动机编号；
- `cycle`：该发动机的运行周期；
- `setting_1~3`：运行设置；
- `sensor_1~21`：传感器观测值。

读取器会检查数据是否恰好为 26 列，异常文件会直接报错，不会静默进入训练。

### 2.4 RUL 标签如何构造

对训练发动机 `i` 在周期 `t` 的样本，先计算：

```text
RUL_raw(i, t) = max_cycle(i) - t
RUL_train(i, t) = min(RUL_raw(i, t), 125)
```

这里采用 125 周期上限的分段线性标签。原因是发动机早期通常处于健康平台期，
此时精确区分“还剩 180 周期”还是“还剩 220 周期”意义有限；封顶可以减少早期
大标签对损失函数的支配，让模型更关注进入退化阶段后的变化。

测试集不自行构造终点标签，而是使用 NASA 给出的 `RUL_FD001.txt`。训练标签与
官方测试真值的来源不同，是面试时需要主动说明的细节。

## 三、工程目录与产物

```text
extensions/phm/
├── pyproject.toml                 # 独立依赖与 CLI 入口
├── uv.lock                       # 可复现依赖锁文件
├── mcp.example.json              # MCP 配置示例
├── src/phm_mcp/
│   ├── config.py                 # 路径、数据源、哈希与默认参数
│   ├── data.py                   # 下载、校验、切分、标准化、滑窗
│   ├── models.py                 # LSTM 与 Transformer
│   ├── training.py               # 训练、早停、指标与模型保存
│   ├── inference.py              # 数据质检、推理、比较与趋势证据
│   ├── jobs.py                   # SQLite 训练任务队列与状态机
│   ├── registry.py               # 活动版本、发布与回滚
│   ├── worker.py                 # 独立异步训练 Worker
│   ├── server.py                 # 只读推理 MCP Server
│   ├── admin_server.py           # 写权限管理 MCP Server
│   ├── admin_cli.py              # 训练、发布和回滚管理命令
│   └── cli.py                    # 数据、离线训练、评估与预测命令
├── tests/                        # 24 项扩展测试
├── runtime/                      # SQLite 队列，Git 忽略
├── data/
│   ├── raw/                      # 原始 FD001，Git 忽略
│   └── processed/                # fd001.npz 与 manifest，Git 忽略
└── artifacts/
    ├── lstm/v1/                  # LSTM 权重与指标，Git 忽略
    └── transformer/v1/           # Transformer 权重与指标，Git 忽略
```

每个模型目录包含：

| 文件 | 用途 |
|---|---|
| `model.pt` | 最佳验证轮次的 PyTorch 权重 |
| `config.json` | 模型结构、训练参数、随机种子、Git 提交 |
| `metrics.json` | 验证集和测试集指标 |
| `history.json` | 每轮训练/验证 MSE |
| `residual_quantiles.json` | 验证残差的 10%/90% 分位数 |

## 四、环境安装

### 4.1 前置条件

- Python 3.11；
- `uv` 包管理器；
- Windows PowerShell、Linux shell 或 macOS Terminal；
- 训练可在 CPU 上完成，不强制要求 GPU。

### 4.2 创建独立环境

从仓库根目录执行：

```powershell
uv sync --project extensions/phm --python 3.11 --extra test
```

该命令会创建 `extensions/phm/.venv`，并根据 `uv.lock` 安装固定版本的 PyTorch、
NumPy 和 pytest。PHM 扩展使用独立环境，不会污染主 Agent 的依赖。

如果默认 PyPI 网络不稳定，可以临时指定镜像：

```powershell
uv sync --project extensions/phm --python 3.11 --extra test `
  --default-index https://pypi.tuna.tsinghua.edu.cn/simple
```

确认 CLI 可用：

```powershell
uv run --project extensions/phm phm --help
```

## 五、下载并校验数据

### 5.1 执行下载

```powershell
uv run --project extensions/phm phm download
```

下载器采用两级策略：

1. 优先访问 NASA Open Data 的 `CMAPSSData.zip`；
2. 官方端超时或归档损坏时，切换到固定提交的只读镜像；
3. 无论来源如何，都对三个 FD001 文件逐一执行 SHA-256 校验；
4. 镜像文件先写入 `.part` 临时文件，校验通过后再原子替换目标文件；
5. 已存在且哈希正确时直接复用，因此命令可以安全重复执行。

固定哈希写在 `src/phm_mcp/config.py`。当前清单为：

| 文件 | SHA-256 |
|---|---|
| `train_FD001.txt` | `963b5e22825b34d8b21c69e1aeb4af3e647050eb672ee8834ba4b5d91d2de0f8` |
| `test_FD001.txt` | `3cda7109ce17bafb5443f2ac926cfcf88154b941b8c4cf95eb55d1ddd6f52851` |
| `RUL_FD001.txt` | `a19c8ec94931949d0485bdc35118206e9c81c4547b422efb9cf86f4ceddbceca` |

如果看到以下警告，表示 NASA 端暂时不可用，程序正在使用经过校验的备用源，并非
训练失败：

```text
RuntimeWarning: NASA archive unavailable (...); using the pinned FD001 mirror.
```

数据来源：[NASA C-MAPSS Open Data](https://data.nasa.gov/dataset/cmapss-jet-engine-simulated-data)。
原始基准论文为 Saxena、Goebel、Simon 和 Eklund 在 PHM 2008 发表的发动机
run-to-failure 仿真研究。

## 六、无泄漏预处理

### 6.1 执行预处理

```powershell
uv run --project extensions/phm phm prepare
```

默认参数：

```text
seed = 42
validation_fraction = 0.2
window_size = 30
RUL cap = 125
```

也可以调整随机种子和验证比例：

```powershell
uv run --project extensions/phm phm prepare --seed 42 --validation-fraction 0.2
```

### 6.2 为什么必须按发动机切分

同一台发动机相邻周期高度相关。如果先生成全部窗口再随机按“样本行”切分，同一台
发动机的相邻窗口可能同时出现在训练集和验证集，造成设备级数据泄漏，验证指标会
虚高。

本项目先按 `unit_id` 将 100 台训练发动机拆成 80 台训练、20 台验证，然后才进行
标准化和滑窗。两组发动机 ID 完全不重叠。

### 6.3 预处理步骤

1. 校验训练、测试文件均为 26 列；
2. 用固定随机种子按发动机 ID 切分训练/验证；
3. 只用 80 台训练发动机拟合均值、标准差和有效特征选择；
4. 删除训练集方差近似为 0 的恒定特征；
5. 用同一组训练统计量变换验证集和官方测试集；
6. 为训练/验证发动机生成长度为 30 的滑动窗口；
7. 对每台测试发动机只取最后 30 个周期，用于预测官方终点 RUL；
8. 保存数组与数据清单。

“只用训练集拟合预处理器”与“按设备切分”是本项目最重要的防泄漏设计。

### 6.4 当前可复现规模

在 `seed=42`、验证比例 20%、窗口长度 30 时：

| 产物 | 数量 |
|---|---:|
| 训练发动机 | 80 |
| 验证发动机 | 20 |
| 官方测试发动机 | 100 |
| 保留的非恒定特征 | 21 |
| 训练窗口 | 14,022 |
| 验证窗口 | 3,709 |

主要输出：

```text
extensions/phm/data/processed/fd001.npz
extensions/phm/data/processed/fd001_manifest.json
```

`fd001_manifest.json` 记录切分 ID、种子、窗口长度、特征名称与原始文件哈希，可用于
审计本次实验是否来自同一份数据。

## 七、训练 LSTM 与 Transformer

### 7.1 推荐复现命令

当前 v1 结果使用以下参数：

```powershell
uv run --project extensions/phm phm train `
  --model lstm --version v1 --epochs 20 --patience 5 --batch-size 128

uv run --project extensions/phm phm train `
  --model transformer --version v1 --epochs 20 --patience 5 --batch-size 128
```

CLI 默认值为 50 轮、batch size 64、学习率 `1e-3`、patience 8。为了快速复现当前
实验，上面的命令采用最多 20 轮和 patience 5。固定随机种子后，CPU 上可得到稳定
结果；不同 PyTorch 版本、硬件和并行后端仍可能带来微小浮点差异。

### 7.2 LSTM 结构

```text
输入：[batch, 30, 21]
  → 2 层 LSTM，hidden size 64，dropout 0.2
  → 取最后一个时间步表示
  → Linear(64, 32) + ReLU + Linear(32, 1)
  → RUL
```

参数量：57,665。LSTM 通过门控递归状态对局部退化过程建模，结构简单、归纳偏置
适合中小规模时序数据，是合理的工业基线。

### 7.3 Transformer 结构

```text
输入：[batch, 30, 21]
  → Linear(21, 64) 特征投影
  → 正弦位置编码
  → 2 层 Transformer Encoder
     4 个注意力头，FFN 维度 128，GELU，dropout 0.1
  → LayerNorm + 最后一个时间步表示
  → Linear(64, 32) + ReLU + Linear(32, 1)
  → RUL
```

参数量：70,593。Transformer 用自注意力直接建模窗口内不同周期之间的关系，便于
扩展到更长窗口和多工况数据，但在 FD001 这种小规模、单工况数据上不保证全面优于
LSTM。

### 7.4 训练策略

- 损失函数：MSE；
- 优化器：Adam；
- 学习率：`1e-3`；
- 梯度裁剪：最大范数 1.0；
- 早停依据：验证集 MSE；
- 最佳权重：保存验证 MSE 最低轮次；
- 随机种子：Python、NumPy、PyTorch 均设为 42；
- 设备：优先 CUDA，否则使用 CPU；
- 推理输出：裁剪到非负数，避免出现负 RUL。

当前 LSTM 在第 9 轮结束，Transformer 在第 10 轮结束，均由早停控制。

## 八、评估指标与当前结果

### 8.1 查看已保存结果

```powershell
uv run --project extensions/phm phm evaluate
uv run --project extensions/phm phm evaluate --model lstm
uv run --project extensions/phm phm evaluate --model transformer
```

### 8.2 指标含义

- **RMSE**：对大误差更敏感，单位为周期，越低越好；
- **MAE**：平均绝对误差，单位为周期，直观且相对稳健，越低越好；
- **NASA Score**：非对称指数惩罚。令 `d = 预测 RUL - 真实 RUL`：
  - `d < 0` 时为 `exp(-d/13) - 1`；
  - `d >= 0` 时为 `exp(d/10) - 1`。

预测过高意味着可能延后维护，NASA Score 对这种误差惩罚更重。该分数是所有样本
惩罚之和，因此不同样本数量之间不能直接横向比较。例如验证集有 3,709 个窗口，
官方测试只有 100 台发动机终点，二者的 NASA Score 绝对值不在同一尺度。

### 8.3 当前 v1 实测结果

| 模型 | 验证 RMSE | 验证 MAE | 测试 RMSE | 测试 MAE | 测试 NASA Score |
|---|---:|---:|---:|---:|---:|
| LSTM v1 | 14.47 | 10.47 | **15.16** | **11.42** | 446.10 |
| Transformer v1 | **13.58** | **9.14** | 15.44 | 11.89 | **432.39** |

如何解读：

- Transformer 在验证 RMSE/MAE 和官方测试 NASA Score 上更好；
- LSTM 在官方测试 RMSE/MAE 上略好；
- 两者不存在“所有指标全面碾压”的关系；
- Agent 的自动推荐规则只看验证 RMSE，因此当前会推荐 Transformer；
- 不能用测试集反复挑模型，否则会把测试集变成隐性验证集。

面试时应如实说明这种差异，它比只报一个最好数字更能体现实验规范。

## 九、单机预测与结果解释

### 9.1 比较两种模型

不指定 `--model` 时，CLI 会同时运行两个模型：

```powershell
uv run --project extensions/phm phm predict --unit-id 42
```

指定单一模型：

```powershell
uv run --project extensions/phm phm predict --unit-id 42 --model lstm
uv run --project extensions/phm phm predict --unit-id 42 --model transformer
```

42 号测试发动机的当前结果：

| 模型 | 最后周期 | 预测 RUL | 80% 经验区间 |
|---|---:|---:|---:|
| LSTM v1 | 156 | 15.528 | [0.000, 32.604] |
| Transformer v1 | 156 | 12.866 | [0.000, 27.397] |

输出还包含：数据集、测试切分、窗口长度、特征数、缺失值、有限值检查、模型版本、
验证指标、测试参考指标和训练时的 Git 提交。

### 9.2 区间是什么，不是什么

区间来自验证残差 `真实值 - 预测值` 的 10% 和 90% 分位数，再平移到当前预测值。
它表示基于历史验证残差的 80% 经验覆盖范围。

它**不是**：

- 严格的贝叶斯置信区间；
- 经过有限样本覆盖证明的 conformal prediction 区间；
- 可直接支撑安全决策的工程裕度。

这个边界应在 Agent 回答和面试中主动说明。

## 十、接入 MCP 与 Agent

### 10.1 配置 MCP Server

将 `extensions/phm/mcp.example.json` 中的 `phm` 节点合并到仓库根目录
`.mcp.json`：

```json
{
  "mcpServers": {
    "phm": {
      "command": "uv",
      "args": ["run", "--project", "extensions/phm", "phm-mcp"],
      "readOnly": true,
      "connectTimeout": 30,
      "toolTimeout": 60
    }
  }
}
```

`.mcp.json` 可能还包含其他服务，不要覆盖整个文件，只合并 `phm` 节点。

启动主 Agent 后检查：

```text
/doctor
/mcp tools
```

### 10.2 六个只读工具

| MCP 工具 | 作用 |
|---|---|
| `list_phm_models` | 列出本地模型版本和已保存指标 |
| `inspect_engine` | 检查发动机是否存在、窗口是否完整、数据是否有限 |
| `predict_rul` | 用指定 LSTM/Transformer 预测 RUL |
| `compare_rul_models` | 同时推理并按验证 RMSE 推荐模型 |
| `get_model_metrics` | 只读取训练时保存的指标，不重新训练 |
| `get_degradation_evidence` | 返回最后窗口中变化最强的标准化特征趋势 |

只读 `phm` MCP 在聊天阶段不会下载数据、训练模型或修改设备状态；启用写权限的
`phm-admin` 后可在对话中提交管理任务，耗时训练仍由独立 Worker 异步执行。

### 10.3 `/phm` 技能的安全调用顺序

技能文件位于 `.claude/skills/phm/SKILL.md`，规定发动机查询必须遵循：

```text
inspect_engine
    ↓ 数据通过
predict_rul 或 compare_rul_models
    ↓ 用户需要解释时
get_degradation_evidence
    ↓
报告模型版本、区间、验证指标与仿真数据边界
```

先质检再推理，可以防止未知发动机、缺失数据或非有限值被直接送入模型。

### 10.4 可以直接尝试的问题

```text
/phm 检查 FD001 测试集中 42 号发动机的数据质量。
/phm 比较 42 号发动机的 LSTM 与 Transformer 剩余寿命预测。
/phm 解释 42 号发动机最后 30 个周期中变化最明显的 5 个特征。
/phm 查看 Transformer v1 的验证与测试指标，并说明指标边界。
```

退化趋势是标准化窗口中的描述性关联，不代表某个传感器物理上“导致”故障。

## 十一、测试、排错与复现检查表

### 11.1 运行扩展测试

```powershell
uv run --project extensions/phm pytest -q extensions/phm/tests
```

当前结果：`24 passed`。测试覆盖标签、设备级切分、标准化、滑窗、哈希校验与备用
下载、模型输出形状、指标公式、MCP 初始化、工具列表和错误返回。

注意：从仓库根目录执行时要显式传入 `extensions/phm/tests`。否则 pytest 可能读取
主项目配置，将主项目测试放到 PHM 的独立环境中运行，造成“缺少主项目依赖”的假
失败。

主项目回归：

```powershell
uv run pytest -q
```

当前主项目结果为 `229 passed`。

代码质量检查：

```powershell
uv run ruff check extensions/phm/src extensions/phm/tests
uv run ruff format --check extensions/phm/src extensions/phm/tests
```

### 11.2 常见问题

#### NASA 下载握手超时

现象：出现 `handshake operation timed out`。处理：无需手工下载，等待程序自动切换
到固定镜像；最终以 SHA-256 是否通过为准。

#### 提示找不到 `fd001.npz`

说明尚未预处理，执行：

```powershell
uv run --project extensions/phm phm prepare
```

#### 提示找不到模型产物

说明尚未训练对应版本，分别执行两条 `phm train` 命令。

#### Transformer 出现 nested tensor 警告

PyTorch 可能提示 `norm_first=True` 导致 nested tensor 优化未启用。这是性能提示，
不影响模型结构、权重加载或预测正确性。

#### 修改窗口长度后模型无法加载

预处理数组和模型结构必须使用一致的窗口长度。修改窗口后应重新 `prepare`，再用新
版本号训练两个模型，避免覆盖原 v1。

### 11.3 从零复现检查表

- [ ] 使用 Python 3.11 安装独立 PHM 环境；
- [ ] `phm download` 得到三个哈希正确的 FD001 文件；
- [ ] `phm prepare` 得到 80/20 台无重叠发动机切分；
- [ ] 检查 `fd001_manifest.json` 的种子、ID 与哈希；
- [ ] 训练 LSTM 与 Transformer，并使用不同版本目录保存实验；
- [ ] 用 `phm evaluate` 查看保存指标；
- [ ] 用 42 号发动机完成 CLI 双模型预测；
- [ ] 用 `/mcp tools` 确认六个 PHM 工具；
- [ ] 运行 24 项扩展测试与 229 项主项目回归；
- [ ] 在结论中注明 FD001 仿真、单工况、单故障模式的限制。

## 十二、面试讲解指南

### 12.1 30 秒项目介绍

> 我在自研本地 Agent 中加入了一套工业装备健康管理能力。项目使用 NASA C-MAPSS
> FD001 数据，先按发动机 ID 做无泄漏切分和训练集独立标准化，再用 30 周期窗口训练
> 两层 LSTM 与轻量 Transformer 预测剩余寿命。官方 100 台测试发动机上，LSTM 的
> RMSE 为 15.16，Transformer 的 NASA Score 为 432.39。最后我把质检、预测、模型
> 比较、指标读取和退化趋势封装为 6 个只读 MCP 工具，由 Agent 负责调用编排与带边界
> 的解释，而不是让 LLM 直接生成数值。

### 12.2 两分钟 STAR 版本

**Situation：** 工业时序模型通常停留在 Notebook，难以被 Agent 安全、可复现地
调用；而让 LLM 直接判断设备寿命又存在数值幻觉和权限不清的问题。

**Task：** 构建一个从可信数据、无泄漏训练到 Agent 工具调用的最小闭环，同时保持
模型足够简单，便于解释与复现。

**Action：**

1. 为 NASA FD001 实现官方源优先、固定镜像兜底和 SHA-256 校验；
2. 按发动机 ID 做 80/20 切分，仅用训练发动机拟合标准化器，构造 30 周期窗口；
3. 用相同数据、种子与早停规则训练 LSTM 和 Transformer；
4. 保存权重、参数、指标、Git 提交和残差分位数；
5. 将能力封装成 6 个只读 MCP 工具，并用技能规则约束“先质检、后推理、再解释”；
6. 用 24 项扩展测试和 229 项主项目回归验证集成没有破坏原系统。

**Result：** LSTM 在官方测试集达到 RMSE 15.16、MAE 11.42；Transformer 达到
NASA Score 432.39。主 Agent 能真实连接 MCP 服务，完成单设备检查、双模型比较和
退化证据查询，形成了可复现、可追踪、有安全边界的 PHM Agent 原型。

### 12.3 高频问题与回答要点

#### Q1：为什么先选 FD001？

FD001 只有一种工况和一种故障模式，规模适中，适合先验证数据、模型、MCP 与 Agent
的完整工程链路。它不是最终工业泛化结论；后续应扩展 FD002/FD004 的多工况数据和
真实设备域适配。

#### Q2：你怎么避免数据泄漏？

先按发动机 ID 切分，再分别构造窗口；同一台发动机不会跨训练/验证集。特征选择、
均值和标准差只在训练发动机上拟合，验证和测试只应用这些统计量。不能先对全量数据
标准化，也不能按窗口随机切分。

#### Q3：为什么窗口长度是 30？

30 是精度、实时性和计算量之间的基线选择：能观察近期退化趋势，又不会让轻量模型
过大。当前实现没有声称 30 是全局最优；严谨做法是在发动机级验证集上比较 20、30、
50 等窗口，并保持官方测试集只做最终评估。

#### Q4：为什么给训练 RUL 封顶 125？

早期健康阶段的精确寿命差异难以由传感器辨别，直接使用很大的线性标签会让 MSE 被
早期样本主导。125 周期平台是 C-MAPSS 常见处理，使模型更关注明显退化阶段。它是
建模假设，需要通过验证集或业务成本进一步选择。

#### Q5：LSTM 和 Transformer 谁更好？

不能简单回答谁全面更好。当前 LSTM 的测试 RMSE/MAE 更低，Transformer 的验证
RMSE/MAE 和测试 NASA Score 更低。自动推荐只依据验证 RMSE，避免用测试集挑模型，
所以当前推荐 Transformer。部署选择还应结合误差成本、延迟、稳定性和硬件约束。

#### Q6：为什么同时使用 RMSE、MAE 和 NASA Score？

MAE 直观，RMSE 强调大误差，NASA Score 体现预测过高比预测过低更危险的非对称
成本。工业健康管理不能只优化平均误差，还要关心错误方向和尾部风险。

#### Q7：为什么验证 NASA Score 比测试分数大很多？

因为该分数是逐样本惩罚之和，不是均值。验证指标覆盖 3,709 个滑动窗口，测试指标
只有 100 个发动机终点，所以绝对值不能直接比较。这不是模型在验证集上异常崩溃。

#### Q8：经验区间可靠吗？

它来自验证残差的 10%/90% 分位数，是简单的经验覆盖区间，可以提示不确定性，但
不是严格校准的概率区间。进一步可采用 split conformal、分工况校准或深度集成，并
监控实际覆盖率。

#### Q9：为什么需要 Agent，普通预测 API 不够吗？

单一 API 只能完成数值推理。Agent 可以根据自然语言先选择设备和模型，执行数据质检、
比较结果、补充指标与趋势证据，并统一披露安全边界。关键是 Agent 只做编排和解释，
不能替代确定性模型计算。

#### Q10：为什么 MCP 工具全部只读？

聊天中的意图可能不完整。将下载、训练和状态修改排除在在线工具之外，可以限制副作用，
降低误触发成本，并使审计更简单。新模型必须通过显式离线命令产生。

#### Q11：趋势证据是不是故障根因？

不是。当前证据只是最后 30 个周期内标准化变化幅度和斜率较大的特征，用于辅助解释
模型输入发生了什么变化。它没有建立物理因果关系，根因诊断需要机理知识、部件映射、
消融实验和真实故障标签。

#### Q12：下一步如何扩展到真实工业项目？

优先级可以是：多工况归一化与 FD002/FD004；严格的超参数验证；conformal 区间；
数据漂移和 OOD 检测；模型注册与版本回滚；在线时序数据接入；维护阈值与成本函数；
最后才是更复杂的网络。真实航空场景还需要专家审核、配置管理、验证确认和适航流程。

### 12.4 五分钟现场演示顺序

1. `phm evaluate`：展示两个模型与可追踪指标；
2. `/mcp tools`：展示六个受控工具和只读边界；
3. `/phm 检查 42 号发动机`：说明先质检；
4. `/phm 比较 42 号发动机`：展示双模型预测、经验区间与推荐规则；
5. `/phm 解释退化证据`：说明描述性证据与因果边界；
6. 打开 `fd001_manifest.json` 和模型 `config.json`：证明数据哈希、随机种子和 Git
   提交可追踪。

演示时不要只展示最终数字，要突出为什么这条链路可信、可复现、可审计。

### 12.5 面试中避免的表述

- 不说“模型准确率 95%”：RUL 是回归任务，当前指标不是分类准确率；
- 不说“Transformer 全面优于 LSTM”：真实结果并不支持；
- 不说“置信区间保证 80% 安全覆盖”：当前只是验证残差经验区间；
- 不说“发现了传感器故障根因”：当前只有描述性趋势；
- 不说“可直接用于真实发动机”：FD001 是仿真、单工况、单故障模式；
- 不把 229 项主项目回归说成 229 项均由本扩展新增：本扩展新增的是 24 项测试。

## 十三、Doro 统一项目定位

PHM 不应作为与 Coding Agent 平行的第二个项目。当前仓库的真实结构是：Doro 提供
统一 Agent 循环、工具、权限、Skill、MCP、会话、记忆、上下文压缩和子 Agent；
Coding 与 PHM 是同一底座上的两类工作流。

```text
Doro Agent Core
├── Coding：文件、搜索、Shell、Web、Git、测试与代码修改
└── PHM：数据质检、RUL 推理、异步训练、候选发布与回滚
```

LLM 负责理解意图和编排；Coding 工具负责真实修改代码；LSTM/Transformer 负责
真实数值计算；SQLite 与 Worker 负责长耗时训练。这个责任分离是统一项目的核心，
而不是把一个模型 Demo 简单放进 Agent 目录。

完整的项目名称、简历 bullet、不同岗位版本、面试开场和真实性检查表见：

- [Doro：可 Coding、可 PHM 的统一智能体项目经历与面试指南](DORO_PROJECT_EXPERIENCE.md)

本技术文档后续只保留可以从当前 PHM 代码和产物验证的实现细节，不再假设旧 Doro
项目与当前仓库使用相同底层，也不要求沿用旧简历中的业务指标或测试数字。

## 十四、统一项目演示方式

面试或作品集演示应使用同一个 Doro 进程连续展示两类能力：

```text
Coding 链路
读取仓库 → 搜索调用链 → 修改文件 → 运行测试 → 检查 Git diff

PHM 推理链路
发现 Skill/MCP → 检查发动机 → 调用活动模型 → 比较结果 → 说明边界

PHM 训练链路
提交候选 → Worker 训练 → 查询 epoch → 检查指标 → 确认发布 → 回滚
```

重点是说明三个问题：同一 Agent 如何选择不同领域工具；只读与写操作怎样隔离；
确定性执行结果如何回到对话中。没有必要在现场等待完整模型训练，可以预先准备一个
已完成候选，同时现场提交一个短任务展示队列和进度。

## 十五、按需在线训练与测试

### 15.1 这里的“在线”具体指什么

本项目不接收实时传感器，也不做每来一条数据就更新一次权重的持续学习。当前实现的
在线能力是：使用已经准备好的 FD001 静态数据，在 Agent 对话或命令行中随时提交
训练任务；训练在独立 Worker 中异步执行；训练结束后先形成候选版本；人工确认后再
原子发布；只读推理服务随后的 `active` 查询会自动使用新版本。

```text
Agent / phm-admin CLI
       ↓ 提交、查询、取消
phm-admin（写 MCP）
       ↓ SQLite 持久化队列
phm-worker（独立进程，单 Worker）
       ↓ CPU / CUDA / MPS 训练
artifacts/candidates/<job_id>/
       ↓ 人工确认发布
artifacts/<model>/<version>/ + registry.json
       ↓ active 版本解析
phm（只读 MCP）→ 在线预测与模型比较
```

这种设计的价值是训练耗时不会占住 MCP 请求，Agent 或终端退出后任务记录仍在
SQLite 中；推理与训练进程也互不阻塞。这里的“实时”是任务状态和发布后的模型切换
可以立即查询，不代表毫秒级流式训练。

### 15.2 启动在线训练系统

先完成环境、下载和预处理：

```powershell
uv sync --project extensions/phm --python 3.11 --extra test
uv run --project extensions/phm phm download
uv run --project extensions/phm phm prepare
```

打开一个单独终端并保持 Worker 运行：

```powershell
uv run --project extensions/phm phm-worker
```

当前实现按“单机、单 Worker”设计。SQLite 的领取事务可以避免同一个任务被重复
领取，但不要同时启动多个长期 Worker；Worker 重启时会把上一次遗留的
`running/cancelling` 任务标记为失败，避免任务永久卡住。

另一个终端提交任务：

```powershell
uv run --project extensions/phm phm-admin submit `
  --model lstm `
  --version lstm-v2 `
  --epochs 20 `
  --batch-size 128 `
  --patience 5 `
  --device auto
```

命令会立即返回 JSON，其中 `id` 是后续操作必须保存的 job ID。状态生命周期为：

```text
queued → running → succeeded
   │        │
   └→ cancelled
            └→ cancelling → cancelled
running → failed
```

查询任务和最近列表：

```powershell
uv run --project extensions/phm phm-admin status JOB_ID
uv run --project extensions/phm phm-admin list --limit 20
```

`progress` 会记录当前 epoch、目标 epoch、训练 MSE、验证 MSE、最佳验证 MSE、
早停累计轮数和实际设备。`succeeded` 后还会返回候选路径与验证/测试指标。

取消排队或运行中的任务：

```powershell
uv run --project extensions/phm phm-admin cancel JOB_ID
```

运行中取消采用协作式检查：Worker 通常在下一个批次检查点或 epoch 边界停止，因此
不会承诺进程被瞬间强杀。这样可以避免在写权重或 JSON 时留下半个正式版本。

### 15.3 候选、发布与回滚

训练成功不会自动上线。候选文件先保存在：

```text
extensions/phm/artifacts/candidates/<job_id>/
```

确认验证指标和训练参数后显式发布：

```powershell
uv run --project extensions/phm phm-admin promote JOB_ID
uv run --project extensions/phm phm-admin registry
```

发布过程先校验五个必需产物、模型名和版本号，再复制到临时目录并原子改名，最后
原子更新 `registry.json`。正式目录已有同名版本时会拒绝覆盖。推理接口默认使用
`version=active`，所以发布后新请求会解析到新版本；也可以显式传 `v1` 等固定版本
复现实验。

如果新版本表现异常，可回滚到上一个活动版本：

```powershell
uv run --project extensions/phm phm-admin rollback --model lstm
```

回滚只改变活动指针，不删除任何模型文件。回滚前一个版本若丢失，系统会拒绝操作。

## 十六、GPU、CPU 与混合精度

训练不再限定 CPU。设备参数支持：

| 参数 | 行为 |
|---|---|
| `auto` | 优先 CUDA，其次 Apple MPS，最后 CPU |
| `cpu` | 强制 CPU |
| `cuda` | 使用默认 NVIDIA GPU，无可用 CUDA 时直接报错 |
| `cuda:0` | 指定 GPU 编号，并校验编号是否存在 |
| `mps` | 使用 Apple Silicon MPS，无可用 MPS 时直接报错 |

CUDA 示例：

```powershell
uv run --project extensions/phm phm-admin submit `
  --model transformer --version transformer-gpu-v2 `
  --epochs 30 --batch-size 256 --device cuda --amp
```

`--amp` 启用 CUDA 自动混合精度，CPU/MPS 不接受该选项。当前锁定环境若安装的是
`torch 2.5.1+cpu`，代码虽然支持 CUDA，运行时仍看不到显卡；需要按照 PyTorch
针对本机 CUDA 驱动的安装方式换成对应 CUDA wheel，然后用下面命令确认：

```powershell
uv run --project extensions/phm python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.device_count())"
```

不要仅凭电脑有 NVIDIA 显卡就判断可用，必须以 `torch.cuda.is_available()` 为准。
由于 CUDA wheel 与操作系统、驱动和 CUDA 版本绑定，本仓库不把某个 CUDA 索引
硬编码进跨机器的 `uv.lock`。

## 十七、为什么分离只读 MCP 与管理 MCP

`mcp.example.json` 注册两个服务：

| Server | `readOnly` | 职责 |
|---|---:|---|
| `phm` | `true` | 质检、预测、比较、指标和趋势证据 |
| `phm-admin` | `false` | 提交/取消训练、发布、回滚、控制状态 |

分离不是模型训练的技术前提，但对 Agent 工程很有价值：

1. 主客户端只有在整个 Server 声明只读时，才可以安全地把其工具用于计划模式或与
   其他只读调用并行；
2. 如果把发布、回滚和预测放在同一个写 Server 中，所有预测都会失去清晰的只读
   身份，权限粒度过粗；
3. 日常分析可以只启用 `phm`，完全不暴露修改模型状态的工具；
4. 管理操作有独立审计入口，便于后续增加确认、用户角色和审批流；
5. 即使管理 MCP 暂时不可用，现有活动版本仍能继续推理。

将 `mcp.example.json` 的两个节点合并到根目录 `.mcp.json` 后，用 `/mcp tools`
检查。不要把 `phm-admin` 错标为只读；它会写 SQLite、候选目录和模型注册表。

管理 MCP 共提供 7 个工具：提交、查询单任务、列出任务、取消、发布、回滚和读取
控制状态。项目技能 `.claude/skills/phm-training/SKILL.md` 进一步规定：发布与回滚
必须在用户明确确认后执行，测试集指标不能用于反复选择候选模型。

## 十八、异步训练面试说明

可以这样解释 Doro 中的在线训练：

> 我没有把耗时 PyTorch 训练直接放进 Agent 的 MCP 请求，而是把 Doro 设计成控制面：
> 写管理 MCP 校验参数并将任务写入 SQLite，独立 Worker 在 CPU/CUDA/MPS 上训练并
> 回写 epoch 进度，产物先进入候选区。用户检查验证结果并明确确认后，系统才原子更新
> 活动模型；推理 MCP 始终只读，训练期间旧版本仍能服务，异常时可以回滚。这使同一个
> Doro 既能执行 Coding 任务，也能安全管理工业模型的训练与推理。

这里的“在线”表示按需提交、状态可查询和发布后立即生效，不表示实时传感器接入或
样本级持续学习。项目经历不要被旧 Doro 指标束缚，也不要把两个底层不同项目的数字
拼在一起；如需量化，应重新为当前 Doro 建立 Coding 与 PHM 任务评测。

统一项目经历与不同岗位版本见：

- [DORO_PROJECT_EXPERIENCE.md](DORO_PROJECT_EXPERIENCE.md)
## 十九、后续新增项目功能建议

可以按“先工程闭环，再算法增强”的顺序扩展：

1. **训练仪表盘**：展示任务队列、实时损失曲线、设备、耗时、活动版本和一键回滚；
2. **静态文件批量测试**：上传 CSV/NPZ 后异步质检和批量预测，无需实时传感器；
3. **模型发布门禁**：验证 RMSE、NASA Score、回归测试和数据哈希全部通过才能发布；
4. **实验追踪**：接入 MLflow，记录参数、Git SHA、数据清单、曲线和模型产物；
5. **FD002～FD004**：增加多工况归一化、故障模式分层指标和跨子集泛化比较；
6. **可信区间**：从残差分位数升级到 split conformal prediction，并监控覆盖率；
7. **退化可解释性**：增加传感器消融、Integrated Gradients 或注意力可视化，但保持
   “统计证据不等于物理因果”的边界；
8. **资源治理**：增加任务优先级、显存预算、超时、磁盘配额和候选清理策略；
9. **可观测性**：结构化日志、任务耗时分位数、错误分类、Worker 心跳和告警；
10. **多 Worker 演进**：引入任务租约、心跳与重试，随后再替换为分布式队列；
11. **权限与审计**：记录操作者、发布审批、不可变事件日志和模型签名；
12. **基线扩充**：加入 XGBoost/TCN 等简单可解释基线，比较精度、参数量和推理
    时延。

建议下一阶段优先做“Web 仪表盘 + 发布门禁 + MLflow”。这三项最能把当前原型提升为
面试中可演示的模型工程系统；FD002～FD004 和 conformal prediction 更偏算法深度。
