# Doro 与 pi agent 对比及可借鉴设计

> 调研日期：2026-09-18。本文中的 pi agent 指 `badlogic/pi-mono` 入口所指向的 Pi Coding Agent，以及相关的 `pi-ai`、`pi-agent-core`，不是其他同名项目。
>
> Doro 基线：本地工作区，HEAD 为 `55d6ffd477a1c7a4be5e9173507edbd2f33656fe`；调研时工作区有未提交修改，本文描述的是实际读取的工作区。Pi 基线：官方仓库提交 `59eb4c393c79310fcbc585fcce8c38f0961cfa69`，下方外部引用固定到该提交。
>
> 方法：阅读 Doro 实现及相关测试，并核对 Pi 官方文档。未运行两者的同题性能评测，也未验证所有 Pi 扩展示例；不据此声称哪一个成功率更高、成本更低或速度更快。后文“建议”均为待实现设计。

## 1. 核心判断

**Doro 最值得借鉴 Pi 的是平台边界、上下文生命周期和可集成接口，而不是简单增加工具数量。**

Doro 已经具备双后端、MCP、Skill、子 Agent、权限模式、长期记忆、会话恢复、分层压缩和 PHM 训练管理。它的主要短板是这些能力在运行时的耦合：模型协议、会话格式、压缩策略、工具调度、权限判断和终端交互，仍有较多逻辑集中在 `Agent` 中。

Pi 的可借鉴之处是把模型接入、Agent 循环、会话与应用、终端展示分层，并提供扩展接口、SDK 和 RPC。其默认 Coding Agent 刻意不内置子 Agent、计划模式等工作流，允许用户通过扩展组合。Doro 则面向 Coding + 工业 PHM 的统一产品，适合保留开箱即用能力，再逐步改善内部边界。[Pi Coding Agent][pi-coding]、[Pi Agent Core][pi-core]

推荐顺序：

1. **优先处理运行正确性**：权限继承、原始会话保存、结构化压缩、长工具链的上下文预算。
2. **逐步统一内核**：统一消息与事件，抽取 provider adapter，避免两套后端重复维护业务流程。
3. **扩大接入能力**：运行中追加指令、JSONL/RPC、受控的扩展接口。
4. **保留 Doro 的产品价值**：Python 工业计算生态、原生 MCP、PHM Worker 与模型发布回滚。

## 2. 定位与架构差异

| 维度 | Doro 当前实现 | Pi 官方设计 | 对 Doro 的含义 |
|---|---|---|---|
| 产品定位 | 本地 Coding + PHM 智能体 | 可定制的终端 Coding Agent 及相关基础库 | Doro 应继续围绕工业任务形成完整工作流 |
| 技术栈 | Python，Anthropic/OpenAI SDK，Rich、Prompt Toolkit | TypeScript/Node.js，独立模型库、Agent 核心及 TUI 等包 | 借鉴分层即可，不需要换语言或立即拆成多个发行包 |
| 核心组织 | `agent.py` 同时负责双后端循环、压缩、子 Agent、MCP 协调、费用等 | `pi-ai` 提供模型接口，`pi-agent-core` 提供状态、工具执行和事件，Coding Agent 负责应用能力 | 将协议差异下沉，减少业务逻辑重复 |
| 扩展方式 | Markdown Skill、自定义 Agent、MCP、Python 源码扩展 | TypeScript 扩展，可注册工具、命令、事件处理器和 UI；另有 Skill、模板、主题及 packages | Doro 缺少统一的运行时扩展契约，而不是缺少扩展能力 |
| 模型接入 | Anthropic 与 OpenAI-compatible 两条协议路径；同后端切模型 | 多 provider 统一接口、自定义 provider、跨 provider 上下文交接 | 优先统一内部消息，再扩大接入范围 |
| 默认工具策略 | 文件、搜索、Shell、Web 等；部分工具延迟激活 | 默认暴露 `read/write/edit/bash`；另外提供可选内置工具 | 可试验默认工具集合，但不能推导“四工具必然更优” |
| MCP | 原生 stdio、Streamable HTTP、SSE 降级、重连、超时、限流 | 通过扩展集成 | 保留 Doro 原生 MCP，将它纳入统一工具注册和调度接口 |
| 子 Agent / 计划模式 | 内置 explore、plan、general，自定义 Agent，Skill fork | 由扩展提供，非默认内置工作流 | 保留能力，改进隔离、预算和权限继承 |
| 会话 | 原子替换的 JSON 快照，按 cwd/backend 筛选恢复 | JSONL 树形会话，条目含 `id/parentId`，支持树导航与分支 | 区分“完整历史”与“当前模型上下文” |
| 压缩 | 工具结果预算、旧结果裁剪、空闲清理、LLM 摘要 | 结构化摘要、近期消息保留、压缩条目、分支摘要与溢出恢复 | 重点补保真、可回溯和单轮内压缩能力 |
| 运行中输入 | REPL 等待当前 `chat()` 完成；有中断接口 | steering 与 follow-up 两类队列 | 改善长工具链中的纠偏和连续交互 |
| 外部集成 | CLI、REPL；可导入 `Agent.run_once()`，但仍与展示等逻辑耦合 | 交互、print/JSON、RPC、SDK | 先提供稳定机器可读事件，再考虑独立前端 |
| 工业工作流 | PHM 推理、SQLite 任务队列、后台 Worker、候选发布与回滚 | 所读核心文档未提供对应 PHM 业务实现 | 这是 Doro 的垂直领域能力，不宜按通用 CLI 功能表低估 |

Pi 一侧依据：[Coding Agent][pi-coding]、[Agent Core][pi-core]、[模型库][pi-ai]、[压缩设计][pi-compact]、[RPC 协议][pi-rpc]。Doro 一侧的具体代码入口见第 9 节。

需要避免两个误读：Pi “默认四工具”不等于总共只有四个工具；Pi “没有内置某工作流”也不等于无法通过扩展实现。当前文档还描述了项目资源信任机制，不能把 Pi 简化为“完全没有权限或信任控制”。项目加载信任与每次工具执行授权是不同层次。[Pi Coding Agent][pi-coding]

## 3. 最值得借鉴的六项设计

### 3.1 统一消息与模型适配层

**现状。** Doro 同时维护 `_anthropic_messages` 和 `_openai_messages`，分别实现 `_chat_anthropic()`、`_chat_openai()` 以及两套摘要和历史裁剪逻辑。会话文件直接保存 provider 格式，`restore_session()` 明确拒绝跨后端恢复。

这使新增能力容易变成“两边都改一次”。例如注入记忆、权限判断、压缩触发和 usage 累加，需要保证两个路径行为一致。

**Pi 的启发。** Pi Agent Core 区分应用消息 `AgentMessage` 与模型消息，并通过 `transformContext()` 和 `convertToLlm()` 将上下文处理与模型输入转换分开；`pi-ai` 提供统一 provider 接口和跨 provider 上下文交接能力。[Pi Agent Core][pi-core]、[Pi AI][pi-ai]

**建议。** 先在现有 Python 包内形成以下边界，不急于拆仓库：

```text
CLI / REPL / 未来 RPC
          ↓
Agent Runtime：统一消息、事件、轮次、预算、取消
          ├── Context Manager：记忆注入、裁剪、摘要
          ├── Tool Executor：权限、调度、结果归档
          ├── Session Store：历史与检查点
          └── Provider Adapter
                 ├── Anthropic
                 └── OpenAI-compatible
```

统一消息至少保留角色、内容块、工具调用 ID、工具结果、停止原因和来源元数据；推理签名等 provider 专有字段需有独立容纳位置。切换 provider 时由 adapter 决定保留、转换或剔除哪些字段，不能简单改一个模型名后原样重放。

模型能力也应由 adapter/配置描述，包括上下文窗口、输出上限、推理参数和工具支持。Doro 当前对网关别名采取显式窗口配置和保守降级是合理约束，应继续保留；不要用一张过期型号表制造虚假的精确度。

**验收。** 两个 adapter 通过同一套运行时契约测试；同一份工具对话回放保持调用配对、费用口径和取消行为一致。跨后端恢复作为后续能力单独验收，不要求第一步就支持。

### 3.2 完整历史与模型工作上下文分离

**现状。** `save_session()` 用临时文件和 `os.replace()` 原子保存 JSON，这是已有优点。但保存对象是当前消息列表：旧工具输出被替换、LLM 压缩重建历史后，下一次保存也会覆盖原快照。

因此，“能恢复压缩后的对话”与“能追溯压缩前完整过程”并不相同。已有的大结果落盘也不能替代完整会话记录。

**Pi 的启发。** Pi 将完整条目保留在 JSONL 中，压缩记录保存摘要和保留边界，用它重建模型上下文；树形条目可支持 `/tree`、`/fork` 等操作。[Pi Coding Agent][pi-coding]、[压缩设计][pi-compact]

**建议分两步实施。**

1. 先增加追加式会话日志，记录用户输入、最终助手消息、工具调用/结果、模型变化、压缩记录和 usage。现有 JSON 可继续作为快速恢复检查点。
2. 再给条目增加 `id/parent_id` 和活动分支指针，提供分支恢复。第一版不必实现复杂树形 TUI。

需要同时定义半行写入恢复、schema 迁移、重复事件识别、文件大小管理和敏感内容处理。JSONL 本身不保证事务安全。

**PHM 收益。** 可将用户要求、实际工具参数、`job_id`、候选模型版本和发布动作关联起来，便于复盘“为什么训练了这个版本”。仍以 PHM 数据库和模型注册表为业务状态来源，聊天记录不能代替任务状态。

**验收。** 连续压缩并重启后仍可读取原始工具结果；截断最后一条未完成记录不损坏此前历史；读取日志不会自动重放发布、写文件等副作用。

### 3.3 结构化压缩、近期消息保留与单轮内预算

**现状。** Doro 并非没有压缩：已有工具结果预算、陈旧结果裁剪、microcompact、大结果落盘和 LLM 摘要。但 `_compact_*()` 主要要求生成简短段落，然后用摘要重建历史，通常只另外保留最后的用户消息。

自动 LLM 摘要检查位于新用户消息进入时。单条用户请求内部运行很长的工具链时，虽然每轮会做轻量裁剪，仍缺少与 Pi 对应的“工具结果追加后，下一次模型请求前”的完整摘要检查流程。

**Pi 的启发。** Pi 根据 token 预算保留近期消息，以结构化摘要记录目标、约束、进度、关键决策和后续步骤，并追踪文件操作；支持工具批次后的运行内压缩及上下文溢出恢复。[压缩设计][pi-compact]、[Pi Coding Agent][pi-coding]

**建议摘要结构。**

```text
目标与验收条件
用户约束与已确认决定
已完成事项及验证证据
当前工作与阻塞项
关键文件、已修改文件、产物路径
工具或后台任务引用（如 job_id、候选版本）
下一步
```

同时保留最近若干 token 的完整消息，切点必须维持工具调用/结果配对。将“上下文变换”做成纯粹的工作视图构建，不破坏原始会话日志。

触发条件应考虑最新一次 API usage 与其后新增输入的估算量，而不只依赖上次调用的输入 token。还要为回复、待返回工具输出与摘要留出预算。Doro 当前全局默认窗口及 `DORO_CONTEXT_WINDOW` 应逐步升级为可按实际端点/模型覆盖的配置。

**验收。** 单次用户请求连续读取大量文件不会因只能等待下次输入才做摘要而失控；压缩后保留验收条件、尚未完成的步骤和 PHM 任务引用；摘要失败时保留旧工作上下文并明确报告，不生成半截历史。

### 3.4 从终端输出事件升级为运行时事件

**现状。** Doro 已有 `OutputEvent`、`OutputCoordinator` 和统一 renderer，不能说它完全没有事件系统。但这些事件主要解决终端输出序列化，尚不是完整、稳定、可订阅的 Agent 运行时协议；`agent.py` 仍直接依赖 UI 功能。

**Pi 的启发。** Pi Core 暴露 Agent、轮次、消息和工具执行事件，Coding Agent 的 RPC 则通过 stdin/stdout JSONL 交换命令、响应和异步事件。[Pi Agent Core][pi-core]、[RPC 协议][pi-rpc]

**建议。** 基于现有输出体系增量演进，定义带 `session_id/run_id/tool_call_id` 的运行时事件，让终端 renderer 成为订阅者之一。第一步提供 `--output jsonl`；第二步增加 `prompt/abort/get_state` 等 RPC 命令，以及独立的请求 ID。

机器模式 stdout 只输出协议数据，诊断走 stderr；请求“已接受”与任务“已完成”必须分别表达。Python 的可导入入口也应使用同一运行时，避免 RPC 再复制一套流程。

**验收。** 无终端模式能运行、取消并接收工具结果；JSONL 输出可逐行解析且无 ANSI 控制符；重连或订阅多个消费者不会影响模型循环；没有客户端可回答授权时，系统使用明确的拒绝或预配置策略。

### 3.5 运行中的纠偏与后续任务队列

**现状。** `run_repl()` 在读取下一条输入前等待 `chat_with_status()` 完成。当前中断接口能表达停止，但缺少“不结束会话，先补充约束”的输入队列。

**Pi 的启发。** steering 在当前助手轮次的工具执行完成后、下一次模型请求前送达；follow-up 等当前工作结束后再执行。队列与取消分别建模。[Pi Coding Agent][pi-coding]、[RPC 协议][pi-rpc]

**建议。** 增加 `steer()`、`follow_up()` 和 `abort()` 三种独立语义。输入消费与 Agent 执行分离，禁止同一会话同时启动两个修改历史的 `chat()`。

例如，Doro 正在分析日志时，用户输入“只考虑 FD001，不要更换模型”，应在安全边界补入约束；“完成后导出报告”则进入后续队列。steering 不代表已经开始的外部操作可撤销。

必须先处理阻塞工具：当前 `_run_shell()` 使用同步 `subprocess.run()`，由异步工具入口直接调用。只改输入界面并不能保证执行 Shell 时仍可及时响应。应采用异步子进程和明确的进程树终止策略；仅放进线程不能自然解决取消子进程的问题。

**PHM 边界。** 中断聊天不等于取消后台训练；取消训练必须针对已知 `job_id` 调用业务接口，不能将 UI 的 Escape 自动映射为终止所有训练。

**验收。** 新约束只送达一次；工具调用与结果保持配对；取消后排队输入可恢复；长 Shell 期间界面仍能响应。

### 3.6 小而稳定的扩展接口

**现状。** Skill 适合表达提示词和工作流，MCP 适合接入外部工具；两者都不能直接替代运行时事件拦截、命令注册、上下文变换等生命周期扩展。Doro 提示词中提及 hooks，也不等于已经实现可注册和执行的 hooks 系统。

**Pi 的启发。** 官方描述的扩展能注册工具、命令、事件处理器和 UI，并可打包分发 Skill、模板等资源。[Pi Coding Agent][pi-coding]

**建议。** 第一版只提供少量接口，例如 `register_tool()`、`register_command()`、`on_event()`、`transform_context()`。把“只观察事件”和“允许改变执行决策”的接口分开，避免普通日志插件意外改写执行流程。

所有插件工具仍经过统一权限与调度器。扩展声明 API 版本、来源和权限需求；异常默认隔离，但权限检查自身失败应拒绝执行。先支持受信任的本地 Python 扩展，再考虑安装器或市场。

PHM 可作为第一个使用者：注册 `/phm` 命令、领域摘要字段与任务事件展示，而训练和模型发布继续由独立 MCP/Worker 承担。无需把重型训练依赖装进 Doro 主环境。

**验收。** 新增一个领域命令和工具不需要修改 `Agent`；卸载扩展后通用 Coding 仍可用；插件无法绕过父会话的权限、预算和取消约束。

## 4. 扩展之前应先补齐的 Doro 约束

这一节是本地源码审阅发现，不表示 Pi 已经为 Doro 的场景提供了可直接照搬的安全实现。

### 4.1 子 Agent 与 fork Skill 的权限继承

`_execute_agent_tool()` 与 fork Skill 创建子 Agent 时，父级为计划模式就使用 `plan`，其他情况下使用 `bypassPermissions`。这意味着普通模式或 `dontAsk` 并未原样传给子 Agent。`bypassPermissions` 又会在 `check_permission()` 开头直接放行。

建议把子 Agent 的有效能力定义为“父级授权范围与子角色工具白名单的交集”，显式传递授权回调。子 Agent 的成本虽然会回记到父级，但创建时也应明确传递剩余预算与取消信号，避免只在完成后才发现超支。

同时审查权限优先级：当前显式 allow 规则在 plan 限制之前处理。若产品承诺 plan 是严格只读，应决定是否允许这一覆盖，并用测试固定规则。当前权限检查是应用层控制，不等于 OS 沙箱；MCP 的 `readOnly` 也属于配置声明。

### 4.2 会话实例隔离

`_activated_tools`、Skill/自定义 Agent 的缓存，以及全局输出 coordinator，适合单 CLI 使用，但在同进程多会话或 SDK 场景需要重新定义作用域。

建议将激活工具、资源发现结果和订阅者纳入 session/runtime 对象；缓存至少按项目目录与配置来源区分，避免一个会话改变另一个会话的可用工具或输出接收者。

### 4.3 大工具结果的可访问性

已有“大结果落盘 + 预览”，但内置工具还会经过 `_truncate_result()`，`read_file` 也没有 offset/limit 参数。因此不能仅凭有落盘函数就声称所有完整输出始终可回读。

建议在截断前归档完整结果，再返回结构化引用；文件读取支持分页。日志保存与模型上下文裁剪应各自有独立上限。这样长训练日志、仓库搜索结果和测试输出才能真正按需读取。

## 5. 不建议直接照搬的部分

| Pi 的选择或潜在模仿方向 | 不直接照搬的原因 | Doro 更合适的做法 |
|---|---|---|
| 默认尽量精简、许多工作流交给扩展 | Doro 的价值包括现成 MCP 和 PHM 流程 | 核心接口稳定，产品默认能力继续随包提供 |
| 仅以默认四工具作为目标 | 工具更少不自动等于成本更低或可靠性更高 | 用相同模型与任务评估默认工具集、延迟加载和专用搜索工具 |
| 用通用 Shell 替代领域工具 | 会削弱参数校验、版本管理和任务状态跟踪 | 保留 PHM typed MCP 接口与独立 Worker |
| 立即复制完整多包仓库、主题系统、插件市场 | 当前首先需要解决运行时耦合和正确性 | 先在单 Python 包内分层，按真实需求拆包 |
| 立即支持所有 provider 和登录方式 | 维护面扩大，现有网关差异尚需统一 | 先稳定两种 adapter，再按实际使用量扩展 |
| 为“树形会话”自动回放工具 | 分支历史不意味着外部副作用也能回滚 | 会话分支只重建上下文；文件回滚、模型回滚单独执行 |
| 将 PHM 搬到 Agent 同步执行 | 会让交互与训练生命周期相互阻塞 | 继续用 SQLite + Worker，以 job_id 连接对话与任务 |

Pi 的扩展设计提供了组合能力，但并不意味着 Doro 应放弃当前的产品默认值。[Pi Coding Agent][pi-coding]

## 6. 建议实施路线与优先级

优先级表示依赖和价值判断，不是开发工期承诺。

| 阶段 | 工作项 | 首要代码入口 | 验收重点 |
|---|---|---|---|
| P0-A | 子 Agent/fork Skill 权限、预算和取消继承 | `agent.py`、`tools.py`、`subagent.py` | 父级拒绝不能被委派绕过，子任务受到剩余预算约束 |
| P0-B | 原始会话日志与压缩前后状态分离 | `session.py`、`agent.py` | 压缩后历史仍可追溯，写入中断可恢复 |
| P0-C | 结构化摘要、近期消息保留、单轮内预算检查 | `agent.py`，新建 context 模块 | 长工具链与重复压缩不丢关键约束，不拆断工具配对 |
| P1-A | 统一消息、provider adapter、运行时事件 | `agent.py`、`output.py`，新建 providers 模块 | 双后端复用运行时，现有 usage 和推理降级行为保持一致 |
| P1-B | 异步工具执行与 steering/follow-up | `tools.py`、`agent.py`、`__main__.py` | 长任务中可纠偏，取消语义清晰 |
| P1-C | JSONL 输出与最小 RPC | `output.py`、`__main__.py`，新建 rpc 模块 | 外部客户端可启动、观察、取消，stdout 无展示污染 |
| P2-A | 扩展注册接口与 PHM 示例 | `commands.py`、`skills.py`、PHM 扩展 | 加领域能力无需改 Agent 循环 |
| P2-B | 会话分支与模型接入扩充 | `session.py`、providers | 在稳定历史模型和协议层上增量实现 |

其中 P0-C 依赖 P0-B 提供历史保留；P1-B/P1-C 应复用 P1-A 的消息和事件，而不是各自再造状态系统。重构过程中先保留旧 CLI、旧会话读取和现有配置名，再逐步迁移。

如果只选三项，建议选择 **权限与预算继承、原始历史保存、结构化压缩**。它们直接影响现有任务的正确性和可恢复性，比先增加主题或 provider 数量更有价值。

## 7. 如何验证“借鉴后确实更好”

代码结构改善不等于任务效果改善。建议在相同模型、推理强度、端点、仓库初始状态和工具权限下做基线与改造版对照；记录缓存状态，重复运行，不用单次耗时下结论。

| 场景 | 要观察的结果 |
|---|---|
| 跨文件修复并运行测试 | 修复成功率、工具调用数、总输入/输出 token、实际费用或明确标记的估算费用 |
| 长工具链与连续压缩 | 验收条件保留、重复读取次数、上下文溢出次数、tool-call 配对完整性 |
| 重启与异常写入恢复 | 已完成步骤是否可恢复、原始结果是否可读、是否错误重放副作用 |
| 运行中变更约束 | 输入送达延迟、下一步是否遵循新约束、是否重复处理队列消息 |
| PHM 训练到候选评审 | job_id/模型版本可追溯、状态与数据库一致、聊天中断不误取消训练 |
| 子 Agent 与 fork Skill | 父级 deny/dontAsk/plan/预算是否被继承，取消是否传播 |

现有 `test_agent.py`、`test_usage_accounting.py`、`test_permissions.py`、`test_session.py`、`test_output.py` 和 MCP 测试可作为回归基础。应补充上述跨模块契约测试，而不仅测试新类是否能够实例化。

本文仅新增对比文档，没有修改运行时代码，也未重新运行测试套件；README 中既有测试数量不作为本次验证结果。

## 8. 适合 Doro 的目标形态

目标可以概括为：**保留 Python、原生 MCP 和 PHM 工作流，在它们下面形成一个消息统一、历史可追溯、行为可观测、权限可继承的运行时。**

Pi 提供的是这类边界设计的参考。Doro 不必成为 Pi 的 Python 复刻版；它更适合让同一套稳定运行时同时承载 Coding 工具、工业模型调用和异步任务管理。

## 9. 代码与资料索引

### Doro 本地代码

相对路径便于文档随仓库移动；函数名用于定位，不依赖可能随修改变化的行号。

| 文件 | 本文对应入口 |
|---|---|
| [agent.py](../doro/agent.py) | `_chat_anthropic/_chat_openai`、`restore_session`、`_auto_save`、`_compact_*`、`_execute_agent_tool`、`_execute_skill_tool` |
| [session.py](../doro/session.py) | `save_session/load_session/list_sessions` |
| [tools.py](../doro/tools.py) | `check_permission`、`execute_tool`、`_run_shell`、`_truncate_result`、延迟工具激活 |
| [output.py](../doro/output.py) | `OutputEvent`、`OutputCoordinator` |
| [__main__.py](../doro/__main__.py) | `parse_args`、`run_repl`、命令处理与会话恢复 |
| [skills.py](../doro/skills.py) | Skill 发现、inline/fork 与参数展开 |
| [subagent.py](../doro/subagent.py) | 内置角色、自定义 Agent、工具白名单 |
| [prompt.py](../doro/prompt.py) | 系统提示词、`CLAUDE.md` 与 rules 加载 |
| [mcp_client.py](../doro/mcp_client.py) | MCP 连接、工具发现、只读声明与调用管理 |
| [PHM 管理服务](../extensions/phm/src/phm_mcp/admin.py) | 训练参数校验、任务提交、Worker 启动 |
| [PHM 模型注册表](../extensions/phm/src/phm_mcp/registry.py) | 候选产物检查、活动版本与回滚 |
| [PHM 使用说明](../extensions/phm/README.md) | 工业工作流及独立运行环境 |

### Pi 官方资料

- [Coding Agent README][pi-coding]：定位、默认工具、扩展、MCP、会话、消息队列、项目资源信任和运行模式。
- [Agent Core README][pi-core]：统一消息、上下文转换、事件、队列与运行时接口。
- [AI README][pi-ai]：provider 抽象、模型能力与跨 provider 上下文交接。
- [Compaction 文档][pi-compact]：摘要、近期上下文、文件追踪、压缩记录与分支摘要。
- [RPC 文档][pi-rpc]：JSONL 协议、输入队列、请求响应和事件语义。

[pi-coding]: https://github.com/badlogic/pi-mono/blob/59eb4c393c79310fcbc585fcce8c38f0961cfa69/packages/coding-agent/README.md
[pi-core]: https://github.com/badlogic/pi-mono/blob/59eb4c393c79310fcbc585fcce8c38f0961cfa69/packages/agent/README.md
[pi-ai]: https://github.com/badlogic/pi-mono/blob/59eb4c393c79310fcbc585fcce8c38f0961cfa69/packages/ai/README.md
[pi-compact]: https://github.com/badlogic/pi-mono/blob/59eb4c393c79310fcbc585fcce8c38f0961cfa69/packages/coding-agent/docs/compaction.md
[pi-rpc]: https://github.com/badlogic/pi-mono/blob/59eb4c393c79310fcbc585fcce8c38f0961cfa69/packages/coding-agent/docs/rpc.md
