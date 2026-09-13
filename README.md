# Mini Claude Code（Python）

一个从零实现的轻量级 Coding Agent。项目现为纯 Python 版本，不需要 Node.js 或 npm。

## 快速开始

```powershell
# 1. 创建环境并安装（推荐 uv，依赖版本严格等于 uv.lock）
uv sync --extra test

# 2. 配置 API Key
Copy-Item .env.example .env
# 编辑 .env，只填一个服务商的 Key

# 3. 验证：测试应全部通过，不需要 API Key
uv run pytest

# 4. 运行
uv run mini-claude
```

## 环境要求

- Python 3.11+
- [uv](https://docs.astral.sh/uv/)（推荐）或 pip，二选一
- Anthropic API Key，或 OpenAI / OpenAI-compatible API Key

### 一定要用 uv 吗

不是。uv 只是一个包管理器，`pip` 同样能跑全部功能。区别只在于依赖版本能否被精确复现：

| 安装方式 | 需要额外装什么 | 依赖版本 |
| --- | --- | --- |
| `uv sync` | uv（单个二进制） | 严格等于 `uv.lock` 锁定的 32 个包 |
| `pip install -e .` | 无，Python 自带 pip | 落在 `pyproject.toml` 的版本范围内，可能和锁定版本不同 |

`uv.lock` 只对 uv 生效，pip 会完全忽略它。想要和别人装到一模一样的依赖，就用 uv。

## 安装

### 方式一：uv（推荐）

```powershell
uv sync --extra test
```

该命令会创建 `.venv` 并按 `uv.lock` 安装锁定版本。之后的命令统一加 `uv run` 前缀：

```powershell
uv run mini-claude
uv run python -m mini_claude
```

### 方式二：pip

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e ".[test]"
```

macOS / Linux：

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[test]"
```

安装完成后可使用 `mini-claude`，也可直接使用 `python -m mini_claude`。

## 验证安装

```powershell
pytest
```

全部测试应当通过（当前 111 项）。测试**不需要任何 API Key**，因此可以在配置密钥之前先跑一遍，确认代码本身没问题。

## 使用 `.env` 配置

复制示例配置，然后只填写一个服务商的 Key：

```powershell
Copy-Item .env.example .env
```

OpenAI 示例：

```dotenv
OPENAI_API_KEY=your-key
OPENAI_BASE_URL=https://your-provider.example/v1
OPENAI_MODEL=gpt-5.6-sol
OPENAI_REASONING_EFFORT=medium
```

Anthropic 示例：

```dotenv
ANTHROPIC_API_KEY=your-key
ANTHROPIC_BASE_URL=
ANTHROPIC_MODEL=claude-opus-5
ANTHROPIC_EFFORT=medium
```

**模型这一项可以不写。** 每个后端各有内置默认：Anthropic → `claude-opus-5`，OpenAI-compatible → `gpt-5.6-sol`。要换模型用 `ANTHROPIC_MODEL` / `OPENAI_MODEL`，它们只作用于自己的后端。（早先的 `MINI_CLAUDE_MODEL` 对全部后端生效，会把一个服务商的模型名带到另一个上，**已移除**。）

**思考强度也可以不写。** 内置默认是 `medium`，可选 `auto`、`off`、`minimal`、`low`、`medium`、`high`、`xhigh`、`max`。各模型支持范围不同：Anthropic 不支持 `minimal`；已知不支持的组合会在本地报错，其余由实际 API 端点校验，不会静默降级。

`.env` 已被 Git 忽略。

程序按以下顺序查找 `.env`，**首个命中者生效**（先找到的值优先，不会被后续文件覆盖）：

1. `--env-file PATH` 指定的文件
2. 当前工作目录的 `.env`
3. 本包源码树根目录的 `.env`
4. `~/.mini-claude/.env`

因此，即使在项目目录之外运行 `mini-claude` 也能正确读取配置；想给某个项目单独换一套 Key，在该项目目录放一个 `.env` 即可。

终端里已设置的环境变量**始终优先于所有 `.env` 文件**。

## 配置与运行

Anthropic：

```powershell
$env:ANTHROPIC_API_KEY = "your-key"
# 可选：$env:ANTHROPIC_BASE_URL = "https://your-proxy.example"
mini-claude "hello"
```

OpenAI 官方接口：

```powershell
$env:OPENAI_API_KEY = "your-key"
mini-claude --model gpt-5.6-sol "hello"
```

OpenAI-compatible 接口：

```powershell
$env:OPENAI_API_KEY = "your-key"
$env:OPENAI_BASE_URL = "https://your-provider.example/v1"
mini-claude --model your-model "hello"
```

也可以用 `--api-base` 临时覆盖 OpenAI-compatible 地址。

模型解析优先级：`--model` > `ANTHROPIC_MODEL` / `OPENAI_MODEL`（仅该后端）> 该后端的内置默认。启动时首行会打印实际生效的后端与模型，例如：

思考强度解析优先级：REPL `/effort` > 显式 `--effort` / `--thinking` > 恢复会话中保存的值 > `OPENAI_REASONING_EFFORT` / `ANTHROPIC_EFFORT` > `MINI_CLAUDE_EFFORT` > 内置 `medium`。`/effort default` 可重置到环境变量或内置默认。

```text
ℹ Backend: openai | model: gpt-5.6-sol (default for openai) | effort: medium (built-in default)
```

## 常用参数

```text
--yolo              跳过权限确认
--plan              只读计划模式
--accept-edits      自动批准文件编辑
--dont-ask          自动拒绝需要确认的操作
--effort LEVEL      设置思考强度（默认 medium）
--thinking          向后兼容别名，等同于 --effort high
--model, -m NAME    指定模型（不写则用该后端的内置默认）
--env-file PATH     指定 .env 文件，跳过自动查找
--resume            恢复最近会话
--max-cost USD      限制累计费用
--max-turns N       限制 Agent 轮次
```

交互式 REPL 还支持会话和模型管理：

```text
/model                 从当前 API 获取模型列表，并按序号或名称选择
/model <模型名>        在当前后端内切换模型，并保留本次对话历史
/effort                使用方向键选择思考强度
/effort <LEVEL>        立即切换当前会话的思考强度
/effort default        恢复环境变量或内置默认强度
/session               使用方向键选择并恢复会话（等同于 /resume）
/sessions              列出当前项目、当前后端的历史会话
/resume                使用方向键选择并恢复会话
/resume <序号或ID>     直接恢复指定会话
/clear                  清空当前对话
/plan                   切换计划模式
/mcp                    列出每个 MCP server 的连接状态与工具数
/mcp tools              按 server 分组列出已发现的 MCP 工具
/mcp reconnect          立即重连失败/待连接的 MCP server
/cost                   显示本轮与会话累计的 token 与费用明细
/compact                压缩当前上下文
/memory                 列出长期记忆
/skills                 列出技能
/config                 显示最终生效的配置及每一项的来源
/doctor                 逐项体检运行环境与 API 配置
```

`/model` 会调用当前后端的模型列表接口。若 OpenAI-compatible 服务没有实现该接口，仍可使用 `/model <模型名>` 直接切换。切换只改变模型，不会改变 OpenAI/Anthropic 后端。

每轮结束打印的 `Tokens:` 是**总输入**，括号里是缓存明细，两个后端对"总输入"的定义恰好相反，必须分开处理：

- **Anthropic**：`input_tokens` 只是**未命中缓存**的那部分，真正的输入要三项相加（`input_tokens` + `cache_read_input_tokens` + `cache_creation_input_tokens`）。命中缓存时 `input_tokens` 会是 0。
- **OpenAI-compatible**：`prompt_tokens` **本身就是全部输入**，缓存过的前缀是它的子集，单独放在 `prompt_tokens_details.cached_tokens`。这里**不能相加**，否则会重复计数、把上下文占用算高。

`/cost` 里 `cached` 和 `written` **分开显示、绝不合并**：缓存读是折扣（输入价的 0.1x/0.5x），缓存写是**加价**（1.25x），把两者加成一个 `cached` 会把加价当成省下的钱，还会凭空造出一个命中率——某些网关（本机用的 rightapi.ai 就是）把每轮新出现的 token 全记成 cache write 且 `input_tokens` 报 0，合并后就会**每轮都显示"100% cached"**，连刚输入的新内容也算进去。剩下的部分即未缓存输入，可由总数相减得出；`/cost` 在检测到"0 fresh"时会明确说明这只是后端自己的口径、不是可验证的命中率。

这个数同时决定自动压缩何时触发，算错会导致长对话撞上下文上限。费用估算按缓存读的折扣折算：Anthropic 0.1x、OpenAI 0.5x（缓存写 1.25x，仅 Anthropic 计费），并且**按模型取价**。

上下文窗口**所有模型共用一个默认值 272,000**，不按模型名查表。原因：能出现在这个位置的名字（`gpt-5.6-sol`、`claude-opus-5`）多半是网关别名，真实上限无从确证，而按名字精确匹配只会让带日期后缀或改过名的别名静默落到兜底值 —— 与其给一个看似合理的错数，不如给一个明确的默认。窗口同时决定状态栏的分母和自动压缩的触发点（`(窗口 - 20000) × 85%`），填错会实实在在改变压缩时机。要改就在 `.env` 里设 `MINI_CLAUDE_CONTEXT_WINDOW`，纯数字，非法值忽略并回落默认。

每轮结束时，**分隔线下面只打印一行**，内容是本轮消耗：

```
──────────────────────────────────────────────────
  Tokens: 19942 in (99% cached) / 63 out · 2 calls (~$0.0074)
```

- 只报**本轮**（含本轮内压缩调用和子 agent 的开销），会话累计需要时用 `/cost` 看。
- 括号里是**缓存读占本轮的百分比**（`cache_read / 本轮输入`）。缓存写不计入这个比例：它本身是加价项，而网关会把每轮新出现的 token 全记成写，算进去就又变成"100% cached"。
- **"本轮"是同一条输入内所有 API 调用的累加**，不是最后一次调用：一条消息一旦触发工具，至少计费两次（一次决定调哪个工具，一次带工具结果作答），因此本轮的输入 token 经常远大于上下文窗口。末尾的 `· N calls` 就是这个累加次数；只有 1 次时写 `1 call`。
- OpenAI-compatible 的推理模型还会多一个 `N reasoning` 标记（是输出 token 的子集，只用于展示）。
- 未知模型的费用由默认价目估算，这里**不写 `estimated`**（行宽所限）；是否需要打折扣、价格从哪来，看 `/cost` 的 `pricing` 行与 `/doctor`。

定价只内置了公开的一手价目（Anthropic、OpenAI 的若干型号）。网关别名（如 `gpt-5.6-sol`、`claude-opus-5`）**没有公开单价，不会被编造**——查不到时回落到一个带说明的默认价目，并在费用后标注 `estimated`，`/doctor` 也会给一条 warning。要拿到真实数字，在 `.env` 里覆盖：

```bash
# 按模型（JSON，USD / 百万 token）
MINI_CLAUDE_PRICES={"gpt-5.6-sol": {"input": 2, "output": 8, "cache_write": 2.5, "cache_read": 0.4}}
# 或对所有模型统一设定
MINI_CLAUDE_PRICE_INPUT=2
MINI_CLAUDE_PRICE_OUTPUT=8
MINI_CLAUDE_PRICE_CACHE_WRITE=2.5
MINI_CLAUDE_PRICE_CACHE_READ=0.4
```

是否命中缓存由服务端决定，客户端无法保证：同一个请求可能这次命中、下次不命中（经多上游轮询的网关尤其如此），这里只如实显示服务端报回的命中量。没有 `(N cached)` 标记不代表代码有问题，而是这一次没命中。同理，命中量高也不代表真的省钱——它只是后端自报的数字。

标准输出固定按 UTF-8 编码，重定向到文件也是如此，不随系统区域设置（Windows 上默认是 ANSI 代码页，如 cp936）变化。

交互输入支持 `/` 命令自动补全：在行首输入 `/` 会立即显示内置命令和可调用 skill，继续输入可缩小范围，也可按 Tab 补全。主对话提示符支持用上下方向键查看最近输入；历史保存在 `~/.mini-claude/input_history`，重启程序后仍然可用。`/model`、`/session` 和 `/resume` 会直接在输出位置绘制可选列表，用上下方向键移动高亮项、Enter 确认、Esc 取消，不会另外打开侧边候选面板。
交互模式会在终端底部持续显示当前目录、上下文占用/窗口、自动压缩状态、累计估算费用、会话 ID、API 后端、模型和运行模式；配置了 MCP server 时还会显示 `mcp: 已连接/总数`。等待输入时由 Prompt Toolkit 绘制单行底栏；模型流式输出和工具执行时切换为同样的单行 Live 底栏，并主动避开终端最后一列，防止 Windows Terminal 自动换行后留下重绘残影。窄终端会优先保留上下文和模型并自动隐藏次要字段；重定向输出和一次性命令不会绘制状态栏。

恢复会话时会自动恢复该会话保存的模型，并继续使用原会话 ID。当前版本只允许在同一 API 后端内恢复；例如，用 OpenAI 后端启动时不会列出或恢复 Anthropic 会话。会话列表还会按当前工作目录隔离。

## 运行时读取的目录

程序会读取**项目级**和**用户级**两处配置，它们都不属于本包的必需文件。

### 项目级（相对当前工作目录，`CLAUDE.md` 会向上逐级查找）

- `CLAUDE.md`
- `.claude/rules/*.md`
- `.claude/skills/*/SKILL.md`
- `.claude/agents/*.md`
- `.claude/settings.json`
- `.mcp.json`

### 用户级（用户主目录，对该用户的所有项目生效）

- `~/.claude/skills/*/SKILL.md`
- `~/.claude/agents/*.md`
- `~/.claude/settings.json`（MCP server 与权限规则）
- `~/.claude/plans/`
- `~/.mini-claude/.env`（用户级兜底配置）

> **复现提示**：用户级目录会显著影响功能表现。如果本机装过 Claude Code 并配置了 skills 或 MCP，`/skills` 列出的内容与权限行为会和别人**不一致**；反之在干净机器上这些目录为空，相关功能会"看起来不存在"。想确认自己的环境，请对照上面两份清单逐项检查。

### MCP server 配置

在 `.mcp.json`（或任一 settings.json 的 `mcpServers` 段）里登记。源码/安装根目录的 `.mcp.json` 会作为 mini-claude 自身的默认 MCP 配置，因此从其他工作目录启动也能发现其中的服务；当前工作目录配置优先级更高。每个 server 二选一：

**stdio（本地进程）**

```json
{ "mcpServers": { "amap": {
    "command": "npx.cmd", "args": ["-y", "@amap/amap-maps-mcp-server"],
    "env": { "AMAP_MAPS_API_KEY": "..." } } } }
```

**HTTP（远程服务）** —— 默认 Streamable HTTP，遇到 404/405/406/415 会自动降级到旧版 HTTP+SSE

```json
{ "mcpServers": { "remote": {
    "url": "https://host/mcp",
    "headers": { "Authorization": "Bearer ..." },
    "transport": "auto" } } }
```

可选调优：`connectTimeout`（默认 15s，覆盖 connect / initialize / tools-list）、`toolTimeout`（默认 60s，单次 tools/call 的上限，超时不会拖死 agent）、`readOnly`（声明该 server 只读，其工具才允许与其他工具并行执行）、`minInterval` / `maxQps`（限制同一 server 的调用频率，见下）。

> **上游限流**：不少托管服务按 API key 限流（高德是 3 QPS）。并行调用的协议层面没问题，但会被上游拒绝。给这类 server 配上 `"maxQps": 3`（或等价的 `"minInterval": 0.34`），客户端会在**每次调用开始之间**留出间隔 —— 既守住限流窗口，又不影响请求本身的并发重叠。不配则完全不限速。

工具以 `mcp__<server>__<tool>` 暴露给模型。server 名只能含字母数字、下划线和连字符，且不能出现 `__`；拼出的工具名需 ≤ 64 字符，不合规的名字会被跳过并打印原因。

> **Windows 注意**：`.cmd` / `.bat` 包装器必须写全扩展名（`npx.cmd`，不能写 `npx`）—— 创建进程时不走 PATHEXT 补全。`uvx` / `node` / `python` 是真 `.exe`，可省略扩展名。如果 MCP 配置本身用 Node.js 才需要装 Node.js。

> **第 0 次启动**：`npx -y` 要先下载包，可能超过 `connectTimeout`。先手动跑一次预热缓存即可。

## 本地状态（无需复现）

以下内容是按项目和机器隔离的运行时数据，由程序自动生成，别人 clone 后为空属正常现象：

- `~/.mini-claude/sessions/` — 会话历史
- `~/.mini-claude/projects/<hash>/memory/` — 长期记忆
- `~/.mini-claude/input_history` — REPL 输入历史
- `~/.mini-claude/tool-results/` — 工具输出缓存

它们不在仓库中，也不需要提交。

## 开发与测试

```powershell
uv sync --extra test                  # 或用 pip：python -m pip install -e ".[test]"
uv run pytest
uv run python -m mini_claude --help
```

测试不需要真实 API Key。真实 API smoke test 可自行配置 Key 后运行：

```powershell
uv run mini-claude --max-turns 1 "Reply with exactly OK"
```

## 源码结构

```text
mini_claude/
├── __main__.py      CLI 与 REPL
├── agent.py         Agent 循环、双后端与上下文压缩
├── tools.py         文件、搜索、Shell、Web 与权限工具
├── prompt.py        系统提示词与项目规则加载
├── session.py       会话持久化
├── memory.py        记忆系统
├── skills.py        技能系统
├── subagent.py      子 Agent
├── mcp_client.py    MCP 客户端（stdio + HTTP/SSE）
├── frontmatter.py   Frontmatter 解析
└── ui.py            终端 UI
```

## NASA C-MAPSS PHM 扩展

`extensions/phm/` 提供独立的工业健康管理扩展：使用 NASA C-MAPSS
FD001 训练 LSTM 与轻量 Transformer，并通过只读 MCP Server 向主 Agent
暴露数据质检、RUL 预测、模型比较、已保存指标和退化趋势证据。

```powershell
uv sync --project extensions/phm --extra test
uv run --project extensions/phm phm download
uv run --project extensions/phm phm prepare
uv run --project extensions/phm phm train --model lstm
uv run --project extensions/phm phm train --model transformer
uv run --project extensions/phm phm predict --unit-id 42
```

下载器优先使用 NASA 官方数据门户；若连接超时，则自动切换到固定提交的
只读镜像，并对三个 FD001 文件逐一校验 SHA-256。

把 `extensions/phm/mcp.example.json` 中的 `phm` 配置合并到项目 `.mcp.json`
后，可使用 `/phm` 执行有数据证据的 RUL 分析。训练是显式离线步骤；聊天中的
MCP 工具只读且不会重新训练模型。详细说明见 `extensions/phm/README.md`。

运行时会话和记忆保存在用户目录下的 `.mini-claude/` 中。

## License

MIT
