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

42 项测试应当全部通过。测试**不需要任何 API Key**，因此可以在配置密钥之前先跑一遍，确认代码本身没问题。

## 使用 `.env` 配置

复制示例配置，然后只填写一个服务商的 Key：

```powershell
Copy-Item .env.example .env
```

OpenAI 示例：

```dotenv
OPENAI_API_KEY=your-key
OPENAI_BASE_URL=https://your-provider.example/v1
MINI_CLAUDE_MODEL=gpt-4o
```

Anthropic 示例：

```dotenv
ANTHROPIC_API_KEY=your-key
ANTHROPIC_BASE_URL=
MINI_CLAUDE_MODEL=claude-sonnet-4-6
```

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
mini-claude --model gpt-4o "hello"
```

OpenAI-compatible 接口：

```powershell
$env:OPENAI_API_KEY = "your-key"
$env:OPENAI_BASE_URL = "https://your-provider.example/v1"
mini-claude --model your-model "hello"
```

也可以用 `--api-base` 临时覆盖 OpenAI-compatible 地址。模型可通过 `--model` 或 `MINI_CLAUDE_MODEL` 指定。

## 常用参数

```text
--yolo              跳过权限确认
--plan              只读计划模式
--accept-edits      自动批准文件编辑
--dont-ask          自动拒绝需要确认的操作
--thinking          启用 Anthropic 扩展思考
--env-file PATH     指定 .env 文件，跳过自动查找
--resume            恢复最近会话
--max-cost USD      限制累计费用
--max-turns N       限制 Agent 轮次
```

交互式 REPL 还支持会话和模型管理：

```text
/model                 从当前 API 获取模型列表，并按序号或名称选择
/model <模型名>        在当前后端内切换模型，并保留本次对话历史
/session               使用方向键选择并恢复会话（等同于 /resume）
/sessions              列出当前项目、当前后端的历史会话
/resume                使用方向键选择并恢复会话
/resume <序号或ID>     直接恢复指定会话
/clear                  清空当前对话
/plan                   切换计划模式
/cost                   显示本次进程内的 token 与费用统计
/compact                压缩当前上下文
/memory                 列出长期记忆
/skills                 列出技能
```

`/model` 会调用当前后端的模型列表接口。若 OpenAI-compatible 服务没有实现该接口，仍可使用 `/model <模型名>` 直接切换。切换只改变模型，不会改变 OpenAI/Anthropic 后端。

交互输入支持 `/` 命令自动补全：在行首输入 `/` 会立即显示内置命令和可调用 skill，继续输入可缩小范围，也可按 Tab 补全。主对话提示符支持用上下方向键查看最近输入；历史保存在 `~/.mini-claude/input_history`，重启程序后仍然可用。`/model`、`/session` 和 `/resume` 会直接在输出位置绘制可选列表，用上下方向键移动高亮项、Enter 确认、Esc 取消，不会另外打开侧边候选面板。

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

MCP 客户端可以启动任意 stdio MCP server。只有当你的 MCP 配置本身使用 Node.js 时，才需要安装 Node.js。

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
├── mcp_client.py    MCP stdio 客户端
├── frontmatter.py   Frontmatter 解析
└── ui.py            终端 UI
```

运行时会话和记忆保存在用户目录下的 `.mini-claude/` 中。

## License

MIT
