# Mini Claude Code（Python）

一个从零实现的轻量级 Coding Agent。项目现为纯 Python 版本，不需要 Node.js 或 npm。

## 环境要求

- Python 3.11+
- Anthropic API Key，或 OpenAI / OpenAI-compatible API Key

## 安装

推荐使用独立虚拟环境：

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e .
```

macOS / Linux：

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e .
```

安装完成后可使用 `mini-claude`，也可直接使用 `python -m mini_claude`。

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
--resume            恢复最近会话
--max-cost USD      限制累计费用
--max-turns N       限制 Agent 轮次
```

交互式 REPL 支持 `/clear`、`/plan`、`/cost`、`/compact`、`/memory` 和 `/skills`。

## 可选项目配置

程序会从当前工作目录读取以下可选配置；它们不属于本包的必需文件：

- `CLAUDE.md` 和 `.claude/rules/*.md`
- `.claude/skills/*/SKILL.md`
- `.claude/agents/*.md`
- `.claude/settings.json` 和 `.mcp.json`

MCP 客户端可以启动任意 stdio MCP server。只有当你的 MCP 配置本身使用 Node.js 时，才需要安装 Node.js。

## 开发与测试

```powershell
python -m pip install -e ".[test]"
python -m pytest
python -m mini_claude --help
```

测试不需要真实 API Key。真实 API smoke test 可自行配置 Key 后运行：

```powershell
mini-claude --max-turns 1 "Reply with exactly OK"
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
