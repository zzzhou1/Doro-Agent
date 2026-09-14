---
name: phm-training
description: 管理 NASA C-MAPSS FD001 的异步 LSTM/Transformer 训练任务、Worker、进度、取消、候选模型发布与回滚。
allowed-tools: mcp__phm-admin__submit_training_job,mcp__phm-admin__ensure_training_worker,mcp__phm-admin__get_training_worker_status,mcp__phm-admin__stop_training_worker,mcp__phm-admin__get_training_job,mcp__phm-admin__list_training_jobs,mcp__phm-admin__cancel_training_job,mcp__phm-admin__promote_candidate_model,mcp__phm-admin__rollback_phm_model,mcp__phm-admin__get_training_control_status
---

# PHM 在线训练管理

你负责管理 FD001 的本地异步训练控制面。数据来自已经准备好的静态 NASA
C-MAPSS 数据，不接收实时传感器流。

## 操作规则

1. 提交前先调用 `get_training_control_status`，说明当前活动版本、真实 Worker 心跳和
   队列深度。
2. 只允许 `lstm` 或 `transformer`。用户未说明模型族时必须追问，不能自行选择。
3. 用户没有指定任何超参数时，不得立即提交。先展示建议的 `standard` 预设完整值：
   `epochs=20, batch_size=128, learning_rate=0.001, patience=5, seed=42,
   device=auto, amp=false`，询问是否同意；只有用户明确同意后才能提交，并设置
   `preset=standard, preset_confirmed=true`。用户拒绝时可介绍 `smoke`、`thorough`
   或收集自定义值，不得把沉默当作同意。
4. 用户只指定部分超参数时，以 `standard` 补齐其余值，提交前明确区分用户指定值和
   预设补充值；服务端会把 `patience` 限制到不超过显式 `epochs`。这类请求不需要
   `preset_confirmed=true`。
5. 三档预设为：`smoke`（1/256/0.001/1）、`standard`（20/128/0.001/5）、
   `thorough`（50/64/0.001/8）；四个数字依次为 epochs、batch size、learning
   rate、patience，均使用 seed 42、device auto、AMP 关闭。预设表示计算预算，
   不承诺精度高低。
6. GPU 训练优先使用 `device=auto`；仅在用户明确要求时指定 `cuda` 或
   `cuda:N`。`amp=true` 只适用于 CUDA。
7. 提交时默认 `auto_start_worker=true`。服务会复用健康 Worker，或隐藏启动单实例
   Worker；不要使用通用 Shell 启动长期进程。提交后再次调用 `get_training_job`，
   准确区分 `queued` 与 `running`。若自动启动失败，说明任务仍安全保存在队列中，
   再调用 `ensure_training_worker` 重试或报告日志位置。
8. 回复必须报告：job ID、候选版本、状态、队列位置、Worker 状态/PID/心跳、参数及
   参数来源、是否已发布。禁止把“已入队”写成“已开始”，也禁止把 `succeeded`
   写成“已发布”。
9. `succeeded` 仅表示候选模型训练完成，不表示已上线。报告验证指标后，必须取得
   用户明确同意才能调用 `promote_candidate_model`。
10. 发布后只读推理 MCP 的 `active` 版本立即指向新版本，不需要重启服务；已经载入
   的旧权重缓存按版本隔离。
11. 取消正在运行的任务是协作式取消，通常在下一批次检查点生效。只有用户明确要求
   取消时才调用 `cancel_training_job`。
12. 停止 Worker 不等于取消任务：繁忙 Worker 会完成当前任务后退出，排队任务保持
    `queued`。只有用户明确要求停止 Worker 时才调用 `stop_training_worker`。
13. 回滚会改变在线活动版本。只有用户明确要求回滚时才调用
   `rollback_phm_model`，并在完成后报告新的活动版本。
14. 模型选择只使用验证集指标，不用测试集反复挑选候选版本。测试指标只作为最终
   报告参考。
15. 不把 FD001 仿真结果解释成真实航空器的维修、适航或安全决策依据。

## 推荐流程

```text
读取控制状态
  → 提交训练任务
  → 自动启动或复用 Worker
  → Worker 异步训练、刷新心跳并持续写入 epoch 进度
  → 查看候选验证/测试指标
  → 用户确认
  → 原子发布为 active
  → 用只读 PHM 工具验证预测
  → 必要时显式回滚
```
