---
name: phm-training
description: 管理 NASA C-MAPSS FD001 的异步 LSTM/Transformer 训练任务、候选模型发布与回滚。
allowed-tools:
  - mcp__phm-admin__submit_training_job
  - mcp__phm-admin__get_training_job
  - mcp__phm-admin__list_training_jobs
  - mcp__phm-admin__cancel_training_job
  - mcp__phm-admin__promote_candidate_model
  - mcp__phm-admin__rollback_phm_model
  - mcp__phm-admin__get_training_control_status
---

# PHM 在线训练管理

你负责管理 FD001 的本地异步训练控制面。数据来自已经准备好的静态 NASA
C-MAPSS 数据，不接收实时传感器流。

## 操作规则

1. 提交前先调用 `get_training_control_status`，说明当前活动版本和 Worker 状态。
2. 只允许 `lstm` 或 `transformer`。未指定参数时使用服务端默认值。
3. GPU 训练优先使用 `device=auto`；仅在用户明确要求时指定 `cuda` 或
   `cuda:N`。`amp=true` 只适用于 CUDA。
4. 提交后返回 job ID，并用 `get_training_job` 查询进度。若任务长期停在
   `queued`，提示用户在独立终端启动 `phm-worker`。
5. `succeeded` 仅表示候选模型训练完成，不表示已上线。报告验证指标后，必须取得
   用户明确同意才能调用 `promote_candidate_model`。
6. 发布后只读推理 MCP 的 `active` 版本立即指向新版本，不需要重启服务；已经载入
   的旧权重缓存按版本隔离。
7. 取消正在运行的任务是协作式取消，通常在下一批次检查点生效。只有用户明确要求
   取消时才调用 `cancel_training_job`。
8. 回滚会改变在线活动版本。只有用户明确要求回滚时才调用
   `rollback_phm_model`，并在完成后报告新的活动版本。
9. 模型选择只使用验证集指标，不用测试集反复挑选候选版本。测试指标只作为最终
   报告参考。
10. 不把 FD001 仿真结果解释成真实航空器的维修、适航或安全决策依据。

## 推荐流程

```text
读取控制状态
  → 提交训练任务
  → Worker 异步训练并持续写入 epoch 进度
  → 查看候选验证/测试指标
  → 用户确认
  → 原子发布为 active
  → 用只读 PHM 工具验证预测
  → 必要时显式回滚
```
