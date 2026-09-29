# 确定性环境中最短成本动作搜索：UCT vs 精确求解器

在确定性、严格正整数动作成本的工具环境中搜索动作序列，比较：

- **DAG-UCT** 启发式搜索（`uctshortest/uct.py`），节点键为
  `(环境状态, 剩余执行预算)`，父边各自保存访问/奖励统计；
- **独立精确求解器** Dijkstra（`uctshortest/exact.py`），给出最优或
  无解证明。

仅使用 Python 3.11 标准库，不接入外部模型。

## 运行

```bash
# 自动化测试（31 个）
python3 -m unittest discover -s tests -v

# 搜索预算曲线（20 种子/点），输出 results/budget_curve.{csv,md}
python3 run_experiment.py
```

## 目录

- `uctshortest/environment.py` — 环境契约、调用计数与转移缓存、方案校验
- `uctshortest/uct.py` — DAG-UCT（奖励/叶子估值/扩展/模拟/回传定义）
- `uctshortest/naive.py` — 故意按环境状态粗暴去重的错误 UCT（反例）
- `uctshortest/exact.py` — 独立 Dijkstra oracle
- `uctshortest/envs/` — 表格环境与转置/陷阱/异成本环/反例/不可达场景
- `uctshortest/experiments.py` — 预算曲线实验
- `tests/` — oracle 正确性、UCT 有效性与预算计费、去重反例、RNG 重放
- `DESIGN.md` — 完整设计分析（规则、支配条件、环图统计、反例推导）

## 关键结论

- 调用预算只按**真实转移调用**计费；缓存命中不冒充调用，**模拟同样
  计费**。
- 同一环境状态、不同剩余预算不会合并；正成本保证搜索图为 DAG；
  共享节点的父边保留独立统计。
- UCT 找不到解不代表无解；找到的最好解也不是已证最优；证明只能由
  精确求解器在适用规模内给出。
- 粗暴状态去重会同时产生假阴性（漏解）与假阳性（认证超支方案），
  见 `tests/test_dedup_counterexample.py` 与 `DESIGN.md` 第 7 节。
