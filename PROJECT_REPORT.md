# Mem0-Cognitive 项目汇报文档（诚实化版本）

> **文档状态（2026 年 4 月更新）**
>
> 本文件是一份**内部导师评审用的项目阶段性汇报**，原始版本撰写于 2026 年 5 月之前、即 PR #5–#12 系列「诚实化改造」落地之前。原始版本包含若干与已落地代码不符的 claim（"Active Inference" 顶层框架、"贝叶斯优化"、"DBSCAN 聚类"、以及一组未复现的数值结果等）。
>
> 为了避免给后续读者造成误解，当前文件已按 Stage 7 的**诚实清单**风格就地 scrub，保留了原始的段落结构与中文叙事，但所有已回撤的 claim 都被显式标注。
>
> **真相来源 (single source of truth)**：
> - 论文正文：`paper/main.tex` 与 `paper/sections/*.tex`
> - 章节对齐状态：`paper/README.md` 的 *Section Status* 表
> - claim ↔ 代码 ↔ 测试的映射：`paper/README.md` 的 *Claim-to-Artifact Mapping* 表
> - 评测框架状态：`evaluation_cognitive/README.md` 的 reproducibility 清单（当前仍为 skeleton）
>
> 本文件**不**被论文、README 或评测脚本 import，仅作为历史快照保留；若本文件与上述四个来源冲突，以上述四者为准。

---

## 项目身份

- **代号**：Mem0-Cognitive
- **统一标题（与 `paper/main.tex` 保持一致）**：*Emotion-Weighted Forgetting and Sleep Consolidation for Adaptive LLM Memory*
- **提交状态**：ACL 2026 匿名评审中；双盲期间不展示投稿人/导师信息（见 `paper/main.tex` 的 `[review]` 选项与根 `README.md` 的 "Contact & Support" 段落）。
- **项目状态（当前真实状态）**：
  - 核心模块代码：已落地，含单测（retention / consolidation / integration / emotion / meta-learner 共 137 个测试，参见 `tests/cognitive/`）。
  - 评测框架：`evaluation_cognitive/` 为 **skeleton-only**（配置 YAML + 适配器接口 + `NotImplementedError` 占位）；CognitiveBench 数据生成与 LoCoMo 跑表仍是 open item。
  - 论文：章节齐全、claim 已与代码对齐（Stage 1–7）；实验表已按 Stage 6 清空为 `---`；一些 headline 数字已按 Stage 1/6/7 撤回。

~~"汇报人 / 日期 / 目标会议" 元信息~~ 由于项目进入双盲评审，此处省略。

---

## 1. 执行摘要（修订后）

本项目 **Mem0-Cognitive** 在开源记忆层项目 `mem0` 之上，提供了一个**机制层面**（mechanism-level）的拓展：三个可独立 ablate 的模块，每个模块由一个配置开关控制。

**定位**：面向 LLM 长期记忆层的**"emotion-weighted retention + asynchronous consolidation + adaptive reweighting"**工程框架。

> **已回撤的 claim**：早期版本曾把本系统叙事为 *Active Inference* 的实例，并宣称
> > "在 1000+ 轮长周期对话模拟中，Token 消耗降低 55% / 关键信息留存率提升 29% / 检索信噪比提升 62%。"
>
> 这些数字与"Active Inference"定位均已按 Stage 7 / Stage 6 撤回：
> - Active-inference 叙事：`paper/sections/introduction.tex` 与 `paper/sections/conclusion.tex` 现在使用机制层面描述；根 `README.md` 的 "Theoretical Foundation" 段落也作了相同修订。
> - 数值结果：`paper/sections/experiments.tex` 的所有表格单元格已清空为 `---`，等待 `evaluation_cognitive/` harness 实际跑通之后再填。

**（此前列出的）研究出发点**（仍然有效）：现有 RAG / 记忆系统（LangChain、MemGPT、Generative Agents、Mem0、Zep 等公开开源方案）普遍面临：
1. **记忆膨胀 (Memory Bloat)**：无限累积导致检索噪声增加。
2. **语义僵化 (Semantic Stagnation)**：缺乏从事件记忆到语义记忆的自动抽象。
3. **千人一面 (Poor Personalization)**：固定的遗忘和检索参数无法适应不同用户。

**我们的（已落地）贡献**（边界化后的 claim，与 `paper/sections/related_work.tex` 对齐）：
> 在我们所调研的一组公开开源 LLM 长期记忆层（LangChain / MemGPT / Generative Agents / Mem0 / Zep）中，我们没有看到同时具备 (a) 情感调制的保留律、(b) 带完整 audit log 的异步巩固回写、(c) 每用户自适应参数调节器，且三者均以独立可 ablate 的配置开关暴露 的公开实现。
>
> 我们**不**主张对上述任何单一认知科学观念（情感对遗忘的调制、睡眠巩固、元认知调节）拥有优先权。

---

## 2. 问题定义与动机

*（保持原始表述）* 当前的 LLM 记忆系统大多基于向量数据库的简单增删改查（CRUD）。其核心逻辑是存储→检索→FIFO 更新。这种范式的问题在于：缺乏重要性区分、缺乏演化能力、缺乏适应性。

研究问题（RQ）：
1. **RQ1**：如何将艾宾浩斯式遗忘曲线形式化为可计算的算法？
2. **RQ2**：能否模拟"睡眠巩固"机制实现从 Episodic 到 Semantic 记忆的自动转化？
3. **RQ3**：如何构建闭环反馈使记忆参数自适应优化？

三个问题的回答具体见论文 §3.1 / §3.3 / §3.4。

---

## 3. 方法论（修订后）

### 3.1 模块一：情感加权的动态遗忘

- **理论启发（不是实现声明）**：艾宾浩斯遗忘曲线 + 情绪增强记忆效应（参见 `paper/sections/methodology.tex` Eq. `\ref{eq:retention}`）。
- **保留律**：$R(m, \Delta t) = \exp\left(-\Delta t / (S_{\text{base}} \cdot (1 + \lambda E))\right)$，其中 $E \in [0, 1]$。
- **单调性**：在 $\lambda \ge 0$、$E \in [0, 1]$ 时，对每个固定的 $\Delta t$，保留概率对 $E$ **单调非减**（由 Stage 2 的 49 个方向测试 pin 住，参见 `tests/cognitive/test_retention_direction.py`）。
- **情感抽取**：两路（LLM + lexicon fallback）+ fail-open 回退到中性 $E = 0$。我们**不**把情感抽取器当作心理学测量工具（详见 `paper/sections/methodology.tex` §3.4 *Emotion Extractor: Implementation and Reliability Caveats*，以及 Stage 8 的 26 个单测 `tests/cognitive/test_emotion_analyzer.py`）。

### 3.2 模块二：异步巩固引擎（schema-style summarisation pass）

- **理论启发（不是实现声明）**：记忆巩固（McClelland 等，1995）；我们**不**声称对海马–皮层回路建立任何细粒度对应（具体参见 `paper/sections/methodology.tex` 的相应段落）。
- **工作流程**：
  1. **触发**：系统空闲时异步启动（Dream Gate 控制频率）。
  2. **聚类（修订）**：~~DBSCAN + Embeddings~~。当前实现是**简单的 greedy compatible-cluster 分派**：每个候选记忆被分到第一个余弦相似度超过 $\theta_{\text{cluster}}$ 的已有簇，或开一个新簇。当前实现**不是** DBSCAN，也不是层次聚类；DBSCAN / HDBSCAN 的 drop-in 替换留作 future work（参见 `mem0_cognitive/consolidation/engine.py::_cluster_memories`）。
  3. **抽象**：调用 LLM 对簇内记录做归纳摘要。
  4. **整合**：新生成的语义记忆写入长期库，原始短期记忆标记为"已巩固"并写入 audit log（Stage 4 补齐；14 个单测 `tests/cognitive/test_sleep_consolidator.py`）。
  5. **非递归回退**：巩固失败时回退到原记忆而不做递归重试。

### 3.3 模块三：自适应参数调节器（top-$k$ 加权启发式）

- **理论启发（不是实现声明）**：神经可塑性 + 轻量级反馈学习。
- **算法（修订）**：~~贝叶斯优化（BO with Expected Improvement）~~。当前实现是**top-$k$ 奖励加权平均的启发式**：
  - 前 `n_initial_samples` 次 uniform-random 探索。
  - 样本量到位后，取历史 top-$k$ 观测，按 reward 加权取参数均值，再按 per-dimension bounds 裁剪。
  - 这**不是** Gaussian Process 代理，也**不**计算 acquisition function。"GP-BO with Expected Improvement" 的提法已按 Stage 3 撤回；参见 `paper/sections/methodology.tex` §3.4 与 `mem0_cognitive/meta_learner/optimizer.py`。
  - 替换为真正的 GP-BO（e.g. scikit-optimize / BoTorch）是 drop-in 拓展，列在 `paper/sections/conclusion.tex` 的 "Future Directions"。
- **测试**：24 个行为契约 + convergence smoke 测试，见 `tests/cognitive/test_meta_learner.py`（Stage 8）。

---

## 4. 系统实现与技术栈（修订后的真实文件布局）

### 4.1 架构概览（与当前仓库一致）

```text
mem0gogogo/
├── mem0/                              # 上游 Mem0 SDK（仅少量 hook 接线，避免与上游冲突）
│   └── memory/main.py                 # [Stage 5] 按 opt-in 调用 cognitive hooks
├── mem0_cognitive/                    # 本项目的独立 package
│   ├── emotion/                       #   情感抽取（LLM + lexicon fallback）
│   ├── retention/                     #   保留律（paper Eq. \ref{eq:retention}）
│   ├── consolidation/                 #   异步巩固（writeback + audit + fallback）
│   ├── meta_learner/                  #   top-k 加权启发式调节器
│   └── integration/                   #   与 Memory 的 opt-in hook
├── tests/cognitive/                   # 137 个单测
│   ├── test_retention_direction.py    #   Stage 2 (49)
│   ├── test_sleep_consolidator.py     #   Stage 4 (14)
│   ├── test_integration_hooks.py      #   Stage 5 (24)
│   ├── test_emotion_analyzer.py       #   Stage 8 (26)
│   └── test_meta_learner.py           #   Stage 8 (24)
├── evaluation_mem0_original/          # 上游 Mem0 评测框架（原样保留，非本项目贡献）
├── evaluation_cognitive/              # 本项目评测（skeleton-only：YAML + 接口占位）
└── paper/                             # LaTeX 论文源码（含 Stage 7 重写的 sections）
```

### 4.2 关键技术细节

- **向量数据库**：Qdrant（通过 `mem0` 的 provider 抽象层）。
- **LLM 接口**：兼容 OpenAI API 标准，情感抽取在配置中可替换 `model_name`（默认 `gpt-4o-mini`）。
- **聚类**：~~Scikit-learn DBSCAN~~ → greedy compatible-cluster assignment（见 §3.2）。无 scikit-learn 硬依赖。
- **自适应优化**：~~自研贝叶斯优化器~~ → top-$k$ 加权均值启发式（见 §3.3）。
- **测试体系**：`pytest` + `ruff` + `isort`，137 tests、<1s 跑完、离线无网络依赖。

### 4.3 部署与兼容性

- **向后兼容**：完全不破坏 `mem0` 原 API。通过 `MemoryConfig.cognitive = True | dict | CognitiveHooksConfig` 或 env `MEM0_COGNITIVE_ENABLED=1` 启用。默认关，启用后的 hook 调用均以 `try/except` 包裹，cognitive 模块失败**不**影响 `mem0` 主路径（Stage 5 的 24 个集成测试覆盖这一契约）。
- **异步处理**：巩固与扫描通过 `Memory.run_sleep_consolidation(adapter)` 显式触发；默认**不**挂在主对话路径上（Stage 4）。
- **零训练**：情感抽取与抽象摘要均为 zero-shot prompt engineering，无模型微调。

---

## 5. 实验设计与结果（当前真实状态）

### 5.1 评测框架 CognitiveBench

**状态：skeleton-only**。`evaluation_cognitive/` 当前包含：
- `configs/`：4 条 ablation YAML（full / no-emotion / no-consolidation / no-meta-learner）+ 1 条 seed manifest schema。
- `generators/cognitivebench.py`、`adapters/locomo.py`、`scripts/run_ablation.py`、`scripts/make_tables.py`：**均为接口骨架**，公开入口抛 `NotImplementedError`，并在 `evaluation_cognitive/README.md` 的 reproducibility checklist 中列出未完成项。

对应地，论文 `paper/sections/experiments.tex` 的 4 张表中**每一个数值单元格都是 `---`**，只保留 RQ / 协议 / ablation matrix / 指标定义骨架，不作实数 claim。

### 5.2 ~~初步结果（模拟数据）~~（已按 Stage 6 / 7 撤回）

> 早期版本在此处展示了一张包含 Vanilla RAG / Mem0 Original / Generative Agents / **Mem0-Cognitive** 的对比表，宣称我方 Retention Rate 79.4% / Noise Ratio 12.3% / Token Savings 55% / Latency Overhead +35ms。
>
> 这些数字没有一条是来自 `evaluation_cognitive/` 的可复现 run。根据 Stage 1 的诚实性原则（*"Drop all unvalidated numbers from paper and README"*），上述表格整体撤回；任何后续的 headline number 必须对应 `evaluation_cognitive/` 的一次具体 run 和已 check-in 的 `expected_outputs/`。

### 5.3 对比基线（仍然是协议级 claim，不是数值 claim）

论文 `paper/sections/experiments.tex` 计划对比以下公开基线。基线选择已与 `paper/sections/related_work.tex` 的 "bounded novelty" claim 对齐：
- **Vanilla RAG**：固定窗口 + 余弦相似度。
- **Mem0 Original**：上游 `evaluation_mem0_original/` 跑表。
- **Generative Agents**：Reflection 机制。

跑表结果待 `evaluation_cognitive/` 落地后填入 `paper/sections/experiments.tex`。

---

## 6. 创新点总结（修订后的 bounded claim）

1. **工程对齐**：在 `paper/`、`mem0_cognitive/`、`evaluation_cognitive/`、`tests/cognitive/` 之间建立可审计的 claim → 代码 → 测试映射（具体见 `paper/README.md` 的 *Claim-to-Artifact Mapping* 表）。
2. **方法层贡献**（scope 已边界化；不声称对底层认知科学观念的优先权）：
   - 单调保留律 + 情感调制（paper Eq. `\ref{eq:retention}`，Stage 2 单测）。
   - 带 audit 的异步巩固引擎 + 非递归回退（Stage 4 单测）。
   - top-$k$ 加权自适应调节器（Stage 3 撤回 GP-BO 之后的 honest 描述 + Stage 8 单测）。
3. **资源贡献**：
   - 开源了 **Mem0-Cognitive** 代码库（模块化、opt-in、默认关）。
   - 发布了 **CognitiveBench** 评测骨架（目前仍为 skeleton-only，不宣称已经跑通）。
   - 提供了跨 paper / code / tests 的一致性检查清单（`paper/README.md`）。

> **已撤回的 claim**：早期版本在此处写过"首次将艾宾浩斯遗忘曲线、睡眠巩固理论和神经可塑性系统化地整合到 LLM 记忆管理框架中"。Stage 7 已把该 claim 替换为边界化版本（见 §1 末尾与 `paper/sections/related_work.tex`）。

---

## 7. 局限性与未来工作（与论文 Conclusion 对齐）

### 7.1 当前局限（显式、可枚举；与 `paper/sections/conclusion.tex` 的 Limitations 段落同步）

1. **情感抽取器不是心理学测量工具**：无 gold-annotation 校准、无 inter-annotator agreement、无 cross-model consistency；只记录抽取路径（`method` 字段 ∈ {llm, lexicon, none}），不按路径调整保留律。
2. **没有 belief update / generative model**：本框架**不**是 active inference 的实现；没有显式预测误差信号。
3. **聚类是 greedy 的**：不是 DBSCAN 也不是 HDBSCAN；我们**没有**测量过替换聚类算法对巩固质量的影响。
4. **自适应调节器是 top-$k$ 启发式**：不是 GP-BO；当前 `_explore_randomly` 的 warm-up 阶段存在一个已知的 attribution caveat（返回的 random $\phi$ 未写回 `_current_params`），见 `tests/cognitive/test_meta_learner.py` 的 convergence-smoke 测试 docstring。
5. **评测 harness 尚未跑通**：所有论文数值表均为 skeleton，headline number 待填。

### 7.2 未来计划（优先级与 `paper/sections/conclusion.tex` 的 "Future Directions" 对齐）

1. **真正的 GP-BO 调节器**：替换 `_weighted_topk_step` 为 `skopt.gp_minimize` / BoTorch；公开 API 保持不变。
2. **gold-annotated 情感校准**：建立一个小规模校准集 + inter-annotator agreement，把情感抽取器从"方法级原语"上升为"可计量器械"。
3. **更强的聚类**：DBSCAN / HDBSCAN / 层次聚类 variants，对巩固质量做对照 ablation。
4. **多模态融合**：把 Whisper 语音、图像视觉等作为额外的 $E$ 证据源。
5. **大规模用户研究**：在 `evaluation_cognitive/` 走通之后再谈。

---

## 8. ~~需要导师指导的关键点~~

在双盲评审期间该段落不再适合出现在仓库内文档中；相关讨论已移至线下。

---

## 附录：项目文件清单（修订后的真实路径）

- **核心代码**：`mem0_cognitive/{emotion,retention,consolidation,meta_learner,integration}/`
- **测试**：`tests/cognitive/` 共 5 个文件、137 个测试
- **演示脚本**：`examples/cognitive_memory_demo.py`（Stage 1 已重写为无 API key 可离线跑的版本）
- **评测框架**：`evaluation_cognitive/`（skeleton-only）与 `evaluation_mem0_original/`（上游原样保留）
- **论文源码**：`paper/main.tex` + `paper/sections/*.tex`
- **对齐状态单一来源**：`paper/README.md`（Section Status 表 + Claim-to-Artifact Mapping 表）
- **迭代依据**：根 `README.md`（"Positioning" 段落与 CI / 贡献指引）

---

*本报告最初由 Hongyi Zhou 撰写；2026 年 4 月 按 Stage 7 的诚实清单风格完成 in-place scrub，保留原始结构与中文叙事，但所有已撤回的 claim 均被显式标注，并指向 Stage 1–8 系列 PR 所落地的真相来源。*
