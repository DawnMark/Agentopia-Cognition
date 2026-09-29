# Agentopia · Idea & Methodology

[English](https://github.com/Neph0s/Agentopia) | **简体中文**

**本项目是 [Neph0s/Agentopia](https://github.com/Neph0s/Agentopia) 的一个分支**，在原框架之上加装了一层
"认知层"：让智能体把自己的经历整理成可检索、可复用、可验证的知识，而不只是把最近发生的事留在上下文里。

原项目解决的是"能否让 AI 智能体社会有效地模拟人类生活"；这个分支追问的是下一步——
**智能体能否从自己过掉的日子里积累出东西来。** 它把一整套认知机制加进了原有的周循环，
同时保留了"一键关掉全部新功能"的对照档，以便把新机制的效果与原有行为分开测量。

---

## 一、原项目简介：Agentopia

[Agentopia](https://github.com/Neph0s/Agentopia) 是一个用于多智能体社会长期生活模拟的框架。
它以年为尺度模拟人类社会生活：在原论文的实验中，100 个智能体在 10 个模拟年份里自主参与社会生活。

它围绕两个问题构建：我们能否构建一个让智能体有效模拟人类生活的 AI 智能体社会，
以及来自这样一个社会的经验与奖励能否提升大语言模型的能力？为了回答后一个问题，
原作者定义了一种**生活奖励（life reward）**，映射社会地位、主观满足感与经济状况，
并用它训练大语言模型，提升其拟人化与角色扮演能力。

每个智能体在模拟中：

- 设定并追求个人目标，发展技能，参与经济活动
- 在情绪、物质与社交维度上发展并满足自身需求
- 与其他智能体互动，在社会中建立关系
- 在此过程中管理自己的长期记忆
- 经历一个每周循环：**计划（Plan）→ 联络（Contact）→ 活动（Activity）→ 回顾（Review）**
- 在每个年末更新档案、申请新职位，并获得一份生活奖励

一个**环境模型**（一个更强的 LLM）作为编排模拟的生成引擎：验证智能体的响应、提供反馈、
调度事件、给出客观结果，无需硬编码规则。这个角色在代码里叫 `God`。

> 完整介绍、安装与运行方式见原项目仓库。

---

## 二、本项目：加了什么、长什么样

### 2.1 出发点

原框架里，智能体的"记忆"主要是每周日记、活动记录和它在自己笔记本里维护的文本。
这套东西足以支撑短期连贯，但有三个结构性的缺口：

1. **只有最近，没有积累。** 笔记本是追加写但读取只看最新版本，能起作用的永远只有最近几周。
2. **经验不会变成能力。** 技能是 `名称 → 数值`，环境模型给多少就加多少；"练过"和"用对方法练过"
   混在同一个数里。
3. **想法不会落地。** 智能体可以产生想法，但没有从"经历"到"想法"到"可复用的做法"到"被真的拿去用"
   再到"被结果检验"的通路。

认知层就是为这三个缺口加的。

### 2.2 目标架构

```text
┌────────────────────────────────────────────┐
│               World / God                  │
│   产生客观结果、环境反馈和状态变化          │
└─────────────────────┬──────────────────────┘
                      │
┌─────────────────────▼──────────────────────┐
│          Immutable Event Ledger            │
│ activity / contact / state / schedule      │
└───────────┬────────────┬────────────┬──────┘
            │            │            │
            ▼            ▼            ▼
      Memory System   Idea Engine  Capability System
       经历与信念      候选假设     Skill + Methodology
            │            │            │
            └────────────┴──────┬─────┘
                                ▼
                    Context / Method Policy
                      下一轮计划和行动
```

三条设计原则贯穿整套实现：

- **原始经历不可变。** 活动、联络、状态、日程是事实账本；记忆、想法、方法只能引用它们，不能改写它们。
- **事实、信念、想法、能力分开。** 想法永远不能被当成事实注入；反思与教训是信念，
  可以成为假设的来源，不能单独当作证据。
- **强化必须有外部证据。** 能力或方法价值的提高必须引用真实的实践结果，反思本身不算。

认知层实现在 `src/agents/cognition/` 下，主要模块：

| 模块 | 职责 |
|---|---|
| `models.py` / `memory_models.py` / `idea_models.py` | 记忆、想法、技能、方法、事件的数据结构 |
| `event_store.py` | 追加认知事件，提供幂等检查 |
| `memory_store.py` / `memory_views.py` | 记忆的持久化、状态流转与可重建视图 |
| `memory_strength.py` / `memory_retriever.py` | 强度计算、衰减、层级与相关性检索 |
| `consolidator.py` | 周级整理：去重、合并、冲突标记、归档 |
| `relation_graph.py` | 记忆之间的关系图 |
| `idea_engine.py` / `idea_pipeline.py` / `idea_store.py` | 从记忆关系生成与筛选候选想法 |
| `lessons.py` | 把智能体自己写下的教训接进记忆与想法 |
| `methodology_policy.py` / `method_hints.py` / `offer_bandit.py` | 方法的选择、提示与生命周期 |
| `reward_model.py` | 把客观结果映射成方法强化信号 |
| `proficiency.py` / `capability_gain.py` | 由练习证据派生能力，并据此限制增益 |
| `materializer.py` | 从事件流重建当前视图 |
| `impressions.py` | 角色之间的印象文档 |

原有的 `state.skills` 在迁移期继续作为兼容投影保留。

---

## 三、新功能概览

三个大块：**记忆归档**、**Idea 引擎**、**方法 / 方法论与实践**。
三块互补：记忆提供材料，Idea 引擎把材料组合成候选，方法层负责让候选接受现实的检验。

### 3.1 记忆归档

**解决什么。** 记忆会不断增长、重复、互相冲突，需要一个可追溯的合并、衰减与遗忘机制。
这里的关键区分是：**遗忘是降低可访问性，不是删除历史。**

**怎么工作。** 每周回顾时做一次整理（consolidation）：

- 从当周证据里抽取结构化的记忆条目，并标注实体、主题、目标、技能、结果极性、障碍与资源；
- 与已有记忆比对，做去重、合并与冲突标记；
- 重算强度，按强度把记忆分到 Hot / Warm / Cold / Archived 四层；
- 建立记忆之间的关系（同一实体、同一目标、因果连续、矛盾、问题—资源互补、方法迁移等）。

**强度不只由召回次数决定**，而是综合初始显著性、时间衰减、主动读取、是否真正用于行动、
使用后是否产生有效结果、与当前目标的相关度、情绪强度、来源可信度与是否被新证据推翻。
这是为了避开"越常召回越容易被召回"的回音室效应。

**归档不等于删除。** 被归档的记忆默认不参与检索，但记录、价值与来源全部保留，
需要时可以被重新激活。

**关键数据结构。**

```json
{
  "memory_id": "memory-101",
  "kind": "episodic|semantic|relationship|goal|belief|lesson",
  "content": "复杂任务中先明确冲突可以减少返工。",
  "entities": ["project-x"],
  "topics": ["planning", "writing"],
  "goal_ids": ["goal-finish-novel"],
  "skill_ids": ["writing"],
  "method_ids": ["method-outline-first"],
  "source_event_ids": ["activity-018"],
  "confidence": 0.76,
  "salience": 0.68,
  "strength": 0.64,
  "last_recalled_at": "Y2022-W04",
  "successful_recall_count": 3,
  "status": "active|superseded|contradicted|archived",
  "protected": false,
  "created_at": "Y2021-W08"
}
```

记忆层级：

```text
Hot       默认进入近期上下文
Warm      查询相关时进入
Cold      强相关或探索检索时进入
Archived  默认不检索，但保留并可重新激活
```

禁止普通遗忘的内容：身份与人格设定、当前资产与日程等客观状态、未履行的承诺、
当前长期目标、重大关系变化、高影响经历。

### 3.2 Idea 引擎

**解决什么。** 智能体应该能组合过往经历生成新的候选假设，但这些假设**不能被误当成事实**。

**这里最重要的一个判断是：相关不等于相似。** 只靠文本相似度会漏掉真正有价值的组合——
高度相似但只是重复的记忆，能产生的想法价值反而最低；**中等相似但高度互补**的记忆才是好材料。
例如：

```text
记忆 A：每次面对空白文档都会拖延。
记忆 B：口头向朋友讲故事时表达很流畅。

→ 想法：先口述内容，再整理成文字，是否能降低写作启动阻力？
```

所以系统建立的是**记忆关系图**，而不是相似度列表。生成想法时用固定的关系模板（motif）触发：

| 模板 | 形态 |
|---|---|
| Goal + Obstacle + Resource | 我想完成 X，但 Y 一直阻碍我，我曾在 Z 中获得资源 → 能否用 Z 缓解 Y？ |
| Repeated Pattern | 多个不同事件里条件 A 都导致结果 B → 是否存在可复用的方法？ |
| Contradiction | 方法 M 有时成功有时失败 → 差异是否由情境 C 导致？ |
| Cross-domain Analogy | 领域 A 的方法解决了结构相似的问题 → 能否迁移到领域 B？ |
| Unused Resource | 存在资源 R，目标 G 尚未推进 → R 能否成为新路径？ |
| Causal Gap | 已知起点 A 和目标 C，缺少中间步骤 → 可以提出候选步骤 B。 |
| Lesson Application | 智能体自己在笔记里写下的教训 + 一个相关目标 → 一个可检验的假设。 |

最后一个模板来自角色原生的"教训"系统：它本来就会在周记和笔记本里写下"下次该怎么做"。
与其它模板的区别在于**来源**——不是系统从记忆关系里"发现"的模式，而是它自己写下的判断。

**每个候选都要过五道检查**才写入想法库：能追溯到 2–4 条记忆或一个明确的新外部事实（grounding）、
不是已有计划或方法的改写（novelty）、符合自身资源与身份（feasibility）、
能设计一次实践来验证（testability）、不会把假设当成事实（consistency）。

**关键数据结构。**

```json
{
  "idea_id": "idea-025",
  "content": "可以尝试先口述初稿，再整理成正式文本。",
  "idea_type": "hypothesis|opportunity|method_candidate|goal_adjustment",
  "motif": "problem_resource",
  "source_memory_ids": ["memory-101", "memory-084"],
  "relationship_types": ["analogy", "problem_resource"],
  "related_goal_ids": ["goal-finish-novel"],
  "related_skill_ids": ["writing"],
  "candidate_method_id": "method-voice-first",
  "novelty": 0.72,
  "feasibility": 0.61,
  "groundedness": 0.83,
  "confidence": 0.35,
  "status": "candidate|tested|adopted|rejected|expired",
  "expires_at": "Y2022-W10"
}
```

角色设定会影响候选的范围与筛选方式（创造力影响类比距离，好奇心影响探索弱关系的概率，
智力影响可行性判断的质量），但**不修改记忆关系的客观分数**。

### 3.3 方法、方法论与实践

**解决什么。** 前两块产出的都是"候选"。这一块负责把它们变成**能被真的拿去用、
并且用结果来检验的东西**。

**方法从哪来。** 四条路径：

1. **直接学习**：课程、阅读、他人指导；
2. **实践总结**：从自己成功或失败的经历里提炼；
3. **想法转化**：由 Idea 引擎的候选转换而来；
4. **方法改进**：已有方法结合新证据生成新版本，并保留父子关系。

前两条和第三条都**只建立低置信度的候选，不提高方法的价值**——价值只能来自实践结果。

**怎么被用起来。** 每周计划之前，系统给智能体一份**方法菜单**：从可提供的方法里挑出最多三条，
渲染成一段预算受限的文本，作为额外的一条消息注入计划请求。菜单的语气明确写着"用不用完全由你决定"。

关键在于**采用必须由智能体自己声明**：它在计划里写下一行 `<method>方法标题</method>` 才算采用；
程序只做匹配，不做任何"隐式推断"。**被展示不算强化**——展示、召回、被放进上下文，
都不改变方法的任何数值。只有它真的采用、并且真的练习了，才产生实践证据与价值更新。

菜单里有一类方法会被标注"（你自己在笔记里总结的教训）"或"（你自己最近想到的做法）"，
因为它们来自上面 3.1 / 3.2 两条链路，是它自己的东西。

**一次练习只产生一套证据。** 采用之后，本周第一条与该方法匹配的活动会被认领，
同一活动不会同时进入真实证据与反事实证据。

**方法的选择是一次情境决策。** 每周的菜单不是固定槽位，而是按情境打分：

```text
score(方法 | 情境) = 该情境下这个方法的价值估计
                   + 证据置信度
                   + 生命周期加分
                   + 出处加分（自己写的教训 / 自己想到的做法）
                   − 被提供却不被采用的比率
                   − 陈旧提供（连续几周被忽略的惩罚）
                   + 探索奖励（由创造力与好奇心缩放）
```

选择过程是**确定的，不含随机数**，所以同情境同历史必得同一份菜单，重放与视图重建都能精确复现。

**被拒绝也要记录。** 每个没有被采用的方法各写一条记录，带上原因
（计划里没写方法 / 采用了别的 / 这周没计划）与连续被忽略的周数。
**拒绝和被忽略同样不改变任何数值**，但必须可观测——否则提供策略只能证明它给过东西，
不能证明它给对了。

**方法会走完一个生命周期。**

```text
proposed → learned → tested → validated
                         ├→ specialized   （只在某个情境下好用）
                         ├→ deprecated    （持续失败）
                         └→ archived      （离开菜单，永不删除）
```

- `specialized` 的判据是"**某一情境**下的价值比该方法自己的整体水准高出一定幅度，
  且该情境下有足够样本"——"在这里好用"，不是"到处好用"；
- 归档只针对**持续失败且长期未被再用**的方法，归档只是离开菜单，价值、证据与归档原因全部保留；
- 后一周换了个说法重新得出的方法，若标题足够接近，记为**同一个方法的新版本**，
  而不是新增一个近义方法。

**能力由练习证据派生。** 除了原有的 `state.skills` 累加量之外，另有一套平行投影：

```text
熟练度            = 有效练习次数 / (有效练习次数 + 锚点)
覆盖度            = 练过的方法 / 该技能的方法总数
选择准确度        = 练习里"达到或超过本人常规"的比例
验证可靠性        = 练过的方法里走到 validated / specialized 的比例
有效能力          = 熟练度 × 各方法因子的加权几何平均（带下限）
```

尺度锚点按**练习次数**定义而不是按点数，这样同一句"练习写作"不会因为环境模型当次给分高低
而整体平移曲线。这套投影用于回答"懂很多却没有有效方法"这类问题，
并在实践中作为**增益封顶**的依据。

**关键数据结构。**

```json
{
  "method_id": "method-outline-first",
  "skill_id": "writing",
  "version": 2,
  "title": "先构建冲突与场景提纲",
  "description": "复杂写作任务中先确定目标、冲突和场景顺序。",
  "status": "validated",
  "source_type": "practice_reflection",
  "parent_method_id": null,
  "applicable_contexts": ["long_form", "complex_structure"],
  "contraindications": ["urgent_short_copy"],
  "steps": ["明确目标", "列出冲突", "建立场景顺序", "检查因果", "开始初稿"],
  "checks": ["每个场景是否推动冲突"],
  "failure_modes": ["规划过度导致迟迟不开始"],
  "global_value": 0.71,
  "confidence": 0.82,
  "practice_count": 14,
  "success_count": 10,
  "context_values": {
    "long_form": {"value": 0.81, "count": 9},
    "urgent_task": {"value": 0.25, "count": 3}
  },
  "source_event_ids": ["activity-001", "activity-018"],
  "last_used": "Y2022-W04"
}
```

方法价值按"向新证据靠拢"的方式更新，学习率主要由**证据数量**决定而不是由智力值决定；
**价值与置信度分离**——一次成功可以得到较高价值，但置信度仍然很低。

### 3.4 可以关掉

全部新功能都可以通过配置关闭，并保留旧行为作为对照组。
一键对照档会同时关掉十一个认知开关与体力恢复开关，
使一次运行可以直接当作"原版行为"的基线；运行目录保存的配置是**应用对照档之后**的版本，
所以任何一次运行都能自证它当时开了什么。

---

## 四、仓库结构

```
├── config.example.json     # 配置模板（复制为 config.json 并填写）
├── requirements.txt
├── data/
│   ├── apartment/          # 示例世界（来自原项目）
│   ├── school/             # 示例世界：学校场景（来自原项目）
│   └── persona_template/   # persona 数据格式模板（来自原项目）
├── docs/
│   ├── README.original.md  # 原项目 README（英文，逐字节保留）
│   ├── README.zh.md        # 原项目 README（简体中文）
│   ├── README.ja.md        # 原项目 README（日本語）
│   ├── README.ko.md        # 原项目 README（한국어）
│   ├── DEPLOY.zh.md        # 部署说明
│   ├── project-report.md   # 项目总报告：这一版与原版的差别
│   └── project-acceptance-3y.md  # 三年运行验收报告（技术）
├── scripts/
│   ├── run_world.py        # 运行模拟的主入口
│   ├── audit_cognition_run.py    # 一次运行的认知层读数与门禁
│   ├── compare_runs.py     # 两次运行的配置与账本差异
│   ├── verify_replay.py    # 重放一致性校验
│   ├── token_accounting.py # token 账
│   ├── build_rft_data.py   # 计算优势值 + 构建 RFT 训练数据
│   ├── compute_metrics.py  # 每智能体 / 每年的定量指标
│   └── time_analysis.py    # 每周墙钟耗时统计
└── src/
    ├── agents/
    │   ├── cognition/      # 认知层：记忆、想法、方法与能力（本项目新增）
    │   ├── role_agent.py   # 角色智能体：计划、联络、活动、回顾
    │   ├── data_manager.py # 数据读写与 prompt 组装
    │   └── prompts.py      # 角色侧提示词
    └── world/              # 模拟引擎：世界、调度、活动、奖励
```

## 五、模拟数据布局

每次运行都会在 `data/` 下获得自己的目录，命名为 `worldname_MMDDHHMM`
（例如 `school_06031205`）。启动时它会从基础世界（例如 `data/school/`）复制而来，
随后所有模拟输出都写入其中。除档案和配置文件外，数据均为追加写入（append-only）的 JSONL。

```
data/<world>_<MMDDHHMM>/      # 一次运行的目录（从基础世界 data/<world>/ 复制而来）
├── config.json               # 本次运行的生效配置（已应用 CLI 覆盖与对照档）
├── checkpoint.json           # 恢复检查点（最后完成的年/周/阶段）
├── worldview.json            # 世界设定 / 背景
├── positions.json            # 生成的可用职位岗位
├── locations.json            # 生成的地图
├── entity_aliases.json       # 实体别名
├── public_events.jsonl       # 世界级公共事件
├── persona/<name>/           # 每个智能体的数据
│   ├── profile/year=<YYYY>.json   # 年度档案快照
│   ├── state.jsonl                # 随时间变化的体力、满足感、技能、资产
│   ├── schedule.jsonl             # 每周日程
│   ├── activity.jsonl             # 活动结果
│   ├── reward.jsonl               # 每智能体奖励结果（社会/主观/经济/总计）
│   ├── achievement.jsonl          # 成就结算
│   ├── generation/year=<YYYY>/week=<W>.jsonl   # 原始 LLM 生成轨迹
│   ├── memory/
│   │   ├── weekly_diary.jsonl     # 每周日记条目
│   │   └── scratchpad/            # 智能体在模拟过程中自主管理的笔记本
│   │       ├── general.jsonl          # 核心笔记：长期目标、计划、进度、教训
│   │       ├── characters/<person>.jsonl   # 对该角色的了解与关系判断
│   │       └── others/<thing>.jsonl        # 其他主题的笔记
│   ├── contact/<person>.jsonl     # 智能体之间的消息记录
│   └── cognition/                 # 认知层（开启认知功能时生成）
│       ├── memory_events.jsonl         # 记忆的追加事件流
│       ├── memory_relation_events.jsonl # 记忆之间的关系事件
│       ├── idea_events.jsonl           # 想法的追加事件流
│       ├── capability_events.jsonl     # 方法、练习与能力事件
│       ├── token_usage.jsonl           # 认知层调用与 token 记账
│       └── views/                 # 从事件流重建的当前视图（可删除重建）
│           ├── memories.json
│           ├── memory_graph.json
│           ├── ideas.json
│           ├── capabilities.json
│           └── retrieval_shadow.json
├── reward/                   # 世界级奖励数据
│   ├── rankings/year=<YYYY>/week=<W>.jsonl   # PageRank 输入（好感度/尊重度）
│   └── metrics/year=<YYYY>/week=<W>.jsonl    # 每智能体的已计算奖励指标
├── position_application_log.jsonl  # 职位申请记录
└── god/<feature>/year=<YYYY>/week=<W>.jsonl  # 环境模型的生成轨迹
```

`cognition/views/` 下的文件是**物化视图**：它们完全可以从同目录的事件流重建，
因此可以删除而不丢信息。事件流本身带有稳定的事件标识与幂等键，
支持中断恢复、重复重放与并行执行。

## 六、快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 配置

```bash
cp config.example.json config.json
```

编辑 `config.json`：设置 `world.name`（例如 `apartment`、`school`）、
把 `role_model` 与 `god_model` 设成 `models` 里定义的模型名、填入对应 API 密钥与接入点，
并用 `world.time.n_year` 控制模拟时长。

**认知层的开关都在 `world.cognition` 段下**，默认打开。要跑一次"原版行为"的对照，
一行即可：

```bash
python scripts/run_world.py --baseline upstream
```

`--baseline cognition` 只关掉认知层与体力恢复，保留本分支对计划提示词的改动，
用来回答"认知层单独带来了什么变化"。

### 3. 运行模拟

```bash
python scripts/run_world.py
```

在运行时覆盖世界设置：

```bash
python scripts/run_world.py --world apartment
```

启动时会打印一行"这次运行实际生效了什么"，该清单同时写进运行目录的 `config.json`，
所以任何一次运行都能自证它当时开了哪些功能。

## 七、模型配置

Agentopia 支持多种 LLM 后端。在 `config.json` 的 `models` 下配置：

| 后端 | 必填字段 |
|---|---|
| 兼容 OpenAI（vLLM、本地） | `url`、`api_key`、`vllm_model_name` |
| Anthropic（Claude） | `api_key`、`anthropic_model_name` |
| Google Gemini（Vertex AI） | `credentials_file`、`project`、`location` |
| Azure OpenAI | `url`、`api_key`、`api_version` |

对于通过 vLLM 提供服务的具备思考能力的模型，在模型配置中设置 `"enable_thinking": true`。

## 八、许可证

本项目继承原项目，基于 MIT 许可证发布。原项目版权归其作者所有。
