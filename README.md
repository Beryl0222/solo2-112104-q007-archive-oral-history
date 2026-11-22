# 口述史转写中的人物关系与异文修订

本项目整理地方档案馆将多次访谈的转写稿用于人物索引和展陈审核。受访者会用亲属称谓、旧称和方言指代同一人物，也可能在复听后修正一句话；把每版转写合并成唯一文本，会丢掉异文证据并误连人物关系。

演练资料位于 `data/oral_history.json`，内容为合成记录，保留不同来源的原始表述和修订关系。项目环境使用 Python 3.11 标准库，无第三方依赖。

## 快速开始

```bash
# 生成本地索引（JSON 到 stdout 或 --out 文件）
python3 -m oral_history.cli build

# 研究人员：按访谈查看人物、关系、事件时间线、待核项
python3 -m oral_history.cli researcher --interview I-7

# 编辑人员：查看引用版本、校订状态、待核原因
python3 -m oral_history.cli editor --interview I-8

# 单项查询
python3 -m oral_history.cli interview I-7
python3 -m oral_history.cli person person:chen-cheng
python3 -m oral_history.cli pending
python3 -m oral_history.cli search 阿成

# 本地资料与索引测试
python3 -m unittest discover -s tests -v
```

也可作为库使用：

```python
from oral_history import build_index

index = build_index()                       # 默认读取 data/oral_history.json
report = index.interview_report("I-7")      # 人物 / 关系 / 事件 / 待核 / 修订 / 引用
queue = index.review_queue()                # 所有待核项及原因
```

## 为什么不合并成“干净文本”

`data/oral_history.json` 的每一条记录都是不可变来源；复听修订是**新增一条记录**，不是覆盖原记录。索引在原文上计算字符区间（`span.start` / `span.end`），任何派生对象都能回指到“哪份访谈、第几版、哪一句话、哪几个字”。

- 原转写 `oral-1` 的文本与人工审核意见始终保留。
- 修正记录 `oral-2` 产生新的关系 `表舅—陈成`；旧关系 `舅舅—阿成` 标记为 `superseded` 并保留 `superseded_by` 指回修正后的关系。
- 引用被修订访谈的展陈稿会被标为 `stale`，但不会被删除或改写。

## 关键规则

### 同一称谓不等于同一人

`data/lexicon.json` 中的称谓（舅舅、表舅、姑姑、二姑……）只是关系词，不绑定人物。只有当**同一分句内**称谓后紧跟唯一人名（如“表舅陈成”）时才建立关系；以下情况一律不自动连边，进入待核队列：

- 只有称谓、没有人名；
- 人名在另一分句；
- 并列结构（“姑姑和阿华”）；
- 称谓后出现多个人名；
- 人名被否定（“不是阿成”）。

待核项带有 `reason`（如 `appellation_without_name`）与 `detail`，研究人员/编辑能看到为什么没有连边。

### “我听说”与亲历不是同一种证据

证据按**分句**判定，并保留命中的原话标记：

| 结果 | 含义 |
| --- | --- |
| `firsthand` | 有亲历标记（我亲眼、带我、我跟着……） |
| `hearsay` | 有传闻标记（听老人讲、听说、据说……） |
| `not_firsthand` | 明确否认亲历（没有亲眼见过、并非亲历……） |
| `mixed` | 同一分句同时含有亲历与传闻标记 |
| `unattributed` | 没有任何来源标记 |

传闻标记只向后作用于同一句的后续分句（“听老人讲，旧桥……拆了”），亲历标记绝不向后传递；否认亲历的表述始终留在自己的分句，不会被包含关系（“没有亲眼见过”含“亲眼见过”）误判为亲历。

### 事件时间线

事件按分句抽取，只记录转写中真实出现的时间词（`time_phrases`），不推算公历日期（输出中 `calendar_dates` 恒为空，并带 `no_calendar_inference: true`）。展陈稿与整理备注不进入事件时间线。

### 复听修订的影响范围

修正记录按 `revision_of`（显式）或同访谈内容匹配定位旧记录，并记录 `target_basis`。只有明确否定某个人名的修订才会：

- 把旧记录中的相关提及标为 `superseded`；
- 把旧关系标为 `superseded` 并指向新关系；
- 给旧事件加上 `revision_ids` 与 `corrected_person_ids`，原句不改。

无法定位目标或无法判断影响范围时，修订本身进入待核队列，不会猜测性改动索引。

### 展陈引用核对

展陈稿中的 `引用 I-8` 会与被引访谈的全部记录核对，输出：

- `cited_version` 与 `cited_record_ids`；
- 声明的证据类型（`claimed_evidence`）与被引原话实际证据类型；
- 问题代码与中文说明（如 `source_is_hearsay`、`archive_review_incomplete`）；
- 是否因后续复听修订而 `stale`。

## 输出对象

`ArchiveIndex` 提供 `to_dict()`、`researcher_report()`、`editor_report()`、`interview_report()`、`person_report()`、`review_queue()`、`search()`。全部输出带 `"synthetic": true`，不生成史实结论。

`to_dict()` 的主要键：

| 键 | 内容 |
| --- | --- |
| `records` | 每份来源文本、版本、修订状态、审核意见 |
| `mentions` | 人名出现位置、证据类型、是否被修订取代 |
| `appellation_uses` | 称谓出现位置、是否显式绑定人物及绑定依据 |
| `relations` | 人物关系边，含证据类型与 superseded 链 |
| `events` | 事件片段、时间词、证据类型、修订指针 |
| `pending` | 待核项、原因与片段 |
| `revisions` | 每条复听修订的目标、影响范围、被替换项 |
| `citations` | 展陈稿引用核对结果 |

## 目录

```
data/oral_history.json     合成访谈记录（来源，不修改）
data/lexicon.json          合成词表：人名、称谓、旧地名、证据标记
oral_history/models.py     来源加载与不可变记录
oral_history/lexicon.py    词表与原文匹配
oral_history/analysis.py   分句、证据判定、称谓绑定
oral_history/index.py      关系/事件/修订/引用/待核索引
oral_history/cli.py        命令行入口
tests/                     标准库 unittest
```
