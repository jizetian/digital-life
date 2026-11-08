# ChatTwin / digital-life

Phase 1 把本机的 **Moect 微信导出 JSON** 流式导入本地 **SQLite**。Phase 2 在这张表上做**会话切分**和 **L1 观察**。Phase 3 把这些观察跨段合并成 **L2 特质**，再写成 **L3 人格档案**。Phase 4 从会话段和 L1 观察里抽出**长期记忆**：事实、事件、关系、看法、情境，连同证据、冲突和时间线。Phase 5 用 **BGE-M3**（或可替换的嵌入客户端）、Qdrant 和 SQLite FTS5 做**混合检索**。

这里没有数字分身聊天 Agent，也没有聊天界面。

ChatTwin is a local-first archive of a WeChat history exported by Moect. Phase 1 streams that JSON into SQLite. Phase 2 segments those messages and distills L1 observations. Phase 3 merges those observations into a personality profile. Phase 4 stores long-term memories with evidence and a timeline. Phase 5 searches segments, memories, and personality traits with embeddings and SQLite FTS5. A chat agent is a later phase and is not in this tree.

## 聊天记录只留在本机

全量导出大约几百 MB、几十万条消息。它包含私人对话。

- 把导出文件放在本机，例如 `data/raw/moect_export/你的导出.json`。
- 这个目录里的 `*.json` 已被 `.gitignore` 忽略，**不要**把全量导出提交到 Git，也不要上传到网盘、工单或聊天工具。
- 生成的 `*.db` 同样被忽略。SQLite 里是清洗后的原文，一样不能提交。
- `personality/` 根目录下生成的档案（`*.md`、`*.json`）也被忽略。仓库里只保留 `personality/examples/` 的合成模板。
- `data/qdrant/` 是向量索引，可以删掉再从 SQLite 重建。不要提交。
- 导入和检索 API 只应绑定在 `127.0.0.1`。它不是对外服务。

仓库里唯一的对话样本是开发用的前 300 条：`tests/fixtures/chattwin_moect_sample.json`。它不能代替你的全量文件。

## 安装

需要 Python 3.11+。在仓库根目录：

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pytest
```

## 导入全量 JSON

路径由你本机决定，下面只是示例：

```bash
python -m scripts.import_data \
  --input data/raw/moect_export/你的导出.json \
  --db data/database/cleaned_messages.db
```

中断之后接着跑（按 `messages` 数组下标断点，重新打开文件，已写入的主键不会重复）：

```bash
python -m scripts.import_data \
  --input data/raw/moect_export/你的导出.json \
  --db data/database/cleaned_messages.db \
  --resume
```

同一文件再导入一次是幂等的：`message_id` 优先用 `server_id`，没有则用 `session:local_id`。已存在的主键记为重复，不会新插一行。

结束时 CLI 会打印导入报告，并写入 `import_jobs.stats_json`。报告包含：总消息数、成功、失败、重复、时间缺失、发送者缺失、各 `kind` / `type_name` 数量、目标人物消息比例、文本消息比例。

如果导出文件的字节大小和未完成任务不一致，`--resume` 会拒绝继续，避免把两个不同文件接在同一次断点上。这时去掉 `--resume` 重新导入即可（主键仍然幂等）。

## 字段怎么对应

身份只看 Moect 的 `is_sender`：

| `is_sender` | 角色 |
|---|---|
| `true` | `owner`（导出者） |
| `false` | `target_person`（对方） |
| 缺失 | `unknown`（不靠昵称猜测） |

`metadata.session` / `metadata.name` 会先写入 `sender_mapping`，作为目标人物的默认身份。

| 统一字段 | Moect 字段 |
|---|---|
| `message_id` | `server_id`，空则 `session` + `local_id` |
| `conversation_id` | `session` |
| `sender_id` / `sender_name` | `sender_wxid` / `sender` |
| `timestamp` | `create_time`（不用 `time_str` 反推） |
| `message_type` | `kind`（不用类型码臆造） |
| `content` | `text`（原文保留；媒体占位如 `[图片]` 仍保留，但 `kind` 不是 `text`） |
| `raw_content` | 去掉 `avatar_url` 后的原始对象 |
| `is_forwarded` | 仅在 `kind` / `type_name` / `extra` 上有明确转发标记时为真，否则为假。引用回复（`appmsg_type` 57）不是转发 |
| `source_file` / `source_position` | 文件名，以及 `messages` 数组下标（从 0 开始） |

不能确认的字段留空。

SQLite 表：`messages`（含会话、发送者、时间、类型索引）、`import_jobs`、`import_errors`、`sender_mapping`、`processing_jobs`（空壳，留给后续阶段）。

大文件用分块流式解析，不把整个 JSON `json.load` 进内存。默认每 1000 条提交一次，可在 `config/settings.yaml` 里改 `import.batch_size`。

## HTTP

只在本机启动：

```bash
PYTHONPATH=src uvicorn api.main:app --host 127.0.0.1 --port 8000
```

```bash
curl -s -X POST http://127.0.0.1:8000/api/v1/import \
  -H 'content-type: application/json' \
  -d '{"input_path":"data/raw/moect_export/你的导出.json","db_path":"data/database/cleaned_messages.db"}'

curl -s http://127.0.0.1:8000/api/v1/import/<job_id>
```

`POST /api/v1/import` 返回 `job_id`。`GET /api/v1/import/{job_id}` 返回状态、进度和统计。可选查询参数 `db` 指向同一个 SQLite 文件。`resume: true` 时继续该文件上最近一个未完成任务。

## Phase 2：切分

切分只读 SQLite 里的 `messages`，按会话把流式结果写成 `conversation_segments`。它不把整段聊天拼成一段无法回溯的文本。

默认参数（`config/settings.yaml` 的 `segmentation`，也可用环境变量覆盖）：

| 参数 | 默认 | 含义 |
|---|---|---|
| 时间间隔 | 30 分钟 | 相邻两条消息的时间差达到这个值就新开一段。缺时间戳的消息不按间隔切开 |
| 段长上限 | 50 条 | 一段里最多存这么多条，含重叠 |
| 重叠 | 4 条 | 因为达到段长而切开时，下一段带上上一段末尾这几条。不跨会话，默认也不跨时间间隔 |

时间间隔切开的新段默认不带重叠。若要带上，加 `--overlap-on-time-gap`。

每段保存：起止时间、有序 `message_ids`、重叠部分的 message id、`target_message_count`（去掉重叠前缀后这段真正覆盖的条数）、`processing_status`。`conversation_segment_messages` 再按位置指向 `messages.message_id`，所以证据可以回到原消息。

```bash
python -m scripts.segment_conversations \
  --db data/database/cleaned_messages.db
```

同一库、同一参数、消息总数没变时，再跑一次不会重写。中断后直接再跑，会从上次提交的消息之后继续，并带上当时还没关闭的那段。

消息条数或切分参数变了，需要显式重建。重建会删掉切分结果，以及级联的蒸馏任务、L1 观察、人格表和长期记忆：

```bash
python -m scripts.segment_conversations \
  --db data/database/cleaned_messages.db \
  --rebuild
```

`--max-segments 20` 可以只关闭若干段就停下，检查点保持 `running`，下次不带这个参数会接着做。大约几十万条消息按页读取（默认 2000），不会整表装进内存。

蒸馏命令本身在这一轮结束时是 `completed`（没有还要做的任务）或 `paused`（还有 `pending` / `retrying` / 中断的 `running` 任务）。HTTP 任务也同样：线程退出后状态是 `completed`、`paused` 或 `failed`，不会停在看起来还在跑的 `running`。单条任务的 `running` 只表示这次模型调用还没写完；进程如果停在那里，下次运行会把它重新领走。

## Phase 2：L1 蒸馏

蒸馏按段调用一个 **OpenAI 兼容** 的 HTTP 接口。本机 Ollama 用它的 `/v1` 地址即可，不需要把聊天记录送到仓库外面。

```bash
# 示例。模型名换成你本机已经拉取的名字。
export CHATTWIN_LLM_BASE_URL=http://127.0.0.1:11434/v1
export CHATTWIN_LLM_MODEL=qwen2.5:7b

python -m scripts.distill_l1 \
  --db data/database/cleaned_messages.db \
  --batch-size 8
```

托管的兼容接口把密钥留在本机环境变量里，不要写进仓库：

```bash
export CHATTWIN_LLM_BASE_URL=https://api.openai.com/v1
export CHATTWIN_LLM_MODEL=gpt-4o-mini
export CHATTWIN_LLM_API_KEY=sk-...
```

模型要返回 JSON。类别只接受：`communication`、`emotion`、`opinion`、`value`、`thinking`、`decision`、`social`、`interest`、`event`、`fact`。每条观察有 `summary`、`evidence_message_ids`、`confidence`、`time_range`。证据 id 必须落在该段的 message id 里，对不上的会被丢掉；一条观察若没有任何有效证据，就不会入库。

任务主键：

```text
task_id = sha256(f"{segment_id}|{prompt_version}|{model_version}")
```

`prompt_version` 当前是 `l1-v1`（`prompts/l1_observation_v1.md`）。`model_version` 默认等于模型名，需要让缓存失效时用 `CHATTWIN_MODEL_VERSION` 或 `--model-version`。

状态：`pending`、`running`、`completed`、`failed`、`retrying`、`skipped`。没有正文的段标记为 `skipped`，不会打模型。`completed` 和 `skipped` 再次运行会跳过。进程停在 `running`、或失败后变成 `retrying`，下次同一条命令会接着做。`--max-tasks` 处理若干次就停，方便续跑。默认最多尝试 3 次，用尽后是 `failed`；`--retry-failed` 才重新排队。

改提示词内容时要同时改 `PROMPT_VERSION`，否则同一 `task_id` 会一直被当成已经完成。

```bash
python -m scripts.show_status --db data/database/cleaned_messages.db
```

HTTP 仍只绑本机。切分和蒸馏在后台线程里跑，立刻返回 `job_id`：

```bash
curl -s -X POST http://127.0.0.1:8000/api/v1/segment \
  -H 'content-type: application/json' \
  -d '{"db_path":"data/database/cleaned_messages.db"}'

curl -s http://127.0.0.1:8000/api/v1/segment/<job_id>

curl -s -X POST http://127.0.0.1:8000/api/v1/distill \
  -H 'content-type: application/json' \
  -d '{"db_path":"data/database/cleaned_messages.db","batch_size":8}'

curl -s http://127.0.0.1:8000/api/v1/distill/<job_id>
curl -s http://127.0.0.1:8000/api/v1/distill/status?db=data/database/cleaned_messages.db
```

测试用假的模型客户端，`pytest` 不需要本机有 Ollama。

## Phase 3：人格蒸馏

L2 只读已经落库的 L1 观察（摘要、类别、证据消息 id、时间），不再把原文发给模型。L3 把通过检查的特质写成六份维度说明，外加总览和证据 JSON。

六个维度：

| 维度 | L1 类别 | 文件 |
|---|---|---|
| 语言与沟通 | `communication` | `communication_style.md` |
| 情绪 | `emotion` | `emotional_patterns.md` |
| 思维 | `thinking`、`opinion` | `thinking_patterns.md` |
| 价值观 | `value` | `values.md` |
| 社交 | `social` | `social_patterns.md` |
| 决策 | `decision` | `decision_patterns.md` |

`interest`、`event`、`fact` 只作为上下文计数，不会变成特质。

一条特质要进入档案的「稳定」部分，支持它的观察必须来自至少两个**独立**证据组（默认 2，可用 `--min-independent` 改）。同一段里的观察算一组；证据消息 id 有交集的也算一组（切分重叠会把消息复制到下一段）。不够独立的特质记为 `provisional`，只出现在 `personality_evidence.json`。同一时段里相反的观察记为反证；反证组数不少于独立证据组时，特质是 `contested`，写在档案里单独一节，不当成已确立的人格。时间上错开的相反观察记成阶段变化，而不是反证。`situational_scope` 记下这个特质在什么情境里成立。

每条特质都带 `confidence`、`evidence_count`、`counter_evidence_count`、`valid_from`、`valid_to`、`evidence_ids`。`evidence_ids` 是 L1 的 `observation_id`。证据 JSON 里再展开到 `message_ids`，所以可以回到原消息，但档案本身不复制原文。

模型调用仍是 OpenAI 兼容接口，和 Phase 2 共用 `CHATTWIN_LLM_BASE_URL`、`CHATTWIN_LLM_MODEL`、`CHATTWIN_LLM_API_KEY`。任务主键：

```text
task_id = sha256(f"{kind}|{dimension}|{batch_key}|{prompt_version}|{model_version}")
```

`kind` 是 `merge`（一段观察合成候选特质）、`reduce`（跨批次合并；只有一个批次时跳过模型）或 `contradict`（对照反证、阶段和情境）。`prompt_version` 默认 `l2-v1`（`prompts/l2_merge_v1.md`、`prompts/l2_reduce_v1.md`、`prompts/l2_contradict_v1.md`）。`model_version` 默认等于模型名。改提示词时要同时改 `PROMPT_VERSION`，否则旧的 `task_id` 会一直被当成已经完成。

`completed` 和 `skipped` 再次运行不会打模型。进程停在 `running`、或失败后变成 `retrying`，下次同一条命令会接着做。一次运行里每个任务只尝试一次，用尽 `--max-attempts`（默认 3）后是 `failed`。`--retry-failed` 才重新排队。档案文件只在这一版提示词和模型的任务全部完成、且没有失败时写出。

```bash
python -m scripts.build_personality \
  --db data/database/cleaned_messages.db \
  --output personality \
  --batch-size 20
```

中断后续跑同一条命令即可。`--max-tasks 4` 处理若干次模型调用后停下。

```bash
python -m scripts.show_status --db data/database/cleaned_messages.db
```

```bash
curl -s -X POST http://127.0.0.1:8000/api/v1/personality \
  -H 'content-type: application/json' \
  -d '{"db_path":"data/database/cleaned_messages.db","output_dir":"personality","batch_size":20}'

curl -s http://127.0.0.1:8000/api/v1/personality/<job_id>
curl -s http://127.0.0.1:8000/api/v1/personality/status?db=data/database/cleaned_messages.db
```

生成的 `personality/*.md` 和 `personality/*.json` 不要提交。格式示例在 `personality/examples/`。切分如果 `--rebuild`，会清掉 L1、人格表和记忆表；档案文件要再跑一次人格命令才会重写。

## Phase 4：长期记忆

记忆只从已经落库的会话段，以及这段上已有的 L1 观察里来。模型看到的是消息 id、角色、时间和截断后的正文，外加观察的摘要和证据 id。写进记忆表的是一句陈述，不是原文副本。证据必须能回到 `message_id`。对不上该段的 id 会被丢掉；一条候选若最后没有任何有效消息证据，不会入库。没有 L1 也能抽，有 L1 时观察 id 可以带上它自己的消息证据。

五种类型：

| 类型 | 含义 |
|---|---|
| `fact` | 个人事实，例如住在哪里、做什么工作 |
| `event` | 发生过的事 |
| `relationship` | 人或其他实体之间的关系 |
| `opinion` | 某人持有的看法 |
| `situational` | 只在某种情境下成立。`situation` 记下这个情境 |

四种状态。冲突时**不删除旧行**：

| 状态 | 何时使用 |
|---|---|
| `active` | 当前仍成立 |
| `historical` | 过去成立，现在有了后续状态。旧行留下，关系是 `historical_of` |
| `superseded` | 旧陈述不再被相信，被新行替换。旧行留下，`superseded_by` 指向新行，关系是 `supersedes` |
| `uncertain` | 两边都声称当前事实，文本分不出对错。两边都留下，关系是 `conflicts_with` |

同一句陈述再次出现时并进原来的行，只追加证据，不新插一行。每条记忆有 `time_start` / `time_end`、`time_precision`、`importance`、`confidence`、`evidence_message_ids`。`memory_evidence` 再把支持证据指到 `message_id` 和 `observation_id`。反证消息记在 `evidence_role = conflict`，不写进 `evidence_message_ids`。

实体在 `entities`。`status` 只允许 `confirmed`（文本写明了）、`inferred`（从上下文推断）、`unknown`（提到了但定不下来）。同一个类型加归一化名字合并成一行。状态只会从 `unknown` 升到 `inferred` 再升到 `confirmed`，不会降回去。`memory_entities` 记下它在这条记忆里的角色：`subject`、`object`、`participant`、`location`、`mentioned`。

模型仍是 OpenAI 兼容接口，和前面阶段共用 `CHATTWIN_LLM_BASE_URL`、`CHATTWIN_LLM_MODEL`、`CHATTWIN_LLM_API_KEY`。每个段两次调用：`extract` 抽候选，`merge` 对照已有记忆。已有记忆以摘要形式送进去，不再重发整段原文。

```text
task_id = sha256(f"{segment_id}|{kind}|{prompt_version}|{model_version}")
```

`kind` 是 `extract` 或 `merge`。`prompt_version` 默认 `memory-v1`（`prompts/memory_extract_v1.md`、`prompts/memory_merge_v1.md`）。两份提示词共用这一个版本号；改了其中一份就要一起改 `PROMPT_VERSION`，否则旧的 `task_id` 会一直被当成已经完成。`model_version` 默认等于模型名。

`completed` 和 `skipped` 不会再次打模型。进程停在 `running`、或失败后变成 `retrying`，下次同一条命令会接着做。一次运行里每个任务只尝试一次，用尽 `--max-attempts`（默认 3）后是 `failed`。`--retry-failed` 才重新排队。没有正文、也没有 L1 的段，extract 标成 `skipped`。extract 没有候选时，merge 也标成 `skipped`，不打模型。

```bash
python -m scripts.build_memories \
  --db data/database/cleaned_messages.db \
  --batch-size 4
```

中断后续跑同一条命令即可。`--max-tasks 4` 处理若干次模型调用后停下，检查点记在 `memory_tasks` 和 `memory_checkpoint`。

时间线把 `historical` 和 `superseded` 也列出来，按 `time_start` 排序，没有时间的放在最后：

```bash
python -m scripts.show_timeline --db data/database/cleaned_messages.db
python -m scripts.show_timeline --db data/database/cleaned_messages.db --status active --entity 北京
python -m scripts.show_status --db data/database/cleaned_messages.db
```

`memories_fts` 是 FTS5 外部内容表，由触发器跟着 `memories.statement` 更新。它只服务记忆表自己的词法查找。跨段、记忆和人格的混合检索在 Phase 5，用的是另一张 `retrieval_fts`。

```bash
curl -s -X POST http://127.0.0.1:8000/api/v1/memory \
  -H 'content-type: application/json' \
  -d '{"db_path":"data/database/cleaned_messages.db","batch_size":4}'

curl -s http://127.0.0.1:8000/api/v1/memory/<job_id>
curl -s http://127.0.0.1:8000/api/v1/memory/status?db=data/database/cleaned_messages.db
curl -s "http://127.0.0.1:8000/api/v1/memory/timeline?db=data/database/cleaned_messages.db&status=active"
```

任务返回值里是计数，不是陈述，也不是原文。时间线接口返回陈述和证据 id，不返回消息正文。切分 `--rebuild` 会清掉这些记忆表。

SQLite 新增：`entities`、`memories`、`memory_evidence`、`memory_entities`、`memory_relations`、`memory_tasks`、`memory_checkpoint`、`memories_fts`。

## Phase 5：混合检索

检索读三张已经落库的表，写成一张可重建的投影，再把稠密向量放进 Qdrant。SQLite 仍是原文和证据的来源。删掉 `data/qdrant/` 之后重跑索引命令就能把向量补回来。投影里没有 Moect 原始 JSON。

| 集合 | SQLite 来源 | 索引进去的文本 |
|---|---|---|
| `conversation_segments` | `conversation_segments` 加上按位置排好的消息 | 角色、时间和正文。空段跳过 |
| `long_term_memories` | `memories` | 类型、状态、陈述、情境、实体名 |
| `personality_observations` | `l2_traits` | 维度、状态、特质陈述、情境范围 |

`personality_observations` 是 L2 特质，不是 L1。证据 id 留在元数据里，可以回到观察或消息，但向量库的 payload 只放过滤字段，不放正文。

默认嵌入模型是 **BGE-M3**，走和聊天模型一样的 OpenAI 兼容接口，路径是 `POST {base}/embeddings`。本机 Ollama 先拉模型再索引：

```bash
# 模型名换成你本机已经拉取的 BGE-M3。Ollama 上通常叫 bge-m3。
ollama pull bge-m3
export CHATTWIN_EMBEDDING_MODEL=bge-m3
export CHATTWIN_EMBEDDING_BASE_URL=http://127.0.0.1:11434/v1

python -m scripts.build_embeddings \
  --db data/database/cleaned_messages.db
```

`CHATTWIN_EMBEDDING_BASE_URL` 为空时，沿用 `CHATTWIN_LLM_BASE_URL`。密钥沿用 `CHATTWIN_LLM_API_KEY`，也可以单独设 `CHATTWIN_EMBEDDING_API_KEY`。不要把密钥写进仓库。

SQLite 里记下的模型或维度变了，或者向量条数和嵌入记录对不上（比如删了 `data/qdrant/`），下一次索引会把向量补回去。文本没变、模型没变、向量也还在的行会跳过。`--rebuild` 则强制全部重嵌。上游切分如果 `--rebuild`，会清掉这张投影；记忆和特质写完之后要再跑一次索引。

`--provider hash` 是测试和离线冒烟用的确定性向量，不是 BGE-M3。`--provider sentence_transformers` 只有本机装了 `sentence-transformers` 时才会加载 `BAAI/bge-m3`，仓库不依赖它，CI 也不需要 GPU。

Qdrant 默认写在 `data/qdrant`（本地目录，不另起服务）。已经有服务时用 `--qdrant-url http://127.0.0.1:6333`。`--backend memory` 只活在当前进程里，命令退出就没了。

查询把稠密检索和 `retrieval_fts`（FTS5 trigram，中英文至少 3 个字）用倒数排名融合到一起，按 `doc_id` 去重。一条文档如果两边都命中，只出现一次，并带上 `vector_rank` 和 `fts_rank`。可选的重排器默认关闭。设了 `CHATTWIN_RERANK_URL` 之后，CLI 加 `--rerank`，或 API 传 `"rerank": true`，才会把融合结果 POST 出去。响应要是 `{"results": [{"id", "score"}]}`。没配地址就不要打开这个开关。

```bash
python -m scripts.search \
  --db data/database/cleaned_messages.db \
  --query "搬到了上海"

python -m scripts.search \
  --db data/database/cleaned_messages.db \
  --query "利弊" \
  --collection personality_observations \
  --no-vector
```

`--memory-status`、`--memory-type`、`--trait-status`、`--dimension` 只约束对应的那一类命中，其它集合仍会返回。时间未知的文档在带了时间范围时会被滤掉。

```bash
curl -s -X POST http://127.0.0.1:8000/api/v1/embeddings \
  -H 'content-type: application/json' \
  -d '{"db_path":"data/database/cleaned_messages.db"}'

curl -s http://127.0.0.1:8000/api/v1/embeddings/<job_id>
curl -s "http://127.0.0.1:8000/api/v1/embeddings/status?db=data/database/cleaned_messages.db"

curl -s -X POST http://127.0.0.1:8000/api/v1/search \
  -H 'content-type: application/json' \
  -d '{"db_path":"data/database/cleaned_messages.db","query":"搬到了上海","limit":10}'
```

索引任务的返回值是计数，不是正文。搜索接口会返回索引进去的文本、分数和证据 id，因为不返回这些就没法看结果。它仍然不是聊天 Agent。

SQLite 新增：`retrieval_documents`、`retrieval_embeddings`、`retrieval_fts`。

## 路线图

1. Phase 1 — 导入与标准化（已在本仓库）
2. Phase 2 — 切分与 L1 观察（已在本仓库）
3. Phase 3 — 人格蒸馏 L2/L3（已在本仓库）
4. Phase 4 — 长期记忆（已在本仓库）
5. Phase 5 — 混合检索（当前）
6. 对话 Agent 与界面

## 开发

```bash
pytest
```

测试使用 `tests/fixtures/chattwin_moect_sample.json`、合成消息和 `tests/fixtures/l1_personality_observations.json`。它们不读取你的全量导出，也不会连接真实模型或 GPU。记忆测试用假的模型客户端。检索测试用确定性的嵌入客户端和内存向量库；Qdrant 适配器用客户端自带的内存模式，不连外部服务。
