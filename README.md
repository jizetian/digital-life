# ChatTwin / digital-life

Phase 1 把本机的 **Moect 微信导出 JSON** 流式导入本地 **SQLite**。Phase 2 在这张表上做**会话切分**和 **L1 观察**的批处理骨架：时间间隔、长度上限、少量重叠，然后用可替换的本地模型客户端抽出带消息编号的观察。

这里没有人格 L2/L3、长期记忆、Qdrant、embedding，也没有数字分身聊天 Agent。

ChatTwin is a local-first archive of a WeChat history exported by Moect. Phase 1 streams that JSON into SQLite. Phase 2 segments those messages and distills L1 observations. Personality, long-term memory, Qdrant, and a chat agent are later phases and are not in this tree.

## 聊天记录只留在本机

全量导出大约几百 MB、几十万条消息。它包含私人对话。

- 把导出文件放在本机，例如 `data/raw/moect_export/你的导出.json`。
- 这个目录里的 `*.json` 已被 `.gitignore` 忽略，**不要**把全量导出提交到 Git，也不要上传到网盘、工单或聊天工具。
- 生成的 `*.db` 同样被忽略。SQLite 里是清洗后的原文，一样不能提交。
- 导入 API 只应绑定在 `127.0.0.1`。它不是对外服务。

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

消息条数或切分参数变了，需要显式重建。重建会删掉切分结果，以及级联的蒸馏任务和 L1 观察：

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

## 路线图

1. Phase 1 — 导入与标准化（已在本仓库）
2. Phase 2 — 切分与 L1 观察骨架（当前）
3. 人格蒸馏（L2/L3）
4. 长期记忆
5. 检索（含向量库）
6. 对话 Agent 与界面

## 开发

```bash
pytest
```

测试使用 `tests/fixtures/chattwin_moect_sample.json` 和合成消息，不读取你的全量导出，也不会连接真实模型。
