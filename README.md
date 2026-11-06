# ChatTwin / digital-life

Phase 1：把本机的 **Moect 微信导出 JSON** 流式导入本地 **SQLite**，并给出导入质量报告。

这一阶段只做导入和标准化。不做人格蒸馏、记忆、向量库、embedding，也不做聊天 Agent。

ChatTwin is a local-first archive of a WeChat history exported by Moect. Phase 1 streams that JSON into SQLite. Personality, memory, Qdrant, and a chat agent are later phases and are not in this tree.

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

## 路线图（本仓库还没有这些代码）

1. Phase 1 — 导入与标准化（当前）
2. 切分与证据整理
3. 人格蒸馏
4. 长期记忆
5. 检索（含向量库）
6. 对话 Agent 与界面

## 开发

```bash
pytest
```

测试使用 `tests/fixtures/chattwin_moect_sample.json`，不读取你的全量导出。
