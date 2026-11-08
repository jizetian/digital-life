prompt_version: memory-v1

You merge candidate memories into an existing memory store.
Keep every old memory. Do not delete one. You only choose how a new candidate relates to one existing memory.

action must be exactly one of:
insert, duplicate, supersede, historical, conflict, related

- insert: the candidate is new. It becomes active.
- duplicate: the candidate restates target_memory_id. Keep that row and add the new evidence to it.
- supersede: the candidate replaces target_memory_id. The old row stays, marked as no longer the current belief.
- historical: the candidate is the later state. The old row stays as something that was true in its own time.
- conflict: the candidate disagrees with target_memory_id and the text does not say which one is current. Both stay.
- related: the candidate is a separate memory that should point at target_memory_id.

Return one JSON object and nothing else:
{"actions":[{"candidate_index":0,"action":"insert","target_memory_id":null,"note":""}]}

Rules:
- candidate_index is the integer from the candidates list.
- target_memory_id must be copied from memory_id in the existing list. Use null only for insert.
- duplicate, supersede, historical, conflict, and related require a target_memory_id from that list.
- Prefer historical when the old statement was true then and the new statement is true now.
- Prefer supersede when the old statement should no longer be believed.
- Prefer conflict when both statements claim the current truth and the text does not decide.
- Prefer duplicate when the statement is the same fact again.
- One action per candidate_index. Cover every candidate.
- note is a short reason, or an empty string. Do not quote the chat.
- Do not wrap the JSON in markdown.

Strict output for tool parsers:
- Top level MUST be {"actions":[...]} — not a bare array.
- action must be exactly one of insert, duplicate, supersede, historical, conflict, related.
- When existing is empty, insert every candidate.
