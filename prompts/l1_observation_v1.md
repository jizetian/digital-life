prompt_version: l1-v1

You extract L1 observations from one segment of a private chat archive.
An L1 observation is a small note backed by specific messages. It is not a personality profile, not long-term memory, and not a reply to the chat.

Return one JSON object and nothing else:
{"observations":[{"category":"fact","summary":"...","evidence_message_ids":["message-id"],"confidence":0.0,"time_range":{"start":0,"end":0}}]}

category must be exactly one of:
communication, emotion, opinion, value, thinking, decision, social, interest, event, fact

Rules:
- summary is one sentence in the same language as the cited messages.
- evidence_message_ids must be copied from the id= fields of the input messages. Every observation cites one or more of those ids.
- Do not cite segment_id. It is not a message id.
- Messages with overlap=true were copied from the previous segment as context. You may cite them when they support the summary.
- confidence is a number from 0 to 1.
- time_range start and end are unix seconds. Use null when the messages do not support a time.
- If nothing is supported, return {"observations":[]}.
- Do not invent message ids, events, or feelings that the text does not support.
- Do not wrap the JSON in markdown.
