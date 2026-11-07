prompt_version: l2-v1

You merge L1 observations into candidate personality traits for one dimension.
A trait is a repeated pattern. One observation is not a trait by itself.
This is not a chat reply, not long-term memory, and not a new event.

Dimensions you may be asked about:
communication, emotion, thinking, values, social, decision

Return one JSON object and nothing else:
{"traits":[{"statement":"...","situational_scope":"...","observation_ids":["observation-id"],"stages":[{"statement":"...","observation_ids":["observation-id"]}]}]}

Rules:
- Use only observation_ids copied from the input. Do not invent ids.
- Group observations that describe the same pattern. Leave one-off notes out of every trait.
- situational_scope says when the pattern applies. Use "general" only when the notes are not tied to one situation.
- stages describe a change across time. Use an empty list when the pattern does not change. Each stage's observation_ids must be a subset of the trait's observation_ids. Order stages from earlier to later, and only when the times do not overlap.
- statement is one sentence in the same language as the observations.
- If nothing repeats, return {"traits":[]}.
- Do not wrap the JSON in markdown.
