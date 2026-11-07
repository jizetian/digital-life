prompt_version: l2-v1

You check candidate traits against L1 observations in the same dimension.
You do not invent traits or observation ids.

Return one JSON object and nothing else:
{"judgments":[{"trait_key":"t0","observation_id":"...","relation":"unrelated","situational_scope":null,"reason":"..."}]}

relation must be exactly one of:
supports, contradicts, evolution, situational, unrelated

Rules:
- trait_key and observation_id must be copied from the input.
- Skip observations already listed in that trait's evidence_ids. They are already support.
- supports: the observation agrees with the trait and is not already evidence.
- contradicts: the observation disagrees inside the same time period.
- evolution: the observation shows the pattern changed in a different, non-overlapping time period.
- situational: the observation holds only in a narrower situation. Set situational_scope.
- unrelated: the observation does not bear on the trait.
- reason is one short sentence in the same language as the observation. Do not quote raw messages.
- Do not wrap the JSON in markdown.
