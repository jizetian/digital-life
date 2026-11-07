prompt_version: l2-v1

You merge candidate traits that were built from separate batches of the same dimension.
The batches did not see each other. Combine traits that describe the same pattern.

Return one JSON object and nothing else:
{"traits":[{"statement":"...","situational_scope":"...","trait_keys":["b0t0"],"stages":[{"statement":"...","observation_ids":["observation-id"]}]}]}

Rules:
- trait_keys must be copied from the input. Do not invent keys.
- A merged trait's stages may cite observation_ids that appear on the cited traits. Do not invent observation ids.
- situational_scope says when the pattern applies.
- stages is an empty list when you do not see a change over time.
- statement is one sentence in the same language as the traits.
- If two traits should stay separate, return them as separate objects.
- Do not wrap the JSON in markdown.
