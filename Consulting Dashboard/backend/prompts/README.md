# System-prompt fragments

Drop `.md` or `.txt` files here. Their contents are concatenated and injected
into the chat system prompt on **every** API call (Claude / Ollama / OpenRouter
alike) — no server restart needed, because files are read fresh each request.

## Conventions

- **Order** — files are sorted by filename. Use numeric prefixes to control
  the order they appear in the prompt:
  ```
  00_role.md          # who the assistant is
  10_alpha_dsl.md     # WorldQuant operator reference / syntax rules
  20_output_format.md # how answers / code should be structured
  30_examples.md      # few-shot examples
  ```
- **Disable a fragment** — rename its extension to anything other than
  `.md` / `.txt` (e.g. `10_alpha_dsl.md.off`). It stays in the folder but is
  no longer injected.
- **Skipped automatically** — this `README.md`, hidden files (leading `.`),
  and empty files.

## Where it's wired

- Loader: `backend/modules/llm/prompt_loader.py` → `load_prompt_fragments()`
- Injection point: `backend/routers/chat.py` → `_build_system()`
- Folder path (override via `PROMPTS_DIR` env var): `backend/config.py`

The final system prompt is assembled as:

```
<base role prompt>  +  <these fragments, in filename order>  +  <selected paper text>
```

## Notes

- Fragments count as input tokens on every turn — keep them tight and
  imperative. Long fragments + full paper text fill the context window fast,
  especially on small free models.
- Free OpenRouter models follow long/complex instructions less reliably than
  Claude. If output drifts from your intent, shorten the fragment and add a
  concrete example rather than adding more rules.
