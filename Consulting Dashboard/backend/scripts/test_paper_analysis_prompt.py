"""Ad-hoc test harness for the paper-analysis prompt.

Runs a real local paper through prompts/tasks/paper_analysis.md using a local
Ollama model (representative of the free OpenRouter tier), so we can eyeball
whether the prompt produces the intended structured output.

Usage:  python -m backend.scripts.test_paper_analysis_prompt <arxiv_id> [model] [char_cap]
"""
import asyncio
import sys
from pathlib import Path

from backend.config import PROMPTS_DIR
from backend.modules.research.arxiv_bridge import get_client
from backend.modules.llm.ollama_provider import OllamaProvider


async def main() -> None:
    arxiv_id = sys.argv[1] if len(sys.argv) > 1 else "2601.22119"
    model    = sys.argv[2] if len(sys.argv) > 2 else "gemma4:latest"
    char_cap = int(sys.argv[3]) if len(sys.argv) > 3 else 24_000

    prompt_path = PROMPTS_DIR / "tasks" / "paper_analysis.md"
    task_prompt = prompt_path.read_text(encoding="utf-8").strip()

    client = get_client()
    meta = client._load_metadata().get(arxiv_id, {})
    rel  = meta.get("relative_path", "")
    res  = client.analyze_local_paper(relative_path=rel)
    if res.get("status") != "success":
        print(f"[ERROR] could not load {arxiv_id}: {res.get('message')}")
        return

    full = res["full_content"]
    body = full[:char_cap]
    truncated = len(full) > char_cap

    system = (
        f"{task_prompt}\n\n"
        f"--- PAPER ---\n"
        f"Title: {meta.get('title','')}\n"
        f"arXiv: {arxiv_id}\n\n"
        f"{body}"
        + ("\n\n(Note: paper truncated for length.)" if truncated else "")
    )
    messages = [{"role": "user", "content": "Analyze this paper following the format."}]

    print(f"=== TEST: {arxiv_id} | model={model} ===")
    print(f"Title: {meta.get('title','')}")
    print(f"Paper chars: {len(full)} (using {len(body)}{' [truncated]' if truncated else ''})")
    print(f"System prompt chars: {len(system)}")
    print("=" * 70)

    provider = OllamaProvider(base_url="http://localhost:11434", model=model)
    out = []
    async for chunk in provider.stream(messages, system):
        out.append(chunk)
        sys.stdout.write(chunk)
        sys.stdout.flush()
    print("\n" + "=" * 70)
    print(f"[done] output chars: {sum(len(c) for c in out)}")


if __name__ == "__main__":
    asyncio.run(main())
