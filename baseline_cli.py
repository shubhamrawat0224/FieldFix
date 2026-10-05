#!/usr/bin/env python3
import csv
import json
import os
import sys
from pathlib import Path
from urllib import error, request

SYSTEM_PROMPT = "You are FieldFix, a copilot for Amperia field technicians."


def load_env_file() -> None:
    dotenv_path = Path(__file__).resolve().parent / ".env"
    if not dotenv_path.exists():
        return

    for raw_line in dotenv_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue

        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {'"', "'"}:
            value = value[1:-1]

        os.environ.setdefault(key, value)


load_env_file()


def ask_llm(question: str) -> str:
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "OPENAI_API_KEY is not set. Export it first: export OPENAI_API_KEY='your-key'"
        )

    preferred_models = []
    configured_model = os.environ.get("OPENAI_MODEL")
    if configured_model:
        preferred_models.append(configured_model)

    preferred_models.extend([
        "openai/gpt-5-mini",
        "openai/gpt-4.1-mini",
        "openai/gpt-4o-mini",
        "openai/gpt-4o",
    ])
    seen = set()
    ordered_models = []
    for model in preferred_models:
        if model and model not in seen:
            seen.add(model)
            ordered_models.append(model)

    last_error = None
    for model in ordered_models:
        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": question},
            ],
            "temperature": 0.2,
        }

        req = request.Request(
            f"{os.environ.get('OPENAI_BASE_URL', 'https://api.openai.com/v1').rstrip('/')}/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {api_key}",
            },
            method="POST",
        )

        try:
            with request.urlopen(req, timeout=60) as response:
                resp_body = response.read().decode("utf-8")
            data = json.loads(resp_body)
            try:
                return data["choices"][0]["message"]["content"].strip()
            except (KeyError, IndexError, TypeError, ValueError):
                raise RuntimeError(f"Unexpected OpenAI API response: {resp_body[:500]}")
        except error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            last_error = RuntimeError(f"OpenAI API request failed for model {model} ({exc.code}): {detail}")
            parsed = json.loads(detail) if detail.startswith("{") else {}
            error_code = parsed.get("error", {}).get("code")
            if error_code not in {"model_not_found", "invalid_request_error"}:
                raise last_error from exc
            continue
        except error.URLError as exc:
            raise RuntimeError(f"OpenAI API connection failed: {exc.reason}") from exc

    if last_error is not None:
        raise last_error
    raise RuntimeError("No OpenAI model was available for this project.")


def pick_best_startup_strategy() -> str:
    scorecard_path = Path(__file__).resolve().parent / "scorecard.csv"
    if not scorecard_path.exists():
        return "hybrid"

    with scorecard_path.open("r", newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        best_name = "hybrid"
        best_score = -1
        for row in reader:
            if row.get("question", "").strip() != "BEST_STRATEGY":
                continue
            for strategy_name in ["bm25", "vector", "hybrid", "rerank"]:
                value = row.get(f"{strategy_name}@5", "")
                if not value or ":" not in value:
                    continue
                _, score = value.split(":", 1)
                score_value = int(score.split("/", 1)[0])
                if score_value > best_score:
                    best_score = score_value
                    best_name = strategy_name
        return best_name


def main() -> None:
    if len(sys.argv) < 2:
        print("Usage: python baseline_cli.py 'your question'", file=sys.stderr)
        sys.exit(1)

    selected = pick_best_startup_strategy()
    print(f"Selected retrieval strategy: {selected}")

    question = sys.argv[1]
    answer = ask_llm(question)
    print(answer)


if __name__ == "__main__":
    main()
