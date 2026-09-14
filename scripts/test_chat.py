"""Streaming chat-completion test against the OpenAI-compatible gateway.

Talks to the Phase 2 gateway at http://localhost:8080/v1, which serves every
registered alias at one fixed port (no per-model ports). Start the control
plane with `make serve` (or `vibe serve`) and load a model first with
`vibe load <alias>`.
"""

import argparse
import json
import sys
import urllib.error
import urllib.request

# The gateway serves every registry alias at one fixed port (ADR-4).
GATEWAY_URL = "http://localhost:8080/v1"
DEFAULT_MODEL = "qwen2.5-coder-7b"

USER_PROMPT = (
    "Write a bullet-point resume entry describing setting up high-performance "
    "local LLM inference engines using vLLM and Docker containers on Linux."
)


def active_model():
    """Return the id of the gateway's active model, or None if there isn't one."""
    with urllib.request.urlopen(f"{GATEWAY_URL}/models", timeout=10) as response:
        body = json.loads(response.read().decode("utf-8"))
    for entry in body.get("data") or []:
        if entry.get("vibe_active"):
            return entry.get("id")
    return None


def stream_chat(model):
    """POST a streaming chat completion and print tokens as they arrive."""
    payload = json.dumps(
        {
            "model": model,
            "messages": [{"role": "user", "content": USER_PROMPT}],
            "temperature": 0.7,
            "stream": True,
        }
    ).encode("utf-8")

    request = urllib.request.Request(
        f"{GATEWAY_URL}/chat/completions",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    with urllib.request.urlopen(request, timeout=30) as response:
        for raw_line in response:
            line = raw_line.decode("utf-8").strip()
            if not line.startswith("data:"):
                continue
            data = line[len("data:"):].strip()
            if data == "[DONE]":
                break
            try:
                chunk = json.loads(data)
            except json.JSONDecodeError:
                continue
            choices = chunk.get("choices") or []
            content = ""
            if choices and choices[0].get("delta"):
                content = choices[0]["delta"].get("content") or ""
            if content:
                print(content, end="", flush=True)
    print("\n")


def report_http_error(exc):
    """Print the gateway's JSON error message (if any) plus a status hint."""
    hints = {
        503: "model not loaded — run 'vibe load <alias>' or enable VIBE_SWAP_ON_REQUEST",
        404: "unknown model alias",
    }
    message = None
    try:
        body = json.loads(exc.read().decode("utf-8"))
        if isinstance(body, dict):
            error = body.get("error")
            if isinstance(error, dict):
                message = error.get("message")
    except (json.JSONDecodeError, UnicodeDecodeError):
        pass
    if message:
        print(f"ERROR ({exc.code}): {message}")
    else:
        print(f"ERROR: gateway returned HTTP {exc.code}")
    hint = hints.get(exc.code)
    if hint:
        print(f"hint: {hint}")
    return 1


def main(argv=None):
    parser = argparse.ArgumentParser(
        description=(
            "Stream a test chat through the OpenAI gateway "
            f"({GATEWAY_URL}); run 'vibe serve' first."
        )
    )
    parser.add_argument(
        "--model",
        default=None,
        help=(
            "model alias to request (default: the gateway's active model, "
            f"falling back to {DEFAULT_MODEL})"
        ),
    )
    args = parser.parse_args(argv)

    model = args.model
    if model is None:
        try:
            model = active_model()
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
            print(f"ERROR: could not query /v1/models ({exc}). Is 'vibe serve' running?")
            return 1
        if model is None:
            model = DEFAULT_MODEL
            print(
                f"no active model found; defaulting to {DEFAULT_MODEL} "
                "(start one with 'vibe load <alias>')"
            )
        else:
            print(f"Using active model: {model}")

    print(f"Connected to: model '{model}' via the gateway\n")
    try:
        stream_chat(model)
    except urllib.error.HTTPError as exc:
        return report_http_error(exc)
    except (urllib.error.URLError, TimeoutError) as exc:
        print(
            f"ERROR: no response from the gateway ({exc}). "
            "Run 'make serve' and 'vibe load <alias>' first."
        )
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
