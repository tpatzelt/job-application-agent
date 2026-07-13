#!/usr/bin/env python3
"""
test_gemma_litellm.py

Minimal script that uses litellm's `completion` API to call a Gemma model.

Usage:
  1. Install litellm: `pip install litellm`
  2. Set your Google API key in the environment: `export GOOGLE_API_KEY=...`
  3. Run: `python test_gemma_litellm.py`

Options:
  - Provide a prompt on the command line to override the default prompt.
  - Use --model to set a different model string.
  - Use --reasoning to set reasoning_effort (none|low|medium|high).

Note: litellm provider configuration may vary. This script expects your
environment to supply the API key as `GOOGLE_API_KEY` (the same header used
by the curl example). If your litellm install expects a different env var,
point litellm to the key accordingly.
"""

import os
import sys
import json
from typing import Any
from argparse import ArgumentParser

try:
    from litellm import completion
except Exception as e:
    print("Error importing litellm. Install with: pip install litellm")
    raise


# Use a provider-prefixed model string so litellm can auto-detect the LLM provider.
# The repo's provider is "gemini", so prefix the model with "gemini/".
DEFAULT_MODEL = "gemini/gemma-3-27b-it"


def extract_text(r: Any) -> str:
    """Try to extract a human-readable text response from various shapes."""
    if r is None:
        return ""
    # If it's a dict-like object
    if isinstance(r, dict):
        # common top-level keys
        for k in ("output", "text", "content", "completion", "message"):
            v = r.get(k)
            if isinstance(v, str) and v.strip():
                return v
        # choices list
        choices = r.get("choices")
        if choices and isinstance(choices, (list, tuple)) and len(choices) > 0:
            first = choices[0]
            if isinstance(first, dict):
                for k in ("text", "message", "content"):
                    v = first.get(k)
                    if isinstance(v, str) and v.strip():
                        return v
    # Try attributes on objects returned by some SDKs
    for attr in ("output", "text", "content", "completion", "message"):
        if hasattr(r, attr):
            val = getattr(r, attr)
            if isinstance(val, str) and val.strip():
                return val
    # Fallback to string representation
    try:
        return str(r)
    except Exception:
        return ""


def main(argv=None):
    argv = argv or sys.argv[1:]
    p = ArgumentParser(description="Test Gemma via litellm completion API")
    p.add_argument(
        "prompt",
        nargs="*",
        help="Prompt to send (default: Explain how AI works in a few words)",
    )
    p.add_argument(
        "--model",
        default=os.environ.get("GEMMA_MODEL", DEFAULT_MODEL),
        help="Model string to use",
    )
    p.add_argument(
        "--reasoning",
        default="none",
        choices=("none", "low", "medium", "high"),
        help="reasoning_effort level",
    )
    args = p.parse_args(argv)

    api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        print("Error: set GOOGLE_API_KEY environment variable with your X-goog-api-key")
        sys.exit(2)

    prompt = (
        " ".join(args.prompt) if args.prompt else "Explain how AI works in a few words"
    )

    # Build message payload
    messages = [{"role": "user", "content": prompt}]

    print(f"Calling model={args.model!r} reasoning_effort={args.reasoning!r}")

    # Only send reasoning_effort when explicitly requested (not the default 'none').
    # Some providers/models reject this parameter and will raise an UnsupportedParamsError
    # from litellm. In that case, retry the request without the parameter.
    kwargs = {"model": args.model, "messages": messages}
    if args.reasoning and args.reasoning != "none":
        kwargs["reasoning_effort"] = args.reasoning

    try:
        resp = completion(**kwargs)
    except Exception as e:
        # Handle litellm providers that don't accept reasoning_effort by retrying
        # without the param. Check the exception text to avoid swallowing other
        # errors.
        msg = str(e)
        if "reasoning_effort" in msg and "UnsupportedParamsError" in msg:
            print("Provider rejected 'reasoning_effort' — retrying without it")
            kwargs.pop("reasoning_effort", None)
            resp = completion(**kwargs)
        else:
            print("Request failed:", e)
            raise

    # Print raw response (JSON friendly)
    print("\nRaw response:")
    try:
        print(json.dumps(resp, indent=2, ensure_ascii=False, default=str))
    except Exception:
        # Some SDK response objects don't serialize cleanly
        try:
            print(resp)
        except Exception:
            print("<unprintable response>")

    # Extract a likely textual answer for easy consumption
    answer = extract_text(resp)
    print("\nExtracted answer:\n")
    print(answer)


if __name__ == "__main__":
    main()
