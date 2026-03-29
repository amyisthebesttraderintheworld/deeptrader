"""
deepseek.py  ─  AI Strategy & Code Analysis
"""

import sys
import json
import requests
import os
from advanced_scanner.utils import get_env_key, strip_ansi, type_print, c
from advanced_scanner.config import BRIGHT_CYAN, BRIGHT_MAGENTA, BOLD, BRIGHT_WHITE, YELLOW, BRIGHT_RED

def report_to_deepseek(report_text=None, include_code=True):
    api_key = get_env_key("DEEPSEEK_API_KEY")
    if not api_key:
        print(c("\n[!] DEEPSEEK_API_KEY not found in .env or environment. Skipping AI analysis.", YELLOW))
        return

    code_context = ""
    if include_code:
        # Core files that define the strategy logic
        core_files = ["scoring.py", "backtest.py", "indicators.py"]
        for fp in core_files:
            try:
                if os.path.exists(fp):
                    with open(fp, "r") as f:
                        code_context += f"\n--- FILE: {fp} ---\n{f.read()}\n"
            except Exception:
                continue

    if report_text:
        clean_report = strip_ansi(report_text)
        prompt = f"""
You are an expert quantitative trading analyst and systems developer.
Analyze the following results AND the codebase logic.

1. RESULTS ANALYSIS: Briefly explain findings and suggest 2 improvements to the strategy.
2. CODE METACOMMENT: Identify architectural or mathematical flaws in the implementation.

STRICT RULES:
- TOTAL LIMIT: 1000 characters.
- NO MARKDOWN (no stars, no hashes, no dashes).
- Plain text only.

BACKTEST REPORT:
{clean_report}

CORE CODEBASE:
{code_context}
"""
    else:
        # If no report (e.g. estimate mode), just comment on the code
        prompt = f"""
You are an expert quantitative developer. Analyze this codebase for a crypto scanner/backtester.
Provide a metacomment on:
1. Logic/Mathematical improvements (alpha, risk management).
2. Performance/System optimization.

STRICT RULES:
- TOTAL LIMIT: 750 characters.
- NO MARKDOWN (no stars, no hashes, no dashes).
- Plain text only.

CORE CODEBASE:
{code_context}
"""
    
    try:
        resp = requests.post(
            "https://api.deepseek.com/v1/chat/completions",
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            json={
                "model": "deepseek-chat",
                "messages": [
                    {"role": "system", "content": "You are a concise quant analyst/developer. You never use markdown and stay under the character limit."},
                    {"role": "user", "content": prompt}
                ],
                "stream": True
            },
            timeout=90,
            stream=True
        )
        resp.raise_for_status()
        
        header_text = "◆ EXPERT ANALYSIS & CODE METACOMMENT" if report_text else "◆ CODE ARCHITECTURE & LOGIC METACOMMENT"
        color = BRIGHT_MAGENTA if report_text else BRIGHT_CYAN

        print("\n" + c("═" * 100, color))
        print(c(f"  {header_text}", BOLD + BRIGHT_WHITE))
        print(c("═" * 100, color) + "\n")

        for line in resp.iter_lines():
            if line:
                decoded_line = line.decode("utf-8")
                if decoded_line.startswith("data: "):
                    data_str = decoded_line[6:]
                    if data_str == "[DONE]":
                        break
                    try:
                        chunk = json.loads(data_str)
                        content = chunk["choices"][0]["delta"].get("content", "")
                        if content:
                            sys.stdout.write(content)
                            sys.stdout.flush()
                    except Exception:
                        continue
        
        print("\n\n" + c("═" * 100, color) + "\n")
            
    except Exception as e:
        print(c(f"\n[!] DeepSeek API error: {e}", BRIGHT_RED))