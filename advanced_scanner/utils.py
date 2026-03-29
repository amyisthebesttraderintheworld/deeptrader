"""
utils.py  ─  Helper Functions & Networking
"""

import os
import re
import sys
import time
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from advanced_scanner.config import TIMEOUT, TYPE_DELAY, RESET

def get_env_key(key_name):
    if os.path.exists(".env"):
        with open(".env", "r") as f:
            for line in f:
                if "=" in line and not line.startswith("#"):
                    parts = line.split("=", 1)
                    if parts[0].strip() == key_name:
                        val = parts[1].strip()
                        if (val.startswith('"') and val.endswith('"')) or (val.startswith("'") and val.endswith("'")):
                            return val[1:-1]
                        return val
    return os.environ.get(key_name)

def strip_ansi(text):
    return re.sub(r"\x1b\[[0-9;?]*[a-zA-Z]", "", text)

def c(text, color=""):
    return f"{color}{text}{RESET}" if color else str(text)

def type_print(text, delay=TYPE_DELAY):
    for ch in text:
        sys.stdout.write(ch); sys.stdout.flush(); time.sleep(delay)
    print()

def build_session():
    s = requests.Session()
    retry = Retry(total=3, backoff_factor=0.5, status_forcelist=(429,500,502,503,504))
    a = HTTPAdapter(max_retries=retry)
    s.mount("http://", a); s.mount("https://", a)
    return s

from datetime import datetime
import json

def log_execution(module_name, args, summary=None):
    """Logs the execution details to execution_history.log."""
    log_file = "execution_history.log"
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # Convert Namespace to dict if it's an argparse result
    arg_dict = vars(args) if hasattr(args, "__dict__") else str(args)
    
    log_entry = {
        "timestamp": timestamp,
        "module": module_name,
        "arguments": arg_dict,
        "summary": summary if summary else "No summary provided"
    }
    
    try:
        with open(log_file, "a") as f:
            f.write(json.dumps(log_entry) + "\n")
    except Exception as e:
        sys.stderr.write(c(f"\n[!] Failed to write to execution_history.log: {e}\n", "\033[91m"))

SHOW_API_HELP = True

def explain_phemex_error(status_code, response_data=None):
    global SHOW_API_HELP
    msg = f"API Error (Status {status_code}): "
    if status_code == 429:
        msg += "Rate limit exceeded. Phemex allows a limited number of requests per minute. Slow down your requests or use a higher 'MAX_WORKERS' with caution."
    elif status_code == 403:
        msg += "Forbidden. This could be due to IP blocking, invalid API keys, or restricted access to certain endpoints."
    elif status_code == 500:
        msg += "Phemex internal server error. This is usually temporary. Try again later."
    elif status_code == 503:
        msg += "Service unavailable. Phemex might be under maintenance or experiencing high load."
    else:
        msg += "Unexpected network error."
    
    if response_data and isinstance(response_data, dict):
        phemex_code = response_data.get("code") or response_data.get("error", {}).get("code")
        phemex_msg = response_data.get("msg") or response_data.get("error", {}).get("message")
        if phemex_code:
            msg += f" | Phemex Code: {phemex_code}"
        if phemex_msg:
            msg += f" | Phemex Message: {phemex_msg}"
    
    if SHOW_API_HELP:
        msg += "\n\n" + c("── HOW PHEMEX API WORKS ────────────────────────────────────────────────────────", "\033[93m") + "\n"
        msg += "  • Phemex uses a REST API for public data (market data) and private data (trading).\n"
        msg += "  • Rate limits are enforced per IP. If you see 429, reduce MAX_WORKERS or add delays.\n"
        msg += "  • Most endpoints return a 'code' field; 0 means success. Non-zero indicates an error.\n"
        msg += "  • Use standard HTTPS. Connection issues are often due to local network or regional blocks.\n"
        msg += c("──────────────────────────────────────────────────────────────────────────────", "\033[93m")
        SHOW_API_HELP = False
        
    return msg

SESSION = build_session()

def get_json(url, params=None):
    try:
        r = SESSION.get(url, params=params, timeout=TIMEOUT)
        if r.status_code != 200:
            error_msg = explain_phemex_error(r.status_code, r.json() if r.text else None)
            sys.stderr.write(c(f"\n[!] {error_msg}\n", "\033[91m"))
            r.raise_for_status()
        
        data = r.json()
        code = data.get("code")
        if code is not None and code != 0:
            error_msg = explain_phemex_error(200, data)
            sys.stderr.write(c(f"\n[!] {error_msg}\n", "\033[91m"))
            
        return data
    except Exception:
        return {}
