#!/usr/bin/env python3
"""
Generate AI pod renders for pods.mtip.ai using OpenAI gpt-image-2.

Resilient version:
- Retries on HTTP 5xx and connection errors with exponential backoff.
- Exit code 0 even if some renders fail, so the commit step still runs
  and partial results get persisted.
- Failed entries are logged at the end so a re-run can pick them up
  (force=false will skip the ones that succeeded, force=true regenerates all).

Env:
  OPENAI_API_KEY  required
  FILTER          optional substring; only renders whose name contains this run
  FORCE           "true" to regenerate even if output exists
"""
import os, sys, json, base64, time, io
from pathlib import Path

import requests
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
RENDERS = ROOT / 'scripts' / 'renders.json'
ASSETS = ROOT / 'assets' / 'images'
API = 'https://api.openai.com/v1/images/edits'
MAX_ATTEMPTS = 4   # initial + 3 retries
RETRY_BACKOFF = [4, 10, 22]

def get_key():
    k = os.environ.get('OPENAI_API_KEY', '').strip()
    if not k:
        sys.exit('OPENAI_API_KEY is missing')
    return k

def should_run(entry, filt):
    if not filt:
        return True
    return filt.lower() in entry['name'].lower()

def already_exists(entry, force):
    out = ASSETS / f"{entry['name']}.jpg"
    return out.exists() and not force

class TransientError(Exception):
    pass

def call_api_once(entry, key):
    ref_path = ROOT / entry['ref']
    if not ref_path.exists():
        raise FileNotFoundError(f"reference {ref_path} missing")
    with open(ref_path, 'rb') as fh:
        files = {'image[]': (ref_path.name, fh.read(), 'image/jpeg')}
    data = {
        'model': entry.get('model', 'gpt-image-2'),
        'size': entry.get('size', '1536x1024'),
        'quality': entry.get('quality', 'high'),
        'n': '1',
        'prompt': entry['prompt'],
    }
    headers = {'Authorization': f'Bearer {key}'}
    try:
        r = requests.post(API, headers=headers, data=data, files=files, timeout=240)
    except (requests.ConnectionError, requests.Timeout) as e:
        raise TransientError(f"connection: {e}")
    if r.status_code in (429, 500, 502, 503, 504):
        raise TransientError(f"HTTP {r.status_code}: {r.text[:160]}")
    if not r.ok:
        try:
            err = r.json().get('error', {}).get('message', '')[:300]
        except Exception:
            err = r.text[:300]
        raise RuntimeError(f"HTTP {r.status_code}: {err}")
    payload = r.json()
    b64 = payload['data'][0].get('b64_json')
    if not b64:
        raise RuntimeError(f"no b64_json: keys={list(payload['data'][0].keys())}")
    return base64.b64decode(b64)

def call_api(entry, key):
    last = None
    for attempt in range(MAX_ATTEMPTS):
        try:
            return call_api_once(entry, key)
        except TransientError as e:
            last = e
            if attempt + 1 < MAX_ATTEMPTS:
                wait = RETRY_BACKOFF[min(attempt, len(RETRY_BACKOFF)-1)]
                print(f"   transient ({e}); retrying in {wait}s [{attempt+1}/{MAX_ATTEMPTS-1}]", flush=True)
                time.sleep(wait)
                continue
            raise RuntimeError(f"transient retries exhausted: {last}")
        except Exception:
            raise

def save_optimized_jpg(png_bytes, out_path, quality=85):
    img = Image.open(io.BytesIO(png_bytes)).convert('RGB')
    img.save(out_path, 'JPEG', quality=quality, optimize=True, progressive=True)

def main():
    key = get_key()
    filt = os.environ.get('FILTER', '').strip()
    force = os.environ.get('FORCE', '').lower() == 'true'

    if not RENDERS.exists():
        sys.exit(f'{RENDERS} missing')
    entries = json.loads(RENDERS.read_text())

    pending = [e for e in entries if should_run(e, filt) and not already_exists(e, force)]
    skipped = [e for e in entries if should_run(e, filt) and already_exists(e, force)]
    print(f"will run: {len(pending)}, skip: {len(skipped)}, total: {len(entries)}, filter='{filt}', force={force}")

    results = []
    for i, entry in enumerate(pending, 1):
        out = ASSETS / f"{entry['name']}.jpg"
        t0 = time.time()
        print(f"[{i}/{len(pending)}] {entry['name']}  size={entry.get('size')} q={entry.get('quality')}", flush=True)
        try:
            png = call_api(entry, key)
            save_optimized_jpg(png, out)
            dur = round(time.time() - t0, 1)
            kb = out.stat().st_size // 1024
            print(f"   -> ok {kb} KB in {dur}s", flush=True)
            results.append({'name': entry['name'], 'status': 'ok'})
        except Exception as e:
            print(f"   -> FAIL {e}", flush=True)
            results.append({'name': entry['name'], 'status': 'fail', 'error': str(e)[:200]})

    ok = sum(1 for r in results if r['status']=='ok')
    fail = sum(1 for r in results if r['status']=='fail')
    print(f"\nSummary: ok={ok} fail={fail} skipped={len(skipped)}")
    if fail:
        print("Failed entries (re-run with same filter to retry):")
        for r in results:
            if r['status']=='fail':
                print(f"  {r['name']}: {r['error']}")
    # exit 0 always — let the commit step persist whatever succeeded
    return 0

if __name__ == '__main__':
    sys.exit(main())
