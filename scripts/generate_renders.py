#!/usr/bin/env python3
"""
Generate AI pod renders for pods.mtip.ai using OpenAI gpt-image-2.

Reads scripts/renders.json (declarative list).
For each entry: skip if output JPG already exists (unless FORCE=true);
otherwise call /v1/images/edits with the listed reference image and prompt,
save the b64 result as an optimized progressive JPG.

Env:
  OPENAI_API_KEY  required
  FILTER          optional substring; only renders whose name contains this run
  FORCE           "true" to regenerate even if output exists
"""
import os, sys, json, base64, time, io, hashlib
from pathlib import Path

import requests
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
RENDERS = ROOT / 'scripts' / 'renders.json'
ASSETS = ROOT / 'assets' / 'images'
API = 'https://api.openai.com/v1/images/edits'

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

def call_api(entry, key):
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
    r = requests.post(API, headers=headers, data=data, files=files, timeout=240)
    if not r.ok:
        try:
            err = r.json().get('error', {}).get('message', '')[:300]
        except Exception:
            err = r.text[:300]
        raise RuntimeError(f"HTTP {r.status_code}: {err}")
    payload = r.json()
    b64 = payload['data'][0].get('b64_json')
    if not b64:
        raise RuntimeError(f"no b64_json in response: {list(payload['data'][0].keys())}")
    return base64.b64decode(b64)

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
        print(f"[{i}/{len(pending)}] {entry['name']}  ref={entry['ref']}  size={entry.get('size')} quality={entry.get('quality')}", flush=True)
        try:
            png = call_api(entry, key)
            save_optimized_jpg(png, out)
            dur = round(time.time() - t0, 1)
            kb = out.stat().st_size // 1024
            print(f"   -> ok {kb} KB in {dur}s", flush=True)
            results.append({'name': entry['name'], 'status': 'ok', 'kb': kb, 'duration_s': dur})
        except Exception as e:
            print(f"   -> FAIL {e}", flush=True)
            results.append({'name': entry['name'], 'status': 'fail', 'error': str(e)[:200]})

    # Print summary
    ok = sum(1 for r in results if r['status']=='ok')
    fail = sum(1 for r in results if r['status']=='fail')
    print(f"\nSummary: ok={ok} fail={fail} skipped={len(skipped)}")
    if fail:
        print("Failed entries:")
        for r in results:
            if r['status']=='fail':
                print(f"  {r['name']}: {r['error']}")
        sys.exit(1)

if __name__ == '__main__':
    main()
