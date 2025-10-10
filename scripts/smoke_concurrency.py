import os
import time
import json
import concurrent.futures as cf
import requests


URL = os.getenv('API_URL', 'http://localhost:8080/predict-prematch')
TOKEN = os.getenv('API_TOKEN')

payload = {
    'season': 2025,
    'venue': 'MA Chidambaram Stadium, Chepauk',
    'city': 'Chennai',
    'stage': 'league',
    'match_number': 10,
    'opponent': 'Mumbai Indians'
}


def call_once(i):
    headers = {'Content-Type': 'application/json'}
    if TOKEN:
        headers['Authorization'] = f'Bearer {TOKEN}'
    t0 = time.perf_counter()
    r = requests.post(URL, headers=headers, data=json.dumps(payload), timeout=20)
    dt = (time.perf_counter() - t0) * 1000.0
    return i, r.status_code, dt


def main():
    concurrency = int(os.getenv('CONCURRENCY', '10'))
    with cf.ThreadPoolExecutor(max_workers=concurrency) as ex:
        futs = [ex.submit(call_once, i) for i in range(concurrency)]
        results = [f.result() for f in futs]
    ok = sum(1 for _, s, _ in results if s == 200)
    p95 = sorted(dt for *_, dt in results)[int(0.95 * len(results)) - 1]
    print(f"OK {ok}/{len(results)} | p95 latency ~ {p95:.2f} ms")


if __name__ == '__main__':
    main()

