"""
Firebase 에 쌓인 측정 기록(logs)과 내가 남긴 기록(labels)을 CSV 로 내려받는다.

사용법 (PC 또는 코랩):
    python tools/export_csv.py --url https://YOUR-PROJECT-default-rtdb.asia-southeast1.firebasedatabase.app
    python tools/export_csv.py --url ... --since 2026-11-11      # 날짜 이후만

읽기는 공개(read: true)로 열어 두었으므로 비밀번호가 필요 없다.
결과: data/raw/logs.csv, data/raw/labels.csv
"""
import argparse
import csv
import json
import os
import urllib.parse
import urllib.request
from datetime import datetime, timedelta, timezone

KST = timezone(timedelta(hours=9))
LOG_COLS = ["time", "ts", "up", "flag", "co2", "temp", "hum", "lux",
            "snd_mean", "snd_std", "snd_max", "snd_spikes", "reed", "state", "vent", "fan"]
LABEL_COLS = ["time", "ts", "focus", "drowsy", "sleeping", "note"]


def fetch(base, path, since_ts=None):
    q = ""
    if since_ts:
        q = "?" + urllib.parse.urlencode({"orderBy": '"$key"', "startAt": f'"{since_ts}"'})
    with urllib.request.urlopen(f"{base.rstrip('/')}/{path}.json{q}", timeout=60) as r:
        data = json.load(r) or {}
    return sorted(data.values(), key=lambda d: d.get("ts", 0))


def write_csv(rows, cols, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8-sig") as f:   # 엑셀에서도 한글이 깨지지 않게
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            r = dict(r)
            r["time"] = datetime.fromtimestamp(r["ts"], KST).strftime("%Y-%m-%d %H:%M:%S")
            w.writerow(r)
    print(f"{path}: {len(rows)}행")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", required=True, help="Firebase Realtime Database 주소")
    ap.add_argument("--since", help="YYYY-MM-DD (한국 시간) 이후만")
    ap.add_argument("--out", default="data/raw")
    a = ap.parse_args()
    since = None
    if a.since:
        since = int(datetime.strptime(a.since, "%Y-%m-%d").replace(tzinfo=KST).timestamp())
    write_csv(fetch(a.url, "logs", since), LOG_COLS, os.path.join(a.out, "logs.csv"))
    write_csv(fetch(a.url, "labels", since), LABEL_COLS, os.path.join(a.out, "labels.csv"))


if __name__ == "__main__":
    main()
