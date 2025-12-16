import requests
import json
import time

OUTPUT_FILE = "leaderboards_v4.jsonl"
KEYWORD = "leaderboard"
SEEN = set()

# Очистим файл
open(OUTPUT_FILE, "w").close()

print("🔍 Ищу репозитории, где ИМЯ (после последнего '/') содержит 'leaderboard'...")

bases = [
    "https://huggingface.co/api/models",
    "https://huggingface.co/api/datasets"
]

total = 0

for base in bases:
    offset = 0
    while offset < 1000:
        url = f"{base}?limit=50&offset={offset}&search={KEYWORD}"
        try:
            resp = requests.get(url, timeout=10)
            if resp.status_code != 200:
                break
            items = resp.json()
            if not items:
                break

            for item in items:
                repo_id = item.get("id")
                if not repo_id or repo_id in SEEN:
                    continue

                # Берём только имя репозитория (после "/")
                repo_name = repo_id.split("/")[-1]
                if KEYWORD in repo_name.lower():
                    SEEN.add(repo_id)
                    link = f"https://huggingface.co/{repo_id}"
                    record = {"id": repo_id, "url": link}
                    with open(OUTPUT_FILE, "a", encoding="utf-8") as f:
                        f.write(json.dumps(record, ensure_ascii=False) + "\n")
                    print(f"✅ {link}")
                    total += 1

            offset += 50
            time.sleep(0.2)
        except Exception as e:
            continue

print(f"\n🎯 Готово! Найдено {len(SEEN)} репозиториев с '{KEYWORD}' в имени.")
print(f"📁 Результат: {OUTPUT_FILE}")