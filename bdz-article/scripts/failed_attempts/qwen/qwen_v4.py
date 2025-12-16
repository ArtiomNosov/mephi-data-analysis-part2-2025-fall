import requests
import json
import time

OUTPUT_FILE = "leaderboards_v4.jsonl"
KEYWORD = "leaderboard"
SEEN = set()

# Очистим файл
open(OUTPUT_FILE, "w").close()

print("🔍 Ищу любые страницы Hugging Face с 'leaderboard' в названии...")

# Hugging Face позволяет искать глобально через /api/repos (но его нет),
# поэтому проверим и модели, и датасеты — но без разделения.
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
                if KEYWORD in repo_id.lower():
                    SEEN.add(repo_id)
                    link = f"https://huggingface.co/{repo_id}"
                    record = {"id": repo_id, "url": link}
                    with open(OUTPUT_FILE, "a", encoding="utf-8") as f:
                        f.write(json.dumps(record, ensure_ascii=False) + "\n")
                    print(f"✅ {link}")
                    total += 1
            offset += 50
            time.sleep(0.2)
        except Exception:
            break

print(f"\n🎯 Готово! Найдено {len(SEEN)} уникальных страниц.")
print(f"📁 Результат: {OUTPUT_FILE}")