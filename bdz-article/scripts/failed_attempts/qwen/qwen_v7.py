# find_spaces_leaderboard_v4.py
import requests
import json
import time

OUTPUT_FILE = "leaderboards_v4.jsonl"
KEYWORD = "leaderboard"
SEEN = set()

open(OUTPUT_FILE, "w").close()

print("🔍 Ищу Hugging Face Spaces с 'leaderboard' в имени репозитория...")

# Официальный search API с фильтром по типу
search_url = "https://huggingface.co/api/search"
offset = 0
limit = 50
total_found = 0

while offset < 500:
    params = {
        "q": KEYWORD,
        "type": "space",        # ← только Spaces!
        "limit": limit,
        "offset": offset
    }

    try:
        resp = requests.get(search_url, params=params, timeout=10)
        if resp.status_code != 200:
            print(f"⚠️ Ошибка API: {resp.status_code}")
            break

        data = resp.json()
        results = data.get("spaces", [])  # именно "spaces" в ответе

        if not results:
            break

        for item in results:
            repo_id = item.get("id")
            if not repo_id or repo_id in SEEN:
                continue

            # Имя репозитория — после последнего "/"
            repo_name = repo_id.split("/")[-1]
            if KEYWORD in repo_name.lower():
                SEEN.add(repo_id)
                url = f"https://huggingface.co/spaces/{repo_id}"
                record = {"id": repo_id, "url": url}
                with open(OUTPUT_FILE, "a", encoding="utf-8") as f:
                    f.write(json.dumps(record, ensure_ascii=False) + "\n")
                print(f"✅ {url}")

        print(f"  → offset {offset}: найдено {len(results)} Spaces")
        offset += limit
        time.sleep(0.3)

        # Если меньше, чем limit — конец
        if len(results) < limit:
            break

    except Exception as e:
        print(f"⚠️ Исключение: {e}")
        break

print(f"\n🎯 Готово! Найдено {len(SEEN)} Spaces с '{KEYWORD}' в имени.")
print(f"📁 Результат: {OUTPUT_FILE}")