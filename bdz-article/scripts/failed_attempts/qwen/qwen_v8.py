# find_spaces_leaderboard_v4.py
import requests
import json
import time
import urllib.parse

OUTPUT_FILE = "leaderboards_v4.jsonl"
KEYWORD = "leaderboard"
SEEN = set()

open(OUTPUT_FILE, "w").close()

print("🔍 Ищу Hugging Face Spaces с 'leaderboard' в имени репозитория...")

# Публичный full-text search (не /api/search!)
base_url = "https://huggingface.co/search/full-text"
offset = 0
limit = 20  # максимум 20 за раз на этом эндпоинте

while offset < 200:  # лимит на 10 страниц = 200 результатов
    params = {
        "q": KEYWORD,
        "type": "space",
        "limit": limit,
        "offset": offset
    }
    url = base_url + "?" + urllib.parse.urlencode(params)

    try:
        # Важно: имитируем браузер, иначе могут вернуть CAPTCHA или 403
        headers = {
            "User-Agent": "Mozilla/5.0 (compatible; LeaderboardBot/1.0; +https://example.com)"
        }
        resp = requests.get(url, headers=headers, timeout=10)

        if resp.status_code != 200:
            print(f"⚠️ Ошибка: {resp.status_code}")
            break

        data = resp.json()
        results = data.get("results", [])

        if not results:
            break

        new_found = 0
        for item in results:
            repo_id = item.get("url")
            if not repo_id or not repo_id.startswith("/spaces/"):
                continue

            # repo_id looks like: "/spaces/open-llm-leaderboard/open_llm_leaderboard"
            clean_id = repo_id[len("/spaces/"):]  # "open-llm-leaderboard/open_llm_leaderboard"

            if clean_id in SEEN:
                continue

            repo_name = clean_id.split("/")[-1]
            if KEYWORD in repo_name.lower():
                SEEN.add(clean_id)
                full_url = "https://huggingface.co" + repo_id
                record = {"id": clean_id, "url": full_url}
                with open(OUTPUT_FILE, "a", encoding="utf-8") as f:
                    f.write(json.dumps(record, ensure_ascii=False) + "\n")
                print(f"✅ {full_url}")
                new_found += 1

        print(f"  → offset {offset}: {len(results)} получено, {new_found} подошло")
        offset += limit
        time.sleep(0.5)

        if len(results) < limit:
            break

    except Exception as e:
        print(f"⚠️ Ошибка: {e}")
        break

print(f"\n🎯 Готово! Найдено {len(SEEN)} Spaces с '{KEYWORD}' в имени.")
print(f"📁 Результат: {OUTPUT_FILE}")