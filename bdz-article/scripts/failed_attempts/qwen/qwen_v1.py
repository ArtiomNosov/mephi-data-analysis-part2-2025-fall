import requests
import json
import time

OUTPUT_FILE = "leaderboards_v2.jsonl"

def is_leaderboard_repo(repo):
    # Проверяем название и описание
    text = f"{repo.get('id', '')} {repo.get('description', '')}".lower()
    if 'leaderboard' in text:
        return True

    # Проверяем README
    try:
        readme_url = f"https://huggingface.co/{repo['id']}/raw/main/README.md"
        resp = requests.get(readme_url, timeout=5)
        if resp.status_code == 200:
            readme_text = resp.text.lower()
            if 'leaderboard' in readme_text:
                return True
    except Exception as e:
        pass  # Игнорируем ошибки при загрузке README

    return False

def fetch_leaderboards():
    leaderboards = []
    page = 1
    per_page = 20
    total_needed = 100

    # Очищаем файл или создаём новый
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        pass  # просто создаём/очищаем

    print("🔍 Starting search for leaderboard datasets on Hugging Face...")

    while len(leaderboards) < total_needed:
        url = f"https://huggingface.co/api/datasets?sort=modified&limit={per_page}&p={page}"
        try:
            resp = requests.get(url, timeout=10)
        except Exception as e:
            print(f"⚠️ Request failed on page {page}: {e}")
            break

        if resp.status_code != 200:
            print(f"⚠️ API returned {resp.status_code} on page {page}. Stopping.")
            break

        repos = resp.json().get('datasets', [])
        if not repos:
            print("ℹ️ No more datasets returned. Stopping.")
            break

        print(f"📄 Processing page {page} ({len(repos)} datasets)...")

        for repo in repos:
            if is_leaderboard_repo(repo):
                result = {
                    "id": repo["id"],
                    "description": repo.get("description"),
                    "lastModified": repo.get("lastModified"),
                    "found_at": time.strftime("%Y-%m-%d %H:%M:%S")
                }
                leaderboards.append(repo["id"])

                # Логируем в консоль
                print(f"✅ Found #{len(leaderboards)}: {repo['id']}")

                # Записываем **сразу** в файл (реальное время)
                with open(OUTPUT_FILE, "a", encoding="utf-8") as f:
                    f.write(json.dumps(result, ensure_ascii=False) + "\n")

                if len(leaderboards) >= total_needed:
                    break

        page += 1

    print(f"\n🎯 Done! Found {len(leaderboards)} leaderboard datasets.")
    print(f"💾 Results saved line-by-line to '{OUTPUT_FILE}'")

if __name__ == "__main__":
    fetch_leaderboards()