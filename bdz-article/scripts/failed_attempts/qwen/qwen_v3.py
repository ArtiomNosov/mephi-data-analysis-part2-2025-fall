import requests
import json
import time

OUTPUT_FILE = "leaderboards_v2.jsonl"

def is_leaderboard_repo(repo):
    text = f"{repo.get('id', '')} {repo.get('description', '')}".lower()
    if 'leaderboard' in text:
        return True

    try:
        readme_url = f"https://huggingface.co/{repo['id']}/raw/main/README.md"
        resp = requests.get(readme_url, timeout=5)
        if resp.status_code == 200:
            if 'leaderboard' in resp.text.lower():
                return True
    except Exception:
        pass

    return False

def fetch_leaderboards():
    seen_ids = set()
    leaderboards = []
    offset = 0
    limit = 50
    total_needed = 100

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        pass

    print("🔍 Starting search for leaderboard datasets on Hugging Face (sorted by lastModified)...")

    while len(leaderboards) < total_needed:
        # ВАЖНО: добавляем sort для стабильной пагинации
        url = f"https://huggingface.co/api/datasets?limit={limit}&offset={offset}&sort=lastModified"
        try:
            resp = requests.get(url, timeout=10)
        except Exception as e:
            print(f"⚠️ Request failed at offset {offset}: {e}")
            break

        if resp.status_code != 200:
            print(f"⚠️ API returned {resp.status_code} at offset {offset}. Response: {resp.text}")
            break

        repos = resp.json()
        if not repos:
            print("ℹ️ No more datasets. Stopping.")
            break

        new_found = 0
        for repo in repos:
            repo_id = repo.get("id")
            if not repo_id:
                continue

            # Пропускаем дубли
            if repo_id in seen_ids:
                continue
            seen_ids.add(repo_id)

            if is_leaderboard_repo(repo):
                result = {
                    "id": repo_id,
                    "description": repo.get("description"),
                    "lastModified": repo.get("lastModified"),
                    "found_at": time.strftime("%Y-%m-%d %H:%M:%S")
                }
                leaderboards.append(repo_id)
                print(f"✅ Found #{len(leaderboards)}: {repo_id}")

                with open(OUTPUT_FILE, "a", encoding="utf-8") as f:
                    f.write(json.dumps(result, ensure_ascii=False) + "\n")

                if len(leaderboards) >= total_needed:
                    break
                new_found += 1

        print(f"📄 Offset {offset}: {len(repos)} received, {new_found} new checked, total found: {len(leaderboards)}")

        # Если новых репозиториев не было — вероятно, достигли конца
        if len(repos) < limit:
            print("ℹ️ Reached end of dataset list.")
            break

        offset += limit
        time.sleep(0.3)  # уважаем rate limits

    print(f"\n🎯 Done! Found {len(leaderboards)} unique leaderboard datasets.")
    print(f"💾 Saved to '{OUTPUT_FILE}'")

if __name__ == "__main__":
    fetch_leaderboards()