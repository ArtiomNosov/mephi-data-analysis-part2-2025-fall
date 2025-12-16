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
    leaderboards = []
    offset = 0
    limit = 50  # максимальный лимит на страницу — 50
    total_needed = 100

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        pass  # очистка файла

    print("🔍 Starting search for leaderboard datasets on Hugging Face...")

    while len(leaderboards) < total_needed:
        url = f"https://huggingface.co/api/datasets?limit={limit}&offset={offset}"
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
            print("ℹ️ No more datasets returned. Stopping.")
            break

        print(f"📄 Processing offset {offset} ({len(repos)} datasets)...")

        for repo in repos:
            if is_leaderboard_repo(repo):
                result = {
                    "id": repo["id"],
                    "description": repo.get("description"),
                    "lastModified": repo.get("lastModified"),
                    "found_at": time.strftime("%Y-%m-%d %H:%M:%S")
                }
                leaderboards.append(repo["id"])
                print(f"✅ Found #{len(leaderboards)}: {repo['id']}")

                with open(OUTPUT_FILE, "a", encoding="utf-8") as f:
                    f.write(json.dumps(result, ensure_ascii=False) + "\n")

                if len(leaderboards) >= total_needed:
                    break

        offset += limit

        # Уважаем rate limits
        time.sleep(0.2)

    print(f"\n🎯 Done! Found {len(leaderboards)} leaderboard datasets.")
    print(f"💾 Results saved to '{OUTPUT_FILE}'")

if __name__ == "__main__":
    fetch_leaderboards()