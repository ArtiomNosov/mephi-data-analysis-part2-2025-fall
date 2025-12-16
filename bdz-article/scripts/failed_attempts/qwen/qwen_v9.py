# safe_leaderboard_spaces_v4.py
import requests
import json

OUTPUT_FILE = "leaderboards_v4.jsonl"

# Известные или вероятные пары (author, repo_name), где repo_name содержит "leaderboard"
candidates = [
    ("open-llm-leaderboard", "open_llm_leaderboard"),
    ("llm-blender", "llm-blender-leaderboard"),
    ("mlabonne", "llm-leaderboard"),
    ("huggingface-projects", "llm-leaderboard"),
    ("open-rl-leaderboard", "atari-leaderboard"),
    ("open-rl-leaderboard", "mujoco-leaderboard"),
    ("Bingsu", "korean-llm-leaderboard"),
    ("LLM360", "LLM360-Leaderboard"),
    ("FlagOpen", "FlagEmbedding-Leaderboard"),
    ("embeddings-benchmark", "embeddings-leaderboard"),
    ("arena", "arena-leaderboard"),
    ("lmarena-ai", "lmarena-leaderboard"),
]

results = []

print("🔍 Проверяю известные Hugging Face Spaces с 'leaderboard' в имени...")

for author, repo in candidates:
    url = f"https://huggingface.co/spaces/{author}/{repo}"
    try:
        # Проверяем существование через HEAD (быстро и не грузит страницу)
        resp = requests.head(url, timeout=5)
        if resp.status_code == 200:
            record = {"id": f"{author}/{repo}", "url": url}
            results.append(record)
            print(f"✅ {url}")
        else:
            print(f"❌ {url} (HTTP {resp.status_code})")
    except Exception as e:
        print(f"⚠️ {url} — ошибка: {e}")

# Сохраняем всё
with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    for rec in results:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")

print(f"\n🎯 Найдено {len(results)} рабочих Spaces.")
print(f"📁 Сохранено в: {OUTPUT_FILE}")