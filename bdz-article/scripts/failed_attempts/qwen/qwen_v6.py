# find_spaces_leaderboard_v4.py
import requests
import json
import time

OUTPUT_FILE = "leaderboards_v4.jsonl"
KEYWORD = "leaderboard"
SEEN = set()

# Очистка файла
open(OUTPUT_FILE, "w").close()

print("🔍 Ищу Hugging Face Spaces с 'leaderboard' в ИМЕНИ репозитория...")

graphql_url = "https://huggingface.co/graphql"
query = """
query SearchSpaces($query: String!, $first: Int, $after: String) {
  spaces(query: $query, first: $first, after: $after) {
    edges {
      node {
        id
        name
        author {
          username
        }
      }
    }
    pageInfo {
      hasNextPage
      endCursor
    }
  }
}
"""

after = None
total = 0

while True:
    variables = {
        "query": KEYWORD,
        "first": 50,
        "after": after
    }

    try:
        resp = requests.post(
            graphql_url,
            json={"query": query, "variables": variables},
            timeout=10
        )
        if resp.status_code != 200:
            print(f"⚠️ HTTP {resp.status_code}")
            break

        data = resp.json()
        spaces = data.get("data", {}).get("spaces", {})
        edges = spaces.get("edges", [])
        page_info = spaces.get("pageInfo", {})

        if not edges:
            break

        for edge in edges:
            node = edge.get("node", {})
            author = node.get("author", {}).get("username", "").strip()
            name = node.get("name", "").strip()
            if not author or not name:
                continue

            # Полный ID: author/name
            repo_id = f"{author}/{name}"

            # Проверяем: именно имя (а не автор) содержит "leaderboard"
            if KEYWORD in name.lower() and repo_ids not in SEEN:
                SEEN.add(repo_id)
                url = f"https://huggingface.co/spaces/{repo_id}"
                record = {"id": repo_id, "url": url}
                with open(OUTPUT_FILE, "a", encoding="utf-8") as f:
                    f.write(json.dumps(record, ensure_ascii=False) + "\n")
                print(f"✅ {url}")

        if not page_info.get("hasNextPage"):
            break
        after = page_info.get("endCursor")
        time.sleep(0.5)

    except Exception as e:
        print(f"⚠️ Ошибка: {e}")
        break

print(f"\n🎯 Готово! Найдено {len(SEEN)} Spaces.")
print(f"📁 Результат: {OUTPUT_FILE}")