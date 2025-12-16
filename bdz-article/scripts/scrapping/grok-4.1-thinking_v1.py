# pip install huggingface_hub   # одна зависимость

import re
import json
from huggingface_hub import HfApi

# Создаём клиент API
api = HfApi()

# Регулярка для поиска "leaderboard" в имени спейса (вторая часть URL)
pattern = re.compile(r"leaderboard", re.IGNORECASE)

links = set()  # set → автоматически уникальные

print("Начинаю сбор лидербордов...")

# search="leaderboard" возвращает все спейсы, где это слово есть в имени или описании
# мы дополнительно фильтруем только по имени спейса — строго как ты просил
for space in api.list_spaces(search="leaderboard"):
    if not space.id or "/" not in space.id:
        continue
    author, name = space.id.split("/", 1)
    if pattern.search(name):                     # ← именно здесь проверка *leaderboard*
        link = f"https://huggingface.co/spaces/{author}/{name}"
        links.add(link)

# Сортируем для красоты и стабильности
links = sorted(links)

print(f"Найдено {len(links)} уникальных лидербордов")

# Вариант 1: обычный JSON-массив (самый удобный для чтения)
with open("hf_leaderboards.json", "w", encoding="utf-8") as f:
    json.dump(links, f, indent=2, ensure_ascii=False)

# Вариант 2: JSONL (по одной ссылке на строку) — ты просил "построчно в json"
with open("hf_leaderboards.jsonl", "w", encoding="utf-8") as f:
    for link in links:
        f.write(json.dumps(link, ensure_ascii=False) + "\n")

print("Готово! Файлы: hf_leaderboards.json и hf_leaderboards.jsonl")