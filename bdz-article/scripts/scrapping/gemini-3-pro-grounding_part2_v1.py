# pip install huggingface_hub tqdm pandas

import json
import os
import re
from pathlib import Path
from tqdm import tqdm
from huggingface_hub import HfApi, hf_hub_download
from huggingface_hub.utils import EntryNotFoundError, RepositoryNotFoundError

# === НАСТРОЙКИ ===
INPUT_FILE = "hf_leaderboards.json"  # Твой файл со списком ссылок
OUTPUT_DIR = Path("leaderboards_data")
OUTPUT_DIR.mkdir(exist_ok=True)

# Ищем файлы, похожие на таблицы данных
PRIORITY_FILES = [
    re.compile(r"results\.csv", re.I),
    re.compile(r"leaderboard\.csv", re.I),
    re.compile(r"data\.csv", re.I),
    re.compile(r".*\.csv$", re.I),     # Любой CSV
    re.compile(r".*\.json$", re.I),    # Любой JSON (часто там конфиги, но бывают и данные)
    re.compile(r".*\.parquet$", re.I)  # Parquet (сжатые данные)
]

def get_repo_id_from_url(url):
    # Превращаем https://huggingface.co/spaces/Author/Name -> Author/Name
    if "/spaces/" not in url:
        return None
    return url.split("/spaces/")[1]

# === ЗАГРУЗКА СПИСКА ===
with open(INPUT_FILE, "r", encoding="utf-8") as f:
    urls = json.load(f)

api = HfApi()

print(f"🔄 Начинаю проверку данных для {len(urls)} лидербордов...")

stats = {"downloaded": 0, "no_data_found": 0, "errors": 0}

for url in tqdm(urls):
    repo_id = get_repo_id_from_url(url)
    if not repo_id:
        continue

    # Папка для конкретного лидерборда
    safe_name = repo_id.replace("/", "__")
    local_folder = OUTPUT_DIR / safe_name
    
    # Если мы уже что-то скачали оттуда — пропускаем (или убери это условие, если хочешь обновлять)
    if local_folder.exists() and any(local_folder.iterdir()):
        continue

    try:
        # 1. Получаем список всех файлов в Space
        # repo_type="space" — это критически важно!
        files = api.list_repo_files(repo_id=repo_id, repo_type="space")
        
        target_file = None
        
        # 2. Ищем самый подходящий файл по приоритету
        for pattern in PRIORITY_FILES:
            matches = [f for f in files if pattern.match(f)]
            if matches:
                # Берем первый подходящий (обычно results.csv или leaderboard.csv)
                target_file = matches[0]
                break
        
        if target_file:
            # 3. Скачиваем файл
            local_folder.mkdir(exist_ok=True)
            downloaded_path = hf_hub_download(
                repo_id=repo_id,
                filename=target_file,
                repo_type="space",
                local_dir=local_folder,
                local_dir_use_symlinks=False
            )
            stats["downloaded"] += 1
        else:
            # Файлов данных явно не видно (возможно данные генерятся на лету)
            stats["no_data_found"] += 1

    except Exception as e:
        # Бывает, что Space удален или закрыт
        # print(f"Ошибка с {repo_id}: {e}")
        stats["errors"] += 1

print("\n=== ОТЧЕТ ===")
print(f"✅ Скачано файлов данных: {stats['downloaded']}")
print(f"⚠️ Не найдено явных файлов (CSV/JSON): {stats['no_data_found']}")
print(f"❌ Ошибки доступа/удалены: {stats['errors']}")
print(f"📁 Все файлы лежат в папке: {OUTPUT_DIR}")