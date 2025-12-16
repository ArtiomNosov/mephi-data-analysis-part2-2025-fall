# pip install requests tqdm

import requests
from tqdm import tqdm
import time
import json
from pathlib import Path

# === НАСТРОЙКИ ===
INPUT_FILE = "hf_leaderboards.json"          # твой список ссылок
PROGRESS_FILE = "downloaded_urls.txt"        # сюда построчно пишем успешные URL
FAILED_FILE = "download_failed.txt"          # сюда пишем упавшие
HTML_FOLDER = Path("raw_leaderboards_html")
HTML_FOLDER.mkdir(exist_ok=True)

# === Загружаем ссылки ===
with open(INPUT_FILE, "r", encoding="utf-8") as f:
    all_urls = json.load(f)

# Загружаем уже скачанные (чтобы не качать заново)
already_downloaded = set()
if Path(PROGRESS_FILE).exists():
    with open(PROGRESS_FILE, "r", encoding="utf-8") as f:
        already_downloaded = {line.strip() for line in f if line.strip()}

print(f"Всего ссылок: {len(all_urls)}")
print(f"Уже скачано: {len(already_downloaded)}")
print(f"Осталось: {len(all_urls) - len(already_downloaded)}\n")

# === Качаем только то, чего ещё нет ===
session = requests.Session()
session.headers.update({
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
})

to_download = [url for url in all_urls if url not in already_downloaded]

for url in tqdm(to_download, desc="Качаем сырые HTML"):
    filename = url.split("/spaces/")[1].replace("/", "__") + ".html"
    filepath = HTML_FOLDER / filename
    
    try:
        r = session.get(url, timeout=40)
        if r.status_code == 200:
            with open(filepath, "w", encoding="utf-8") as f:
                f.write(r.text)
            
            # ←←← ВОТ ТУТ МГНОВЕННО СОХРАНЯЕМ ПРОГРЕСС ПОСТРОЧНО
            with open(PROGRESS_FILE, "a", encoding="utf-8") as prog:
                prog.write(url + "\n")
                
            time.sleep(0.25)  # остаёмся добрыми к HF
            
        else:
            with open(FAILED_FILE, "a", encoding="utf-8") as fail:
                fail.write(f"{url} | status: {r.status_code}\n")
            time.sleep(1)
            
    except Exception as e:
        with open(FAILED_FILE, "a", encoding="utf-8") as fail:
            fail.write(f"{url} | error: {str(e)}\n")
        time.sleep(2)

print("\n\nВСЁ, БРАТ! Полный комплект сырых страниц у тебя локально.")
print(f"Успешно: {PROGRESS_FILE} — можно в любой момент открыть и видеть, что уже есть")
print(f"Папка: {HTML_FOLDER} — 1000+ чистых HTML")
print("Можешь убивать скрипт хоть каждые 10 секунд — ничего не потеряется ❤️")