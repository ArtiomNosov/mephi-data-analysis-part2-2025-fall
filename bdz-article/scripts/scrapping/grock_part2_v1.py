# pip install huggingface_hub beautifulsoup4 tqdm

import json
import time
import re
from pathlib import Path
from tqdm import tqdm
from huggingface_hub import HfApi
from bs4 import BeautifulSoup
import requests

api = HfApi()
session = requests.Session()
session.headers.update({"User-Agent": "HF-Leaderboard-Parser/2.0"})

# Загружаем список ссылок (любой из двух файлов, что я тебе дал)
with open("hf_leaderboards.json", "r", encoding="utf-8") as f:
    leaderboard_urls = json.load(f)

results = []
failed_urls = []

print(f"Начинаю парсинг {len(leaderboard_urls)} лидербордов...")

for url in tqdm(leaderboard_urls, desc="Парсим лидерборды"):
    try:
        # Некоторые спейсы — просто картинки или демо, а не настоящий лидерборд
        # Делаем быстрый запрос и сразу проверяем, есть ли таблица
        r = session.get(url, timeout=20)
        if r.status_code != 200:
            failed_urls.append({"url": url, "error": "status_code_" + str(r.status_code)})
            continue

        soup = BeautifulSoup(r.text, "html.parser")

        # Главный признак настоящего лидерборда — таблица с классом "leaderboard-table"
        table = soup.find("table", {"class": "leaderboard-table"}) or soup.find("div", {"data-target": "leaderboard-table"})
        
        if not table:
            # Дополнительная проверка: может быть React-таблица с data-testid
            if not soup.find("div", {"data-testid": re.compile("leaderboard", re.I)}):
                failed_urls.append({"url": url, "error": "no_leaderboard_table"})
                continue

        # Извлекаем базовую инфу
        title = soup.find("h1") or soup.find("title")
        title_text = title.get_text(strip=True) if title else "Unknown"

        author_tag = soup.find("a", href=re.compile(r"^/[^/]+$"))
        author = author_tag.get_text(strip=True) if author_tag else url.split("/spaces/")[1].split("/")[0]

        # Ищем все строки моделей
        rows = soup.find_all("tr", {"class": re.compile(r"leaderboard-row|model-row", re.I)})
        
        models = []
        metrics = set()

        for row in rows:
            # Пропускаем заголовки
            if row.find("th"):
                # Заодно собираем названия метрик из заголовка
                headers = row.find_all("th")
                for h in headers:
                    text = h.get_text(strip=True)
                    if text and text not in ["#", "Model", "Rank", ""]:
                        metrics.add(text)
                continue

            cells = row.find_all("td")
            if len(cells) < 3:
                continue

            model_cell = cells[1]  # обычно вторая колонка — название модели
            model_link_tag = model_cell.find("a")
            
            model_name = model_cell.get_text(strip=True)
            model_repo = None
            model_revision = None

            if model_link_tag and "/models/" in model_link_tag.get("href", ""):
                href = model_link_tag["href"]
                if href.startswith("http"):
                    model_repo = href.split("huggingface.co/")[1]
                else:
                    model_repo = href.lstrip("/")
                
                # Ищем ревизию (часто в query-параметрах или в отдельном теге)
                rev_tag = model_cell.find("span", {"class": re.compile("revision|commit", re.I)})
                if rev_tag:
                    model_revision = rev_tag.get_text(strip=True)

            # Значения метрик
            values = {}
            for i, cell in enumerate(cells[2:], start=2):
                metric_val = cell.get_text(strip=True)
                if metric_val and metric_val not in ["-", "N/A", ""]:
                    # Попробуем взять название метрики из заголовка таблицы, если есть
                    # Но проще просто по порядку — обычно порядок одинаковый
                    # Поэтому собираем всё подряд, а метрики потом отдельно
                    values[f"col_{i}"] = metric_val

            # Если есть явные названия метрик — лучше их использовать
            metric_cells = cells[2:]
            if len(metric_cells) == len(metrics):
                for metric_name, cell in zip(sorted(metrics), metric_cells):
                    val = cell.get_text(strip=True)
                    if val and val not in ["-", "N/A", ""]:
                        values[metric_name] = val

            # Флаги (Average ▲, Rigorous, etc.)
            flags = []
            flag_spans = row.find_all("span", {"class": re.compile("flag|badge|pill", re.I)})
            for f in flag_spans:
                flag_text = f.get_text(strip=True)
                if flag_text:
                    flags.append(flag_text)

            models.append({
                "rank": cells[0].get_text(strip=True),
                "model_name": model_name,
                "model_repo": model_repo,
                "model_revision": model_revision,
                "values": values,
                "flags": flags
            })

        # Определяем тип задачи (если указано)
        task_tag = soup.find("span", text=re.compile(r"Task:", re.I))
        task = task_tag.find_next_sibling(string=True).strip() if task_tag else None

        leaderboard_data = {
            "url": url,
            "title": title_text,
            "author": author,
            "task": task,
            "models_count": len(models),
            "metrics": sorted(list(metrics)),
            "last_updated": soup.find("time", {"class": "timeago"})["datetime"] if soup.find("time", {"class": "timeago"}) else None,
            "models": models[:100]  # ограничиваем до топ-100, если очень много
        }

        results.append(leaderboard_data)
        time.sleep(0.3)  # очень вежливо к HF

    except Exception as e:
        failed_urls.append({"url": url, "error": str(e)})
        continue

# Сохраняем всё
with open("hf_leaderboards_full_data.json", "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2, ensure_ascii=False)

with open("hf_leaderboards_full_data.jsonl", "w", encoding="utf-8") as f:
    for item in results:
        f.write(json.dumps(item, ensure_ascii=False) + "\n")

with open("failed_leaderboards.json", "w", encoding="utf-8") as f:
    json.dump(failed_urls, f, indent=2, ensure_ascii=False)

print(f"\nГОТОВО! Успешно спаршено: {len(results)} лидербордов")
print(f"Не удалось (не являются таблицами): {len(failed_urls)}")
print("Файлы: hf_leaderboards_full_data.json и .jsonl")