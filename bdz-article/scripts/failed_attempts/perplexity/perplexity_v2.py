import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
import json
import re
import time

def find_and_save_leaderboards(filename='leaderboards.jsonl'):
    base_url = "https://huggingface.co/spaces"
    leaderboard_keywords = {
        'leaderboard', 'ranking', 'rankings', 'top', 'best', 
        'leaderboard-v2', 'rank', 'score', 'scores', 'halloffame'
    }
    
    all_leaderboards = []
    
    page = 1
    while True:
        print(f"Парсинг страницы {page}...")
        url = f"{base_url}?page={page}"
        response = requests.get(url)
        soup = BeautifulSoup(response.text, 'html.parser')
        
        spaces_links = []
        for a in soup.find_all('a', href=True):
            href = a['href']
            full_url = urljoin(base_url, href)
            parsed = urlparse(full_url)
            
            if '/spaces/' in parsed.path:
                path_parts = parsed.path.strip('/').split('/')
                if len(path_parts) >= 3:
                    last_part = path_parts[-1].lower()
                    # Проверяем наличие любого ключевого слова в последней части
                    if any(keyword in last_part for keyword in leaderboard_keywords):
                        spaces_links.append(full_url)
        
        unique_links = list(set(spaces_links))
        if not unique_links:
            break  # Нет больше страниц
        
        # Сохраняем каждую ссылку сразу в JSONL
        for link in unique_links:
            entry = {'url': link, 'found_at': 'huggingface.co/spaces', 'page': page}
            with open(filename, 'a', encoding='utf-8') as f:
                f.write(json.dumps(entry) + '\n')
            print(f"Сохранено: {link}")
        
        all_leaderboards.extend(unique_links)
        page += 1
        time.sleep(1)  # Пауза для уважения к серверу
    
    print(f"Всего найдено: {len(all_leaderboards)} лидербордов. Сохранено в {filename}")

# Запуск
find_and_save_leaderboards()
