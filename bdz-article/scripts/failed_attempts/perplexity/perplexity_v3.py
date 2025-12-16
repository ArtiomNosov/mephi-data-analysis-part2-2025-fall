import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
import json
import re
import time

def find_and_save_leaderboards(filename='leaderboards.jsonl'):
    base_url = "https://huggingface.co/spaces"
    all_leaderboards = set()  # Глобальный уникальный набор
    leaderboard_keywords = ['leaderboard', 'ranking', 'rankings', 'top', 'best', 'rank', 'score']
    
    page = 1
    while True:
        print(f"Парсинг страницы {page}...")
        url = f"{base_url}?page={page}"
        response = requests.get(url)
        soup = BeautifulSoup(response.text, 'html.parser')
        
        new_on_page = set()
        for a in soup.find_all('a', href=True):
            href = a['href']
            full_url = urljoin(base_url, href)
            parsed = urlparse(full_url)
            
            if '/spaces/' in parsed.path:
                path_parts = parsed.path.strip('/').split('/')
                if len(path_parts) >= 3:
                    last_part = path_parts[-1].lower()
                    # Строгая регулярка: /spaces/*/*leaderboard
                    if re.match(r'^.*leaderboard', last_part):
                        if full_url not in all_leaderboards:
                            new_on_page.add(full_url)
        
        # Сохраняем только новые уникальные ссылки
        for link in new_on_page:
            entry = {'url': link, 'found_at': 'huggingface.co/spaces', 'page': page}
            with open(filename, 'a', encoding='utf-8') as f:
                f.write(json.dumps(entry) + '\n')
            print(f"Сохранено: {link}")
            all_leaderboards.add(link)
        
        if not new_on_page:
            break
        
        page += 1
        time.sleep(1)
    
    print(f"Всего уникальных: {len(all_leaderboards)}")

# Запуск
find_and_save_leaderboards()
