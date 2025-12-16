import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
import re

def find_leaderboard_spaces():
    base_url = "https://huggingface.co/spaces"
    response = requests.get(base_url)
    soup = BeautifulSoup(response.text, 'html.parser')
    
    spaces_links = []
    for a in soup.find_all('a', href=True):
        href = a['href']
        full_url = urljoin(base_url, href)
        parsed = urlparse(full_url)
        
        # Только Spaces: /spaces/username/space_name
        if '/spaces/' in parsed.path:
            # Паттерн: leaderboard в конце пути (например, /leaderboard или /my-leaderboard)
            path_parts = parsed.path.strip('/').split('/')
            if len(path_parts) >= 3 and 'leaderboard' in path_parts[-1].lower():
                spaces_links.append(full_url)
    
    return list(set(spaces_links))  # Удаляем дубликаты

# Запуск
leaderboards = find_leaderboard_spaces()
for url in leaderboards:
    print(url)
