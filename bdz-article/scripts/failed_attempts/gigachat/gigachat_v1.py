import re
import json
import urllib.parse
from googlesearch import search

def fetch_leaderboard_links(query):
    """
    Извлекает ссылки из поисковых результатов Google.
    
    :param query: Строка поискового запроса
    :return: Список извлечённых ссылок
    """
    return list(search(query, num_results=10))

def filter_links_by_regex(links, regex_pattern):
    """
    Фильтрация ссылок по регулярному выражению.
    
    :param links: Список исходных ссылок
    :param regex_pattern: Шаблон регулярного выражения
    :return: Отфильтрованный список ссылок
    """
    pattern = re.compile(regex_pattern)
    filtered_links = []
    for link in links:
        if pattern.match(link):
            filtered_links.append(link)
    return filtered_links

def save_to_json(data, filename='leaderboard_links.json'):
    """
    Сохраняет данные в JSON-файл.
    
    :param data: Данные для сохранения
    :param filename: Имя выходного файла
    """
    with open(filename, 'w') as f:
        json.dump(data, f, indent=4)

def main():
    # Настройки поиска и фильтрации
    search_query = "site:huggingface.co spaces leaderboard"
    regex_pattern = r'^https://huggingface.co/spaces/.+?/.*leaderboard.*$'
    
    # Шаг 1: Сбор ссылок через Google
    raw_links = fetch_leaderboard_links(search_query)
    
    # Шаг 2: Фильтрация по регулярному выражению
    filtered_links = filter_links_by_regex(raw_links, regex_pattern)
    
    # Шаг 3: Убираем дубликаты
    unique_links = list(set(filtered_links))
    
    # Шаг 4: Сохраняем результат в JSON
    save_to_json(unique_links)
    
    print(f"Сохранено {len(unique_links)} уникальных ссылок.")

if __name__ == "__main__":
    main()