#!/usr/bin/env python3
"""
Скрипт для парсинга лидербордов Hugging Face целиком.
Сохраняет все данные с лидербордов для последующего анализа.
"""

import json
import requests
import time
import signal
import sys
import re
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional
import logging

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('fetch_leaderboards.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def parse_json_from_html(html_content: str) -> List[Dict]:
    """
    Извлекает JSON данные из HTML страницы.
    
    Args:
        html_content: HTML содержимое страницы
    
    Returns:
        Список найденных JSON объектов
    """
    json_data = []
    
    # Паттерн для поиска JSON в script тегах
    patterns = [
        r'<script[^>]*type=["\']application/json["\'][^>]*>(.*?)</script>',
        r'<script[^>]*id=["\']__NEXT_DATA__["\'][^>]*>(.*?)</script>',
        r'window\.__INITIAL_STATE__\s*=\s*({.*?});',
        r'const\s+data\s*=\s*({.*?});',
    ]
    
    for pattern in patterns:
        matches = re.findall(pattern, html_content, re.DOTALL | re.IGNORECASE)
        for match in matches:
            try:
                data = json.loads(match)
                json_data.append(data)
            except json.JSONDecodeError:
                continue
    
    return json_data


def extract_all_models_from_data(obj, path="", depth=0, max_depth=10):
    """
    Рекурсивно извлекает все записи, похожие на модели из структуры данных.
    
    Args:
        obj: Объект для поиска
        path: Путь в структуре (для отладки)
        depth: Текущая глубина рекурсии
        max_depth: Максимальная глубина рекурсии
    
    Returns:
        Список найденных моделей
    """
    if depth > max_depth:
        return []
    
    models_found = []
    
    if isinstance(obj, dict):
        # Проверяем, похоже ли это на запись модели
        # Ключевые признаки записи модели:
        model_indicators = ['model', 'model_id', 'name', 'average', 'humaneval', 'mbpp', 
                          'gsm8k', 'mmlu', 'hellaswag', 'arc', 'pass@1', 'pass@k',
                          'bigcodebench', 'ds-1000', 'codexglue']
        
        has_model_indicator = any(key in obj for key in model_indicators)
        has_metrics = any(key in obj for key in ['results', 'metrics', 'scores', 'benchmarks'])
        
        if has_model_indicator or has_metrics:
            # Это может быть запись модели
            models_found.append(obj)
        
        # Продолжаем поиск в дочерних элементах
        for key, value in obj.items():
            models_found.extend(extract_all_models_from_data(value, f"{path}.{key}", depth + 1, max_depth))
    
    elif isinstance(obj, list):
        # Если список не пустой и первый элемент - словарь
        if obj and isinstance(obj[0], dict):
            first_keys = set(obj[0].keys())
            # Проверяем, похоже ли это на список моделей
            model_indicators = ['model', 'model_id', 'name', 'average', 'humaneval']
            if any(key in first_keys for key in model_indicators):
                # Это список моделей
                models_found.extend(obj)
            else:
                # Рекурсивно ищем в элементах списка
                for idx, item in enumerate(obj):
                    models_found.extend(extract_all_models_from_data(item, f"{path}[{idx}]", depth + 1, max_depth))
        else:
            # Рекурсивно ищем в элементах списка
            for idx, item in enumerate(obj):
                models_found.extend(extract_all_models_from_data(item, f"{path}[{idx}]", depth + 1, max_depth))
    
    return models_found


def fetch_bigcode_leaderboard() -> Optional[Dict]:
    """
    Парсит BigCode Open LLM Leaderboard целиком.
    
    Returns:
        Словарь с данными лидерборда или None
    """
    logger.info("Парсинг BigCode Open LLM Leaderboard...")
    
    leaderboard_url = "https://huggingface.co/spaces/bigcode/bigcode-models"
    
    try:
        # Попытка получить данные через API Space
        api_url = "https://huggingface.co/api/spaces/bigcode/bigcode-models"
        api_response = requests.get(api_url, timeout=15, headers={
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })
        
        api_data = None
        if api_response.status_code == 200:
            try:
                api_data = api_response.json()
                logger.info("Получены данные через Space API")
            except:
                pass
        
        # Загружаем HTML страницу
        response = requests.get(leaderboard_url, timeout=30, headers={
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8'
        })
        
        if response.status_code != 200:
            logger.error(f"Ошибка при загрузке BigCode Leaderboard: HTTP {response.status_code}")
            return None
        
        html_content = response.text
        
        # Извлекаем JSON данные из HTML
        json_data_list = parse_json_from_html(html_content)
        
        # Добавляем данные из API, если есть
        if api_data:
            json_data_list.append(api_data)
        
        # Ищем данные лидерборда
        leaderboard_data = {
            "source": "bigcode/bigcode-models",
            "url": leaderboard_url,
            "fetched_at": datetime.now().isoformat(),
            "models": [],
            "raw_json": json_data_list
        }
        
        # Извлекаем все модели из JSON данных
        all_models = []
        for json_data in json_data_list:
            models = extract_all_models_from_data(json_data)
            all_models.extend(models)
        
        # Удаляем дубликаты (по model_id или name, если есть)
        seen = set()
        unique_models = []
        for model in all_models:
            if isinstance(model, dict):
                # Создаем ключ для проверки уникальности
                model_key = None
                for key in ['model', 'model_id', 'name', 'id']:
                    if key in model:
                        model_key = str(model[key]).lower()
                        break
                
                if model_key and model_key not in seen:
                    seen.add(model_key)
                    unique_models.append(model)
                elif not model_key:
                    # Если нет ключа, добавляем все равно (может быть уникальная структура)
                    unique_models.append(model)
        
        leaderboard_data["models"] = unique_models
        
        # Если не нашли в JSON, парсим HTML таблицу
        if not leaderboard_data["models"]:
            logger.info("Модели не найдены в JSON, парсим HTML таблицу...")
            models_from_html = parse_leaderboard_table_from_html(html_content)
            leaderboard_data["models"] = models_from_html
        
        logger.info(f"Найдено моделей в BigCode Leaderboard: {len(leaderboard_data['models'])}")
        return leaderboard_data
        
    except Exception as e:
        logger.error(f"Ошибка при парсинге BigCode Leaderboard: {e}", exc_info=True)
        return None


def fetch_open_llm_leaderboard() -> Optional[Dict]:
    """
    Парсит Open LLM Leaderboard целиком.
    
    Returns:
        Словарь с данными лидерборда или None
    """
    logger.info("Парсинг Open LLM Leaderboard...")
    
    leaderboard_url = "https://huggingface.co/spaces/HuggingFaceH4/open_llm_leaderboard"
    
    try:
        # Попытка получить данные через API Space
        api_url = "https://huggingface.co/api/spaces/HuggingFaceH4/open_llm_leaderboard"
        api_response = requests.get(api_url, timeout=15, headers={
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })
        
        api_data = None
        if api_response.status_code == 200:
            try:
                api_data = api_response.json()
                logger.info("Получены данные через Space API")
            except:
                pass
        
        # Загружаем HTML страницу
        response = requests.get(leaderboard_url, timeout=30, headers={
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8'
        })
        
        if response.status_code != 200:
            logger.error(f"Ошибка при загрузке Open LLM Leaderboard: HTTP {response.status_code}")
            return None
        
        html_content = response.text
        
        # Извлекаем JSON данные из HTML
        json_data_list = parse_json_from_html(html_content)
        
        # Добавляем данные из API, если есть
        if api_data:
            json_data_list.append(api_data)
        
        # Ищем данные лидерборда
        leaderboard_data = {
            "source": "HuggingFaceH4/open_llm_leaderboard",
            "url": leaderboard_url,
            "fetched_at": datetime.now().isoformat(),
            "models": [],
            "raw_json": json_data_list
        }
        
        # Извлекаем все модели из JSON данных
        all_models = []
        for json_data in json_data_list:
            models = extract_all_models_from_data(json_data)
            all_models.extend(models)
        
        # Удаляем дубликаты (по model_id или name, если есть)
        seen = set()
        unique_models = []
        for model in all_models:
            if isinstance(model, dict):
                # Создаем ключ для проверки уникальности
                model_key = None
                for key in ['model', 'model_id', 'name', 'id']:
                    if key in model:
                        model_key = str(model[key]).lower()
                        break
                
                if model_key and model_key not in seen:
                    seen.add(model_key)
                    unique_models.append(model)
                elif not model_key:
                    # Если нет ключа, добавляем все равно (может быть уникальная структура)
                    unique_models.append(model)
        
        leaderboard_data["models"] = unique_models
        
        # Если не нашли в JSON, парсим HTML таблицу
        if not leaderboard_data["models"]:
            logger.info("Модели не найдены в JSON, парсим HTML таблицу...")
            models_from_html = parse_leaderboard_table_from_html(html_content)
            leaderboard_data["models"] = models_from_html
        
        logger.info(f"Найдено моделей в Open LLM Leaderboard: {len(leaderboard_data['models'])}")
        return leaderboard_data
        
    except Exception as e:
        logger.error(f"Ошибка при парсинге Open LLM Leaderboard: {e}", exc_info=True)
        return None


def parse_leaderboard_table_from_html(html_content: str) -> List[Dict]:
    """
    Парсит таблицу лидерборда из HTML.
    
    Args:
        html_content: HTML содержимое страницы
    
    Returns:
        Список словарей с данными моделей
    """
    models = []
    
    try:
        from bs4 import BeautifulSoup
        soup = BeautifulSoup(html_content, 'html.parser')
        
        # Ищем таблицы
        tables = soup.find_all('table')
        
        for table in tables:
            rows = table.find_all('tr')
            if not rows:
                continue
            
            # Пытаемся определить заголовки
            header_row = rows[0]
            headers = [th.get_text(strip=True) for th in header_row.find_all(['th', 'td'])]
            
            if not headers:
                continue
            
            # Парсим строки данных
            for row in rows[1:]:
                cells = row.find_all(['td', 'th'])
                if not cells:
                    continue
                
                model_data = {}
                for idx, cell in enumerate(cells):
                    if idx < len(headers):
                        header = headers[idx].lower().replace(' ', '_').replace('-', '_')
                        value = cell.get_text(strip=True)
                        
                        # Пытаемся преобразовать в число, если возможно
                        try:
                            if '.' in value or '%' in value:
                                value = float(re.sub(r'[^\d.]', '', value))
                            else:
                                value = int(re.sub(r'[^\d]', '', value))
                        except:
                            pass
                        
                        model_data[header] = value
                
                if model_data:
                    models.append(model_data)
        
    except ImportError:
        logger.warning("BeautifulSoup не установлен, пропускаем парсинг HTML таблиц")
    except Exception as e:
        logger.debug(f"Ошибка при парсинге HTML таблицы: {e}")
    
    return models


def save_leaderboard_data(leaderboard_data: Dict, filename: str):
    """
    Сохраняет данные лидерборда в JSON файл.
    
    Args:
        leaderboard_data: Данные лидерборда
        filename: Имя файла для сохранения
    """
    output_file = Path(filename)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(leaderboard_data, f, ensure_ascii=False, indent=2)
    
    file_size = output_file.stat().st_size / (1024 * 1024)  # MB
    logger.info(f"Данные сохранены в {filename} ({file_size:.2f} MB)")


def main():
    """Основная функция скрипта."""
    logger.info("=" * 80)
    logger.info("НАЧАЛО ПАРСИНГА ЛИДЕРБОРДОВ")
    logger.info("=" * 80)
    
    all_leaderboards = {}
    
    # Парсим BigCode Leaderboard
    bigcode_data = fetch_bigcode_leaderboard()
    if bigcode_data:
        all_leaderboards["bigcode"] = bigcode_data
        save_leaderboard_data(bigcode_data, "bigcode_leaderboard.json")
        time.sleep(2)  # Пауза между запросами
    
    # Парсим Open LLM Leaderboard
    open_llm_data = fetch_open_llm_leaderboard()
    if open_llm_data:
        all_leaderboards["open_llm"] = open_llm_data
        save_leaderboard_data(open_llm_data, "open_llm_leaderboard.json")
    
    # Сохраняем все данные вместе
    if all_leaderboards:
        all_leaderboards["summary"] = {
            "fetched_at": datetime.now().isoformat(),
            "total_leaderboards": len(all_leaderboards) - 1,  # -1 для summary
            "bigcode_models": len(all_leaderboards.get("bigcode", {}).get("models", [])),
            "open_llm_models": len(all_leaderboards.get("open_llm", {}).get("models", []))
        }
        
        save_leaderboard_data(all_leaderboards, "all_leaderboards.json")
    
    logger.info("=" * 80)
    logger.info("ИТОГИ:")
    logger.info("=" * 80)
    
    if "bigcode" in all_leaderboards:
        logger.info(f"BigCode Leaderboard: {len(all_leaderboards['bigcode']['models'])} моделей")
    
    if "open_llm" in all_leaderboards:
        logger.info(f"Open LLM Leaderboard: {len(all_leaderboards['open_llm']['models'])} моделей")
    
    logger.info("=" * 80)


if __name__ == "__main__":
    main()

