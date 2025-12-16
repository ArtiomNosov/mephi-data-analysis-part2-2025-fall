#!/usr/bin/env python3
"""
Скрипт для скачивания метрик моделей с лидербордов Hugging Face.
Записывает результаты в JSONL формат построчно.
Получает метрики с лидербордов, а не с карточек моделей.
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
from huggingface_hub import HfApi

# Глобальная переменная для отслеживания состояния
shutdown_flag = False
processed_count = 0

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('fetch_metrics.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def signal_handler(sig, frame):
    """Обработчик сигнала для корректной остановки."""
    global shutdown_flag
    logger.warning("\nПолучен сигнал остановки. Завершаю текущую модель и сохраняю результаты...")
    shutdown_flag = True


# Регистрация обработчиков сигналов
signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)


def get_open_llm_leaderboard_metrics(model_id: str) -> Optional[Dict]:
    """
    Получает метрики модели с Open LLM Leaderboard.
    
    Args:
        model_id: ID модели на Hugging Face
    
    Returns:
        Словарь с метриками или None
    """
    metrics = {}
    
    try:
        # Open LLM Leaderboard использует различные эндпоинты
        # Попытка 1: Через Space API для получения данных
        space_api_url = "https://huggingface.co/api/spaces/HuggingFaceH4/open_llm_leaderboard"
        
        # Попытка 2: Прямой запрос к странице лидерборда для парсинга
        leaderboard_url = "https://huggingface.co/spaces/HuggingFaceH4/open_llm_leaderboard"
        
        response = requests.get(leaderboard_url, timeout=15, headers={
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
            'Accept': 'text/html,application/json'
        })
        
        if response.status_code == 200:
            html_content = response.text
            
            # Ищем JSON данные в скриптах страницы
            json_pattern = r'<script[^>]*type=["\']application/json["\'][^>]*>(.*?)</script>'
            json_matches = re.findall(json_pattern, html_content, re.DOTALL)
            
            for json_str in json_matches:
                try:
                    data = json.loads(json_str)
                    # Ищем модель в данных
                    if isinstance(data, (dict, list)):
                        # Рекурсивный поиск модели
                        def find_model_in_data(obj, target_model_id):
                            if isinstance(obj, dict):
                                # Проверяем, есть ли здесь наша модель
                                for key, value in obj.items():
                                    if isinstance(value, (dict, list)):
                                        result = find_model_in_data(value, target_model_id)
                                        if result:
                                            return result
                                    elif key in ['model', 'model_id', 'name'] and value:
                                        if target_model_id.lower() in str(value).lower():
                                            # Нашли модель, возвращаем её данные
                                            return obj
                            elif isinstance(obj, list):
                                for item in obj:
                                    result = find_model_in_data(item, target_model_id)
                                    if result:
                                        return result
                            return None
                        
                        model_data = find_model_in_data(data, model_id)
                        if model_data:
                            # Извлекаем метрики
                            for key in ['results', 'metrics', 'average', 'humaneval', 'gsm8k', 'mmlu', 
                                       'hellaswag', 'arc', 'truthfulqa', 'winogrande']:
                                if key in model_data:
                                    if isinstance(model_data[key], dict):
                                        metrics.update(model_data[key])
                                    else:
                                        metrics[key] = model_data[key]
                except json.JSONDecodeError:
                    continue
            
            # Если не нашли в JSON, парсим HTML таблицу
            if not metrics:
                html_lower = html_content.lower()
                model_pattern = rf'{re.escape(model_id.lower())}'
                if model_pattern in html_lower:
                    # Ищем метрики в таблице
                    benchmarks = {
                        'humaneval': ['humaneval', 'human eval'],
                        'gsm8k': ['gsm8k', 'gsm 8k'],
                        'mmlu': ['mmlu'],
                        'hellaswag': ['hellaswag'],
                        'arc': ['arc'],
                        'truthfulqa': ['truthfulqa', 'truthful qa'],
                        'winogrande': ['winogrande', 'winogrande'],
                    }
                    
                    for benchmark_key, search_terms in benchmarks.items():
                        for term in search_terms:
                            if term in html_lower:
                                pattern = rf'{re.escape(term)}[:\s]+([0-9]+\.?[0-9]*)'
                                matches = re.findall(pattern, html_lower)
                                if matches:
                                    try:
                                        value = float(matches[0])
                                        if benchmark_key not in metrics or value > metrics[benchmark_key]:
                                            metrics[benchmark_key] = value
                                    except ValueError:
                                        pass
        
    except Exception as e:
        logger.debug(f"Ошибка при получении метрик с Open LLM Leaderboard для {model_id}: {e}")
    
    return metrics if metrics else None


def get_bigcode_leaderboard_metrics(model_id: str) -> Optional[Dict]:
    """
    Получает метрики модели с BigCode Open LLM Leaderboard (для кодовых моделей).
    
    Args:
        model_id: ID модели на Hugging Face
    
    Returns:
        Словарь с метриками или None
    """
    metrics = {}
    
    try:
        # BigCode Leaderboard Space
        # Попытка получить данные через Space API или напрямую
        space_url = "https://huggingface.co/spaces/bigcode/bigcode-models"
        
        # Попытка получить данные через API
        api_url = f"https://huggingface.co/api/spaces/bigcode/bigcode-models"
        
        # Альтернативный способ - парсинг страницы лидерборда
        leaderboard_url = "https://huggingface.co/spaces/bigcode/bigcode-models"
        
        response = requests.get(leaderboard_url, timeout=15, headers={
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
            'Accept': 'application/json'
        })
        
        if response.status_code == 200:
            # Пытаемся найти JSON данные в HTML или получить напрямую
            html_content = response.text
            
            # Ищем JSON данные в скриптах страницы
            import re
            json_pattern = r'<script[^>]*type=["\']application/json["\'][^>]*>(.*?)</script>'
            json_matches = re.findall(json_pattern, html_content, re.DOTALL)
            
            for json_str in json_matches:
                try:
                    data = json.loads(json_str)
                    # Ищем модель в данных
                    if isinstance(data, dict):
                        # Проверяем различные структуры данных
                        if 'models' in data:
                            for model_entry in data['models']:
                                if isinstance(model_entry, dict):
                                    entry_model_id = model_entry.get('model', model_entry.get('model_id', model_entry.get('name', '')))
                                    if entry_model_id and model_id.lower() in entry_model_id.lower():
                                        if 'results' in model_entry:
                                            metrics.update(model_entry['results'])
                                        elif 'metrics' in model_entry:
                                            metrics.update(model_entry['metrics'])
                        elif 'results' in data:
                            # Прямые результаты
                            for entry in data['results']:
                                if isinstance(entry, dict):
                                    entry_model_id = entry.get('model', entry.get('model_id', entry.get('name', '')))
                                    if entry_model_id and model_id.lower() in entry_model_id.lower():
                                        if 'results' in entry:
                                            metrics.update(entry['results'])
                                        elif 'metrics' in entry:
                                            metrics.update(entry['metrics'])
                except json.JSONDecodeError:
                    continue
            
            # Если не нашли в JSON, пытаемся парсить HTML таблицу
            if not metrics:
                # Ищем метрики в HTML таблице лидерборда
                html_lower = html_content.lower()
                
                # Ищем строку с нашей моделью
                model_pattern = rf'{re.escape(model_id.lower())}'
                if model_pattern in html_lower:
                    # Популярные бенчмарки для кодовых моделей
                    code_benchmarks = {
                        'humaneval': ['humaneval', 'human eval', 'human-eval', 'pass@1'],
                        'mbpp': ['mbpp'],
                        'ds-1000': ['ds-1000', 'ds1000'],
                        'codexglue': ['codexglue', 'codex glue'],
                        'bigcodebench': ['bigcodebench', 'bigcode bench'],
                    }
                    
                    for benchmark_key, search_terms in code_benchmarks.items():
                        for term in search_terms:
                            if term in html_lower:
                                # Ищем числовые значения рядом с названием бенчмарка
                                pattern = rf'{re.escape(term)}[:\s]+([0-9]+\.?[0-9]*)'
                                matches = re.findall(pattern, html_lower)
                                if matches:
                                    try:
                                        value = float(matches[0])
                                        if benchmark_key not in metrics or value > metrics[benchmark_key]:
                                            metrics[benchmark_key] = value
                                    except ValueError:
                                        pass
        
    except Exception as e:
        logger.debug(f"Ошибка при получении метрик с BigCode Leaderboard для {model_id}: {e}")
    
    return metrics if metrics else None


def get_leaderboard_metrics_from_api(model_id: str) -> Optional[Dict]:
    """
    Получает метрики модели через различные API лидербордов.
    
    Args:
        model_id: ID модели на Hugging Face
    
    Returns:
        Словарь с метриками или None
    """
    all_metrics = {}
    
    # Пытаемся получить метрики с BigCode Leaderboard (специализированный для кода)
    bigcode_metrics = get_bigcode_leaderboard_metrics(model_id)
    if bigcode_metrics:
        all_metrics.update(bigcode_metrics)
        logger.debug(f"Найдено {len(bigcode_metrics)} метрик с BigCode Leaderboard")
    
    # Пытаемся получить метрики с Open LLM Leaderboard (общий)
    open_llm_metrics = get_open_llm_leaderboard_metrics(model_id)
    if open_llm_metrics:
        all_metrics.update(open_llm_metrics)
        logger.debug(f"Найдено {len(open_llm_metrics)} метрик с Open LLM Leaderboard")
    
    return all_metrics if all_metrics else None


def fetch_metrics_for_model(model_id: str, api: HfApi) -> Dict:
    """
    Получает метрики для одной модели с лидербордов Hugging Face.
    
    Args:
        model_id: ID модели
        api: Экземпляр HfApi (не используется, но оставлен для совместимости)
    
    Returns:
        Словарь с метриками
    """
    result = {
        "model_id": model_id,
        "metrics": {},
        "sources": []
    }
    
    # Получаем метрики с лидербордов
    leaderboard_metrics = get_leaderboard_metrics_from_api(model_id)
    if leaderboard_metrics:
        result["metrics"].update(leaderboard_metrics)
        result["sources"].append("leaderboard")
    
    return result


def write_jsonl_line(output_file: Path, data: Dict):
    """Записывает одну строку в JSONL файл."""
    with open(output_file, 'a', encoding='utf-8') as f:
        f.write(json.dumps(data, ensure_ascii=False) + '\n')
        f.flush()  # Принудительная запись на диск


def main():
    """Основная функция скрипта."""
    global shutdown_flag, processed_count
    
    # Путь к файлу с моделями
    input_file = Path('code_models.json')
    output_file = Path('code_models_metrics.jsonl')
    progress_file = Path('fetch_metrics_progress.json')
    
    if not input_file.exists():
        logger.error(f"Файл {input_file} не найден!")
        return
    
    # Проверяем, есть ли сохраненный прогресс
    start_idx = 0
    if progress_file.exists():
        try:
            with open(progress_file, 'r', encoding='utf-8') as f:
                progress = json.load(f)
                start_idx = progress.get('last_processed_idx', 0)
                logger.info(f"Найден сохраненный прогресс. Продолжаем с модели {start_idx + 1}")
        except:
            logger.warning("Не удалось загрузить прогресс. Начинаем с начала.")
    
    logger.info(f"Загрузка моделей из {input_file}...")
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    models = data.get('models', [])
    total_models = len(models)
    logger.info(f"Найдено моделей: {total_models}")
    
    # Инициализация API (для совместимости, но не используется для получения метрик)
    api = HfApi()
    
    # Статистика
    stats = {
        "models_with_metrics": 0,
        "models_without_metrics": 0,
        "errors": 0
    }
    
    # Обработка моделей
    logger.info("Начало скачивания метрик с лидербордов...")
    logger.info(f"Результаты будут записываться в: {output_file}")
    
    try:
        for idx, model in enumerate(models[start_idx:], start=start_idx + 1):
            if shutdown_flag:
                logger.warning("Получен сигнал остановки. Сохраняю прогресс...")
                break
            
            model_id = model.get('model_id')
            if not model_id:
                continue
            
            logger.info(f"[{idx}/{total_models}] Обработка модели: {model_id}")
            
            try:
                metrics_data = fetch_metrics_for_model(model_id, api)
                
                # Добавляем базовую информацию о модели
                metrics_data.update({
                    "downloads": model.get('downloads'),
                    "likes": model.get('likes'),
                    "created_at": model.get('created_at'),
                    "url": model.get('url'),
                    "processed_at": datetime.now().isoformat(),
                    "index": idx
                })
                
                if metrics_data.get('metrics'):
                    stats["models_with_metrics"] += 1
                    logger.info(f"  ✓ Найдено метрик: {len(metrics_data['metrics'])} (источники: {', '.join(metrics_data.get('sources', []))})")
                else:
                    stats["models_without_metrics"] += 1
                    logger.warning(f"  ✗ Метрики не найдены на лидербордах")
                
                # Записываем результат сразу в JSONL
                write_jsonl_line(output_file, metrics_data)
                processed_count = idx
                
                # Сохраняем прогресс каждые 10 моделей
                if idx % 10 == 0:
                    with open(progress_file, 'w', encoding='utf-8') as f:
                        json.dump({
                            'last_processed_idx': idx,
                            'processed_count': idx,
                            'stats': stats,
                            'timestamp': datetime.now().isoformat()
                        }, f, ensure_ascii=False, indent=2)
                
                # Небольшая задержка, чтобы не перегружать API
                time.sleep(1.0)  # Увеличена задержка для лидербордов
                
            except Exception as e:
                logger.error(f"  ✗ Ошибка при обработке {model_id}: {e}")
                stats["errors"] += 1
                error_data = {
                    "model_id": model_id,
                    "error": str(e),
                    "metrics": {},
                    "processed_at": datetime.now().isoformat(),
                    "index": idx
                }
                write_jsonl_line(output_file, error_data)
    
    except KeyboardInterrupt:
        logger.warning("\nПрервано пользователем. Сохраняю прогресс...")
    
    finally:
        # Сохраняем финальный прогресс
        with open(progress_file, 'w', encoding='utf-8') as f:
            json.dump({
                'last_processed_idx': processed_count,
                'processed_count': processed_count,
                'stats': stats,
                'timestamp': datetime.now().isoformat(),
                'completed': not shutdown_flag
            }, f, ensure_ascii=False, indent=2)
        
        logger.info("=" * 80)
        logger.info("ИТОГИ:")
        logger.info(f"Обработано моделей: {processed_count} из {total_models}")
        logger.info(f"Моделей с метриками: {stats['models_with_metrics']}")
        logger.info(f"Моделей без метрик: {stats['models_without_metrics']}")
        logger.info(f"Ошибок: {stats['errors']}")
        logger.info(f"Результаты сохранены в: {output_file}")
        logger.info(f"Прогресс сохранен в: {progress_file}")
        logger.info("=" * 80)


if __name__ == "__main__":
    main()
