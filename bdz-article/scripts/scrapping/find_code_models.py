#!/usr/bin/env python3
"""
Скрипт для поиска всех моделей на Hugging Face, в названии которых есть "code".
"""

from huggingface_hub import HfApi
import json
from typing import List, Dict
from datetime import datetime


def find_code_models(search_term: str = "code", limit: int = None) -> List[Dict]:
    """
    Находит все модели на Hugging Face, содержащие указанный термин в названии.
    
    Args:
        search_term: Термин для поиска в названии модели (по умолчанию "code")
        limit: Максимальное количество моделей для возврата (None = без ограничений)
    
    Returns:
        Список словарей с информацией о моделях
    """
    api = HfApi()
    
    print(f"Поиск моделей с '{search_term}' в названии на Hugging Face...")
    print("Это может занять некоторое время...\n")
    
    models = []
    page = 0
    per_page = 100
    
    try:
        # Используем list_models с фильтрацией по поисковому запросу
        # search_term будет искаться в названии модели
        all_models = api.list_models(
            search=search_term,
            sort="downloads",
            direction=-1,  # Сортировка по убыванию популярности
            limit=limit if limit else 10000  # Большое число для получения всех результатов
        )
        
        for model in all_models:
            model_info = {
                "model_id": model.id,
                "author": model.author if hasattr(model, 'author') else None,
                "downloads": model.downloads if hasattr(model, 'downloads') else None,
                "likes": model.likes if hasattr(model, 'likes') else None,
                "tags": model.tags if hasattr(model, 'tags') else [],
                "created_at": model.created_at.isoformat() if hasattr(model, 'created_at') and model.created_at else None,
                "updated_at": model.updated_at.isoformat() if hasattr(model, 'updated_at') and model.updated_at else None,
                "url": f"https://huggingface.co/{model.id}"
            }
            models.append(model_info)
            
            if limit and len(models) >= limit:
                break
        
        print(f"Найдено моделей: {len(models)}")
        
    except Exception as e:
        print(f"Ошибка при поиске моделей: {e}")
        return []
    
    return models


def save_results(models: List[Dict], filename: str = "code_models.json"):
    """
    Сохраняет результаты поиска в JSON файл.
    
    Args:
        models: Список словарей с информацией о моделях
        filename: Имя файла для сохранения
    """
    output = {
        "search_term": "code",
        "total_found": len(models),
        "search_date": datetime.now().isoformat(),
        "models": models
    }
    
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    
    print(f"\nРезультаты сохранены в файл: {filename}")


def print_summary(models: List[Dict], top_n: int = 20):
    """
    Выводит краткую сводку по найденным моделям.
    
    Args:
        models: Список словарей с информацией о моделях
        top_n: Количество топ-моделей для вывода
    """
    print(f"\n{'='*80}")
    print(f"ТОП-{min(top_n, len(models))} моделей по популярности (скачивания):")
    print(f"{'='*80}\n")
    
    # Сортируем по количеству скачиваний (если доступно)
    sorted_models = sorted(
        models, 
        key=lambda x: x.get('downloads', 0) or 0, 
        reverse=True
    )[:top_n]
    
    for i, model in enumerate(sorted_models, 1):
        downloads = model.get('downloads', 'N/A')
        likes = model.get('likes', 'N/A')
        print(f"{i:3d}. {model['model_id']}")
        print(f"     Скачивания: {downloads:,}" if downloads != 'N/A' else f"     Скачивания: {downloads}")
        print(f"     Лайки: {likes}")
        print(f"     URL: {model['url']}")
        print()


def main():
    """Основная функция скрипта."""
    # Поиск моделей с "code" в названии
    models = find_code_models(search_term="code", limit=None)
    
    if not models:
        print("Модели не найдены.")
        return
    
    # Вывод краткой сводки
    print_summary(models, top_n=30)
    
    # Сохранение результатов в JSON
    save_results(models, filename="code_models.json")
    
    # Также сохраним краткий список только названий моделей
    model_names = [model['model_id'] for model in models]
    with open("code_models_list.txt", 'w', encoding='utf-8') as f:
        f.write(f"Всего найдено моделей: {len(model_names)}\n")
        f.write(f"Дата поиска: {datetime.now().isoformat()}\n\n")
        f.write("\n".join(model_names))
    
    print("Список названий моделей сохранен в файл: code_models_list.txt")


if __name__ == "__main__":
    main()

