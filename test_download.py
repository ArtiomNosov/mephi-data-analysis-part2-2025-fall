#!/usr/bin/env python3
"""
Тестирование функции скачивания датасета
"""

import os
import requests
import zipfile
from io import BytesIO
import time

def download_yandex_dataset(url, output_dir='images_dataset'):
    """
    Скачивает датасет с Яндекс.Диска по публичной ссылке
    """
    try:
        # Создаем директорию если не существует
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        print(f"Попытка скачивания датасета с {url}...")
        
        # Для Яндекс.Диска нужно получить прямую ссылку на скачивание
        # Публичные ссылки Яндекс.Диска имеют формат: https://disk.yandex.ru/d/ID
        # Нужно преобразовать в: https://disk.yandex.ru/i/ID
        if 'disk.yandex.ru/d/' in url:
            file_id = url.split('/d/')[-1]
            download_url = f"https://disk.yandex.ru/i/{file_id}"
        else:
            download_url = url
        
        print(f"Преобразованная ссылка: {download_url}")
        
        # Заголовки для имитации браузера
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        
        # Пытаемся скачать файл
        response = requests.get(download_url, headers=headers, stream=True, timeout=30)
        
        print(f"Статус ответа: {response.status_code}")
        print(f"Content-Type: {response.headers.get('content-type', 'неизвестно')}")
        print(f"Content-Length: {response.headers.get('content-length', 'неизвестно')}")
        
        if response.status_code == 200:
            # Определяем тип файла по заголовкам
            content_type = response.headers.get('content-type', '')
            
            if 'zip' in content_type or download_url.endswith('.zip'):
                # Если это ZIP файл
                zip_path = os.path.join(output_dir, 'dataset.zip')
                print(f"Скачиваем ZIP файл в {zip_path}...")
                
                with open(zip_path, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)
                
                print(f"ZIP файл скачан, размер: {os.path.getsize(zip_path)} байт")
                
                # Распаковываем ZIP
                print("Распаковываем ZIP файл...")
                with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                    zip_ref.extractall(output_dir)
                
                # Удаляем ZIP файл
                os.remove(zip_path)
                print(f"Датасет успешно скачан и распакован в {output_dir}")
                return True
            else:
                # Если это не ZIP, сохраняем как есть
                filename = f"dataset_{int(time.time())}.bin"
                file_path = os.path.join(output_dir, filename)
                print(f"Скачиваем файл как {filename}...")
                
                with open(file_path, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)
                
                print(f"Файл скачан как {file_path}, размер: {os.path.getsize(file_path)} байт")
                return True
        else:
            print(f"Ошибка скачивания: HTTP {response.status_code}")
            print(f"Ответ сервера: {response.text[:500]}...")
            return False
            
    except Exception as e:
        print(f"Ошибка при скачивании датасета: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Тестирование скачивания"""
    url = 'https://disk.yandex.ru/d/Jr1gmKoYgQVE5Q'
    output_dir = 'test_images_dataset'
    
    print("Тестирование скачивания датасета...")
    print(f"URL: {url}")
    print(f"Выходная директория: {output_dir}")
    print("-" * 50)
    
    success = download_yandex_dataset(url, output_dir)
    
    if success:
        print("\n✅ Скачивание успешно!")
        if os.path.exists(output_dir):
            files = os.listdir(output_dir)
            print(f"Файлов в директории: {len(files)}")
            for f in files[:10]:  # Показываем первые 10 файлов
                print(f"  - {f}")
            if len(files) > 10:
                print(f"  ... и еще {len(files) - 10} файлов")
    else:
        print("\n❌ Скачивание не удалось!")
        
        # Пробуем альтернативные методы
        print("\nПробуем альтернативные методы...")
        
        # Метод 1: Прямая ссылка на скачивание
        alt_url = url.replace('/d/', '/i/')
        print(f"Пробуем: {alt_url}")
        success2 = download_yandex_dataset(alt_url, output_dir)
        
        if not success2:
            # Метод 2: Через API Яндекс.Диска
            print("Пробуем через API Яндекс.Диска...")
            try:
                # Получаем публичную ссылку через API
                api_url = f"https://cloud-api.yandex.net/v1/disk/public/resources/download?public_key={url}"
                api_response = requests.get(api_url)
                
                if api_response.status_code == 200:
                    download_info = api_response.json()
                    direct_url = download_info.get('href')
                    print(f"Получена прямая ссылка: {direct_url}")
                    
                    success3 = download_yandex_dataset(direct_url, output_dir)
                    if success3:
                        print("✅ Скачивание через API успешно!")
                    else:
                        print("❌ Скачивание через API не удалось")
                else:
                    print(f"❌ Ошибка API: {api_response.status_code}")
            except Exception as e:
                print(f"❌ Ошибка API: {e}")

if __name__ == "__main__":
    main()