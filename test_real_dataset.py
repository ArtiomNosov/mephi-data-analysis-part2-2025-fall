#!/usr/bin/env python3
"""
Тестирование работы с реальным датасетом
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from PIL import Image
import os
import warnings
warnings.filterwarnings('ignore')

def load_and_process_images(directory):
    """Загрузка и обработка изображений"""
    images = []
    labels = []
    filenames = []
    
    for filename in os.listdir(directory):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            try:
                # Загружаем изображение
                img_path = os.path.join(directory, filename)
                img = Image.open(img_path)
                
                # Конвертируем в RGB если необходимо
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                
                # Изменяем размер до стандартного (64x64)
                img = img.resize((64, 64))
                
                # Конвертируем в массив и нормализуем
                img_array = np.array(img)
                img_vector = img_array.flatten() / 255.0  # Нормализация к [0,1]
                
                images.append(img_vector)
                filenames.append(filename)
                
                # Извлекаем метки из имени файла
                if 'synthetic' in filename:
                    # Для синтетических данных
                    idx = int(filename.split('_')[1].split('.')[0])
                    if idx < 33:
                        labels.append({'pose': 1, 'mood': 0, 'glasses': 0})
                    elif idx < 66:
                        labels.append({'pose': 0, 'mood': 1, 'glasses': 0})
                    else:
                        labels.append({'pose': 0, 'mood': 0, 'glasses': 1})
                    
                    if np.random.random() > 0.5:
                        labels[-1]['glasses'] = 1
                else:
                    # Для реальных данных извлекаем метки из имени файла
                    # Формат: name_pose_mood_glasses.jpg
                    name_without_ext = filename.split('.')[0]
                    parts = name_without_ext.split('_')
                    
                    if len(parts) >= 4:  # name_pose_mood_glasses
                        try:
                            # Извлекаем позу
                            pose_str = parts[1].lower()
                            if pose_str in ['straight', 'up']:
                                pose = 1
                            elif pose_str in ['left', 'right']:
                                pose = 0
                            else:
                                pose = np.random.randint(0, 2)
                            
                            # Извлекаем настроение
                            mood_str = parts[2].lower()
                            if mood_str in ['happy', 'neutral']:
                                mood = 1
                            elif mood_str in ['sad', 'angry']:
                                mood = 0
                            else:
                                mood = np.random.randint(0, 2)
                            
                            # Извлекаем солнцезащитные очки
                            glasses_str = parts[3].lower()
                            if glasses_str in ['open']:
                                glasses = 0
                            elif glasses_str in ['sunglasses']:
                                glasses = 1
                            else:
                                glasses = np.random.randint(0, 2)
                            
                            labels.append({'pose': pose, 'mood': mood, 'glasses': glasses})
                            
                        except (IndexError, ValueError):
                            labels.append({
                                'pose': np.random.randint(0, 2),
                                'mood': np.random.randint(0, 2),
                                'glasses': np.random.randint(0, 2)
                            })
                    else:
                        labels.append({
                            'pose': np.random.randint(0, 2),
                            'mood': np.random.randint(0, 2),
                            'glasses': np.random.randint(0, 2)
                        })
                        
            except Exception as e:
                print(f"Ошибка при обработке {filename}: {e}")
                continue
    
    return np.array(images), labels, filenames

def test_real_dataset():
    """Тестирование с реальным датасетом"""
    print("="*60)
    print("ТЕСТИРОВАНИЕ С РЕАЛЬНЫМ ДАТАСЕТОМ")
    print("="*60)
    
    # Загружаем изображения
    dataset_dir = 'images_dataset'
    if os.path.exists(dataset_dir):
        images_array, labels_list, filenames_list = load_and_process_images(dataset_dir)
        
        print(f"Загружено {len(images_array)} изображений")
        print(f"Размер вектора изображения: {images_array.shape[1]}")
        
        # Создаем DataFrame с метками
        labels_df = pd.DataFrame(labels_list)
        print(f"\nРаспределение классов:")
        print(f"Поза: {labels_df['pose'].value_counts().to_dict()}")
        print(f"Настроение: {labels_df['mood'].value_counts().to_dict()}")
        print(f"Солнцезащитные очки: {labels_df['glasses'].value_counts().to_dict()}")
        
        # Показываем примеры файлов
        print(f"\nПримеры файлов:")
        for i, filename in enumerate(filenames_list[:10]):
            print(f"  {i+1}. {filename}")
        
        # Анализ с PCA
        print(f"\nАнализ с PCA...")
        pca = PCA(n_components=2)
        images_pca = pca.fit_transform(images_array)
        
        print(f"Объясненная дисперсия: {pca.explained_variance_ratio_}")
        print(f"Суммарная объясненная дисперсия: {pca.explained_variance_ratio_.sum():.4f}")
        
        # Классификация
        print(f"\nКлассификация...")
        X_train, X_test, y_train, y_test = train_test_split(
            images_pca, labels_df['pose'], test_size=0.3, random_state=42, stratify=labels_df['pose']
        )
        
        model = LogisticRegression(random_state=42, max_iter=1000)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        print(f"Точность предсказания позы: {accuracy:.4f}")
        
        # Анализ с t-SNE
        print(f"\nАнализ с t-SNE...")
        tsne = TSNE(n_components=2, random_state=42, max_iter=1000)
        images_tsne = tsne.fit_transform(images_array)
        
        # Классификация на t-SNE данных
        X_train_tsne, X_test_tsne, y_train_tsne, y_test_tsne = train_test_split(
            images_tsne, labels_df['pose'], test_size=0.3, random_state=42, stratify=labels_df['pose']
        )
        
        model_tsne = LogisticRegression(random_state=42, max_iter=1000)
        model_tsne.fit(X_train_tsne, y_train_tsne)
        y_pred_tsne = model_tsne.predict(X_test_tsne)
        accuracy_tsne = accuracy_score(y_test_tsne, y_pred_tsne)
        
        print(f"Точность предсказания позы (t-SNE): {accuracy_tsne:.4f}")
        
        print(f"\n✅ Тестирование с реальным датасетом завершено успешно!")
        return True
        
    else:
        print(f"❌ Директория {dataset_dir} не найдена!")
        return False

if __name__ == "__main__":
    test_real_dataset()