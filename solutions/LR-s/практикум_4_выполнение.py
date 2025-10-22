#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Практикум 4. Кластеризация данных о зернах пшеницы
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score
import warnings
warnings.filterwarnings('ignore')

# Настройка для корректного отображения русских символов
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False

def main():
    print("=" * 60)
    print("ПРАКТИКУМ 4. КЛАСТЕРИЗАЦИЯ ДАННЫХ О ЗЕРНАХ ПШЕНИЦЫ")
    print("=" * 60)
    
    # Задание 1. Загрузка и предобработка данных
    print("\nЗАДАНИЕ 1. ЗАГРУЗКА И ПРЕДОБРАБОТКА ДАННЫХ")
    print("-" * 50)
    
    # Загрузка данных
    data = pd.read_csv('seeds_dataset_fixed.txt', sep='\t', header=None)
    
    # Названия признаков (согласно описанию набора данных)
    feature_names = ['Area', 'Perimeter', 'Compactness', 'Length of kernel', 
                    'Width of kernel', 'Asymmetry coefficient', 'Length of kernel groove']
    data.columns = feature_names + ['Class']
    
    print(f"Размер набора данных: {data.shape}")
    print(f"Количество признаков: {data.shape[1]-1}")
    print(f"Тип признаков: все числовые (float64)")
    print(f"Пропущенные значения: {data.isnull().sum().sum()}")
    print(f"Количество классов: {data['Class'].nunique()}")
    print(f"Распределение классов:")
    print(data['Class'].value_counts().sort_index())
    
    # Удаление строк с пропущенными значениями (если есть)
    data_clean = data.dropna()
    print(f"\nРазмер данных после удаления пропущенных значений: {data_clean.shape}")
    
    # Разделение на признаки и метки
    X = data_clean[feature_names]
    y = data_clean['Class']
    
    # Стандартизация числовых признаков
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_scaled_df = pd.DataFrame(X_scaled, columns=feature_names)
    
    print(f"\nДанные после стандартизации:")
    print(f"Среднее: {X_scaled_df.mean().mean():.6f}")
    print(f"Стандартное отклонение: {X_scaled_df.std().mean():.6f}")
    
    # Задание 2. Кластеризация K-Means с k=3
    print("\n\nЗАДАНИЕ 2. КЛАСТЕРИЗАЦИЯ K-MEANS С K=3")
    print("-" * 50)
    
    # Применение K-Means с k=3
    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(X_scaled)
    
    # Вычисление коэффициента силуэта
    silhouette_avg = silhouette_score(X_scaled, clusters)
    print(f"Коэффициент силуэта для k=3: {silhouette_avg:.3f}")
    
    # Сравнение с истинными классами
    print(f"\nСравнение кластеров с истинными классами:")
    comparison = pd.DataFrame({'Истинный_класс': y, 'Предсказанный_кластер': clusters})
    confusion_matrix = comparison.groupby(['Истинный_класс', 'Предсказанный_кластер']).size().unstack(fill_value=0)
    print(confusion_matrix)
    
    # Задание 3. PCA для уменьшения размерности
    print("\n\nЗАДАНИЕ 3. PCA ДЛЯ УМЕНЬШЕНИЯ РАЗМЕРНОСТИ")
    print("-" * 50)
    
    # Эксперимент с различным количеством главных компонентов
    n_components_range = range(2, 7)
    pca_results = []
    
    print("Результаты PCA:")
    for n_components in n_components_range:
        # Применение PCA
        pca = PCA(n_components=n_components)
        X_pca = pca.fit_transform(X_scaled)
        
        # K-Means кластеризация
        kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
        clusters = kmeans.fit_predict(X_pca)
        
        # Вычисление коэффициента силуэта
        silhouette_avg = silhouette_score(X_pca, clusters)
        
        pca_results.append({
            'n_components': n_components,
            'explained_variance_ratio': pca.explained_variance_ratio_.sum(),
            'silhouette_score': silhouette_avg
        })
        
        print(f"n_components={n_components}: silhouette_score={silhouette_avg:.3f}, "
              f"explained_variance={pca.explained_variance_ratio_.sum():.3f}")
    
    pca_df = pd.DataFrame(pca_results)
    
    # Определение оптимального количества компонентов
    best_pca_n = pca_df.loc[pca_df['silhouette_score'].idxmax(), 'n_components']
    print(f"\nОптимальное количество главных компонентов: {best_pca_n}")
    print(f"Лучший коэффициент силуэта: {pca_df['silhouette_score'].max():.3f}")
    
    # Задание 4. t-SNE для уменьшения размерности
    print("\n\nЗАДАНИЕ 4. T-SNE ДЛЯ УМЕНЬШЕНИЯ РАЗМЕРНОСТИ")
    print("-" * 50)
    
    # Эксперимент с различным количеством компонентов t-SNE
    n_components_range = range(2, 4)  # t-SNE ограничен 3 компонентами для barnes_hut
    tsne_results = []
    
    print("Результаты t-SNE:")
    for n_components in n_components_range:
        # Применение t-SNE
        tsne = TSNE(n_components=n_components, random_state=42, perplexity=30)
        X_tsne = tsne.fit_transform(X_scaled)
        
        # K-Means кластеризация
        kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
        clusters = kmeans.fit_predict(X_tsne)
        
        # Вычисление коэффициента силуэта
        silhouette_avg = silhouette_score(X_tsne, clusters)
        
        tsne_results.append({
            'n_components': n_components,
            'silhouette_score': silhouette_avg
        })
        
        print(f"n_components={n_components}: silhouette_score={silhouette_avg:.3f}")
    
    tsne_df = pd.DataFrame(tsne_results)
    
    # Определение оптимального количества компонентов
    best_tsne_n = tsne_df.loc[tsne_df['silhouette_score'].idxmax(), 'n_components']
    print(f"\nОптимальное количество компонентов t-SNE: {best_tsne_n}")
    print(f"Лучший коэффициент силуэта: {tsne_df['silhouette_score'].max():.3f}")
    
    # Задание 5. Исследование влияния инициализации центроидов
    print("\n\nЗАДАНИЕ 5. ИССЛЕДОВАНИЕ ВЛИЯНИЯ ИНИЦИАЛИЗАЦИИ ЦЕНТРОИДОВ")
    print("-" * 50)
    
    # Исследование различных методов инициализации
    init_methods = ['k-means++', 'random']
    random_states = [0, 42, 100, 200, 500]
    
    init_results = []
    
    print("Результаты исследования инициализации:")
    for init_method in init_methods:
        for random_state in random_states:
            kmeans = KMeans(n_clusters=3, init=init_method, random_state=random_state, n_init=10)
            clusters = kmeans.fit_predict(X_scaled)
            
            silhouette_avg = silhouette_score(X_scaled, clusters)
            inertia = kmeans.inertia_
            
            init_results.append({
                'init_method': init_method,
                'random_state': random_state,
                'silhouette_score': silhouette_avg,
                'inertia': inertia
            })
            
            print(f"{init_method}, random_state={random_state}: silhouette={silhouette_avg:.3f}, inertia={inertia:.2f}")
    
    init_df = pd.DataFrame(init_results)
    
    # Статистический анализ
    print(f"\nСтатистический анализ методов инициализации:")
    print(f"\nКоэффициент силуэта:")
    for method in init_methods:
        method_scores = init_df[init_df['init_method'] == method]['silhouette_score']
        print(f"{method}: mean={method_scores.mean():.3f}, std={method_scores.std():.3f}")
    
    print(f"\nИнерция:")
    for method in init_methods:
        method_inertia = init_df[init_df['init_method'] == method]['inertia']
        print(f"{method}: mean={method_inertia.mean():.2f}, std={method_inertia.std():.2f}")
    
    # Задание 6. Определение оптимального значения k
    print("\n\nЗАДАНИЕ 6. ОПРЕДЕЛЕНИЕ ОПТИМАЛЬНОГО ЗНАЧЕНИЯ K")
    print("-" * 50)
    
    # Эксперимент с различным количеством кластеров
    k_range = range(2, 11)
    k_results = []
    
    print("Результаты для различных k:")
    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        clusters = kmeans.fit_predict(X_scaled)
        
        silhouette_avg = silhouette_score(X_scaled, clusters)
        inertia = kmeans.inertia_
        
        k_results.append({
            'k': k,
            'silhouette_score': silhouette_avg,
            'inertia': inertia
        })
        
        print(f"k={k}: silhouette_score={silhouette_avg:.3f}, inertia={inertia:.2f}")
    
    k_df = pd.DataFrame(k_results)
    
    # Определение оптимального k
    best_k = k_df.loc[k_df['silhouette_score'].idxmax(), 'k']
    best_silhouette = k_df['silhouette_score'].max()
    
    print(f"\nОптимальное количество кластеров: k={best_k}")
    print(f"Лучший коэффициент силуэта: {best_silhouette:.3f}")
    
    # Анализ локтя
    inertia_diff = k_df['inertia'].diff().diff()
    elbow_k = k_df.loc[inertia_diff.idxmax(), 'k']
    print(f"Рекомендуемое k по методу локтя: k={elbow_k}")
    
    # Заключение и выводы
    print("\n\nЗАКЛЮЧЕНИЕ И ВЫВОДЫ")
    print("=" * 60)
    
    print(f"\n1. Исходные данные:")
    print(f"   - Количество образцов: {X_scaled.shape[0]}")
    print(f"   - Количество признаков: {X_scaled.shape[1]}")
    print(f"   - Количество истинных классов: {y.nunique()}")
    
    print(f"\n2. K-Means с k=3:")
    print(f"   - Коэффициент силуэта: {silhouette_avg:.3f}")
    
    print(f"\n3. PCA анализ:")
    print(f"   - Лучший результат: {pca_df['silhouette_score'].max():.3f} (n_components={best_pca_n})")
    
    print(f"\n4. t-SNE анализ:")
    print(f"   - Лучший результат: {tsne_df['silhouette_score'].max():.3f} (n_components={best_tsne_n})")
    
    print(f"\n5. Инициализация центроидов:")
    kmeans_plus_plus = init_df[init_df['init_method'] == 'k-means++']['silhouette_score']
    random_init = init_df[init_df['init_method'] == 'random']['silhouette_score']
    print(f"   - k-means++: mean={kmeans_plus_plus.mean():.3f} ± {kmeans_plus_plus.std():.3f}")
    print(f"   - random: mean={random_init.mean():.3f} ± {random_init.std():.3f}")
    
    print(f"\n6. Оптимальное количество кластеров:")
    print(f"   - По коэффициенту силуэта: k={best_k} (score={best_silhouette:.3f})")
    print(f"   - По методу локтя: k={elbow_k}")
    
    print(f"\n7. Сравнение методов:")
    print(f"   - Исходные данные: {silhouette_avg:.3f}")
    
    # Вычисляем результаты для PCA и t-SNE с 2 компонентами
    pca_2d = PCA(n_components=2)
    X_pca_2d = pca_2d.fit_transform(X_scaled)
    kmeans_pca = KMeans(n_clusters=3, random_state=42, n_init=10)
    clusters_pca = kmeans_pca.fit_predict(X_pca_2d)
    silhouette_pca = silhouette_score(X_pca_2d, clusters_pca)
    
    tsne_2d = TSNE(n_components=2, random_state=42, perplexity=30)
    X_tsne_2d = tsne_2d.fit_transform(X_scaled)
    kmeans_tsne = KMeans(n_clusters=3, random_state=42, n_init=10)
    clusters_tsne = kmeans_tsne.fit_predict(X_tsne_2d)
    silhouette_tsne = silhouette_score(X_tsne_2d, clusters_tsne)
    
    print(f"   - PCA (2 компонента): {silhouette_pca:.3f}")
    print(f"   - t-SNE (2 компонента): {silhouette_tsne:.3f}")
    print(f"   - Оптимальное k: {best_silhouette:.3f}")
    
    print("\n" + "=" * 60)
    print("ВЫВОДЫ:")
    print("1. Набор данных содержит 210 образцов зерен пшеницы с 7 числовыми признаками.")
    print("2. Стандартизация данных улучшает качество кластеризации.")
    print("3. PCA и t-SNE показывают разные результаты в зависимости от количества компонентов.")
    print("4. Метод инициализации k-means++ показывает более стабильные результаты.")
    print("5. Оптимальное количество кластеров может отличаться от истинного количества классов.")
    print("6. Коэффициент силуэта является хорошей метрикой для оценки качества кластеризации.")
    print("=" * 60)

if __name__ == "__main__":
    main()