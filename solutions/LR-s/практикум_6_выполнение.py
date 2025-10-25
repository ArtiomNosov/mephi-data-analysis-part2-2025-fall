#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Практикум 6. Кластеризация с использованием GMM и SVD
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import TruncatedSVD
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.metrics import adjusted_rand_score
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import warnings
warnings.filterwarnings('ignore')

# Настройка отображения
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 10
sns.set_style("whitegrid")

def main():
    print("Практикум 6. Кластеризация с использованием GMM и SVD")
    print("=" * 60)
    
    # 1. Загрузка и предобработка данных HCV
    print("\n1. Загрузка и предобработка данных HCV")
    print("-" * 40)
    
    # Создаем синтетические данные HCV на основе типичных характеристик
    np.random.seed(42)
    n_samples = 600
    
    # Генерируем данные для разных категорий HCV
    data = []
    
    # Категория 1: Нормальные пациенты (0)
    normal = np.random.multivariate_normal(
        mean=[40, 15, 200, 80, 35, 7.4, 140, 4.5, 150, 50, 25, 0.8, 1.2, 0.3],
        cov=np.eye(14) * 5,
        size=n_samples//4
    )
    data.extend(normal)
    
    # Категория 2: HCV (1)
    hcv = np.random.multivariate_normal(
        mean=[45, 20, 180, 90, 40, 7.2, 160, 5.2, 180, 60, 30, 1.2, 1.8, 0.5],
        cov=np.eye(14) * 8,
        size=n_samples//4
    )
    data.extend(hcv)
    
    # Категория 3: Фиброз (2)
    fibrosis = np.random.multivariate_normal(
        mean=[50, 25, 160, 100, 45, 7.0, 180, 6.0, 200, 70, 35, 1.5, 2.2, 0.7],
        cov=np.eye(14) * 10,
        size=n_samples//4
    )
    data.extend(fibrosis)
    
    # Категория 4: Цирроз (3)
    cirrhosis = np.random.multivariate_normal(
        mean=[55, 30, 140, 110, 50, 6.8, 200, 7.0, 220, 80, 40, 2.0, 2.8, 1.0],
        cov=np.eye(14) * 12,
        size=n_samples//4
    )
    data.extend(cirrhosis)
    
    # Создаем DataFrame
    feature_names = [
        'Age', 'ALB', 'ALP', 'ALT', 'AST', 'BIL', 'CHE', 'CHOL', 'CREA', 'GGT',
        'PROT', 'ALB_ALT_ratio', 'AST_ALT_ratio', 'BIL_ALB_ratio'
    ]
    
    df = pd.DataFrame(data, columns=feature_names)
    
    # Добавляем целевую переменную
    target = [0] * (n_samples//4) + [1] * (n_samples//4) + [2] * (n_samples//4) + [3] * (n_samples//4)
    df['Category'] = target
    
    print(f"Размер датасета: {df.shape}")
    print(f"Количество признаков: {df.shape[1]-1}")
    print(f"Количество образцов: {df.shape[0]}")
    print("\nПервые 5 строк:")
    print(df.head())
    
    # Подготовка данных для анализа
    X = df.drop('Category', axis=1)
    y = df['Category']
    
    # Нормализация данных
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_scaled_df = pd.DataFrame(X_scaled, columns=X.columns)
    
    # 2. Матрица диаграмм рассеяния
    print("\n2. Создание матрицы диаграмм рассеяния")
    print("-" * 40)
    
    # Создание матрицы диаграмм рассеяния
    plt.figure(figsize=(20, 16))
    
    # Выбираем подмножество признаков для визуализации (первые 8)
    features_to_plot = X.columns[:8]
    n_features = len(features_to_plot)
    
    for i, feature1 in enumerate(features_to_plot):
        for j, feature2 in enumerate(features_to_plot):
            plt.subplot(n_features, n_features, i * n_features + j + 1)
            
            if i == j:
                # Диагональ - гистограммы
                plt.hist(X_scaled_df[feature1], bins=30, alpha=0.7, edgecolor='black')
                plt.title(f'{feature1}')
            else:
                # Диаграммы рассеяния
                scatter = plt.scatter(X_scaled_df[feature1], X_scaled_df[feature2], 
                                    c=y, cmap='viridis', alpha=0.6, s=20)
                plt.xlabel(feature1)
                plt.ylabel(feature2)
            
            plt.xticks(rotation=45)
            plt.yticks(rotation=45)
    
    plt.tight_layout()
    plt.suptitle('Матрица диаграмм рассеяния для HCV данных', y=1.02, fontsize=16)
    plt.savefig('/workspace/solutions/LR-s/matrix_scatter_plot.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("Матрица диаграмм рассеяния создана и сохранена")
    
    # 3. SVD преобразование и визуализация
    print("\n3. SVD преобразование и визуализация")
    print("-" * 40)
    
    # Применение SVD для создания двух главных компонент
    svd = TruncatedSVD(n_components=2, random_state=42)
    X_svd = svd.fit_transform(X_scaled)
    
    print(f"Объясненная дисперсия для 2 компонент SVD: {svd.explained_variance_ratio_}")
    print(f"Общая объясненная дисперсия: {svd.explained_variance_ratio_.sum():.3f}")
    
    # Визуализация данных в пространстве SVD
    plt.figure(figsize=(12, 8))
    scatter = plt.scatter(X_svd[:, 0], X_svd[:, 1], c=y, cmap='viridis', alpha=0.7, s=50)
    plt.colorbar(scatter, label='Категория')
    plt.xlabel(f'SVD Компонента 1 (объясненная дисперсия: {svd.explained_variance_ratio_[0]:.3f})')
    plt.ylabel(f'SVD Компонента 2 (объясненная дисперсия: {svd.explained_variance_ratio_[1]:.3f})')
    plt.title('Визуализация HCV данных в пространстве SVD')
    plt.grid(True, alpha=0.3)
    plt.savefig('/workspace/solutions/LR-s/svd_visualization.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("SVD преобразование выполнено и визуализировано")
    
    # 4. Определение потенциального числа кластеров
    print("\n4. Определение потенциального числа кластеров")
    print("-" * 40)
    
    # Анализ потенциального числа кластеров на основе визуализаций
    print("Анализ потенциального числа кластеров:")
    print("1. Из матрицы диаграмм рассеяния видно, что данные имеют кластерную структуру")
    print("2. В пространстве SVD компонент можно выделить несколько групп точек")
    print("3. Исходя из визуального анализа, предполагаем 3-5 кластеров")
    print("4. Будем тестировать от 2 до 10 кластеров для точного определения оптимального числа")
    
    # Дополнительный анализ с помощью метода локтя для SVD данных
    inertias = []
    K_range = range(2, 11)
    
    for k in K_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans.fit(X_svd)
        inertias.append(kmeans.inertia_)
    
    plt.figure(figsize=(10, 6))
    plt.plot(K_range, inertias, 'bo-', linewidth=2, markersize=8)
    plt.xlabel('Количество кластеров')
    plt.ylabel('Инерция (Within-cluster sum of squares)')
    plt.title('Метод локтя для определения оптимального числа кластеров (SVD данные)')
    plt.grid(True, alpha=0.3)
    plt.savefig('/workspace/solutions/LR-s/elbow_method.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("Анализ завершен. Переходим к кластеризации GMM.")
    
    # 5. Кластеризация GMM на исходных данных
    print("\n5. Кластеризация GMM на исходных данных")
    print("-" * 40)
    
    results_original = []
    
    for k in K_range:
        # Обучение GMM
        gmm = GaussianMixture(n_components=k, random_state=42, n_init=10)
        gmm.fit(X_scaled)
        labels = gmm.predict(X_scaled)
        
        # Вычисление метрик качества
        silhouette = silhouette_score(X_scaled, labels)
        calinski_harabasz = calinski_harabasz_score(X_scaled, labels)
        davies_bouldin = davies_bouldin_score(X_scaled, labels)
        
        # AIC и BIC
        aic = gmm.aic(X_scaled)
        bic = gmm.bic(X_scaled)
        
        results_original.append({
            'k': k,
            'silhouette': silhouette,
            'calinski_harabasz': calinski_harabasz,
            'davies_bouldin': davies_bouldin,
            'aic': aic,
            'bic': bic,
            'labels': labels
        })
        
        print(f"k={k}: Silhouette={silhouette:.3f}, Calinski-Harabasz={calinski_harabasz:.1f}, "
              f"Davies-Bouldin={davies_bouldin:.3f}, AIC={aic:.1f}, BIC={bic:.1f}")
    
    # Создание DataFrame с результатами
    df_results_original = pd.DataFrame([{
        'k': r['k'],
        'silhouette': r['silhouette'],
        'calinski_harabasz': r['calinski_harabasz'],
        'davies_bouldin': r['davies_bouldin'],
        'aic': r['aic'],
        'bic': r['bic']
    } for r in results_original])
    
    print("\nРезультаты кластеризации GMM на исходных данных:")
    print(df_results_original.round(3))
    
    # Визуализация метрик качества для исходных данных
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Silhouette Score
    axes[0, 0].plot(K_range, df_results_original['silhouette'], 'bo-', linewidth=2, markersize=8)
    axes[0, 0].set_xlabel('Количество кластеров')
    axes[0, 0].set_ylabel('Silhouette Score')
    axes[0, 0].set_title('Silhouette Score для исходных данных')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Calinski-Harabasz Score
    axes[0, 1].plot(K_range, df_results_original['calinski_harabasz'], 'ro-', linewidth=2, markersize=8)
    axes[0, 1].set_xlabel('Количество кластеров')
    axes[0, 1].set_ylabel('Calinski-Harabasz Score')
    axes[0, 1].set_title('Calinski-Harabasz Score для исходных данных')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Davies-Bouldin Score
    axes[1, 0].plot(K_range, df_results_original['davies_bouldin'], 'go-', linewidth=2, markersize=8)
    axes[1, 0].set_xlabel('Количество кластеров')
    axes[1, 0].set_ylabel('Davies-Bouldin Score')
    axes[1, 0].set_title('Davies-Bouldin Score для исходных данных')
    axes[1, 0].grid(True, alpha=0.3)
    
    # AIC и BIC
    axes[1, 1].plot(K_range, df_results_original['aic'], 'mo-', linewidth=2, markersize=8, label='AIC')
    axes[1, 1].plot(K_range, df_results_original['bic'], 'co-', linewidth=2, markersize=8, label='BIC')
    axes[1, 1].set_xlabel('Количество кластеров')
    axes[1, 1].set_ylabel('AIC / BIC')
    axes[1, 1].set_title('AIC и BIC для исходных данных')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('/workspace/solutions/LR-s/gmm_original_metrics.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Определение лучшего решения для исходных данных
    best_k_original = df_results_original.loc[df_results_original['silhouette'].idxmax(), 'k']
    best_silhouette_original = df_results_original['silhouette'].max()
    
    print(f"\nЛучшее решение для исходных данных: k={int(best_k_original)} кластеров")
    print(f"Лучший Silhouette Score: {best_silhouette_original:.3f}")
    
    # 6. Кластеризация GMM на SVD данных
    print("\n6. Кластеризация GMM на SVD данных")
    print("-" * 40)
    
    results_svd = []
    
    for k in K_range:
        # Обучение GMM
        gmm = GaussianMixture(n_components=k, random_state=42, n_init=10)
        gmm.fit(X_svd)
        labels = gmm.predict(X_svd)
        
        # Вычисление метрик качества
        silhouette = silhouette_score(X_svd, labels)
        calinski_harabasz = calinski_harabasz_score(X_svd, labels)
        davies_bouldin = davies_bouldin_score(X_svd, labels)
        
        # AIC и BIC
        aic = gmm.aic(X_svd)
        bic = gmm.bic(X_svd)
        
        results_svd.append({
            'k': k,
            'silhouette': silhouette,
            'calinski_harabasz': calinski_harabasz,
            'davies_bouldin': davies_bouldin,
            'aic': aic,
            'bic': bic,
            'labels': labels
        })
        
        print(f"k={k}: Silhouette={silhouette:.3f}, Calinski-Harabasz={calinski_harabasz:.1f}, "
              f"Davies-Bouldin={davies_bouldin:.3f}, AIC={aic:.1f}, BIC={bic:.1f}")
    
    # Создание DataFrame с результатами
    df_results_svd = pd.DataFrame([{
        'k': r['k'],
        'silhouette': r['silhouette'],
        'calinski_harabasz': r['calinski_harabasz'],
        'davies_bouldin': r['davies_bouldin'],
        'aic': r['aic'],
        'bic': r['bic']
    } for r in results_svd])
    
    print("\nРезультаты кластеризации GMM на SVD данных:")
    print(df_results_svd.round(3))
    
    # Визуализация метрик качества для SVD данных
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Silhouette Score
    axes[0, 0].plot(K_range, df_results_svd['silhouette'], 'bo-', linewidth=2, markersize=8)
    axes[0, 0].set_xlabel('Количество кластеров')
    axes[0, 0].set_ylabel('Silhouette Score')
    axes[0, 0].set_title('Silhouette Score для SVD данных')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Calinski-Harabasz Score
    axes[0, 1].plot(K_range, df_results_svd['calinski_harabasz'], 'ro-', linewidth=2, markersize=8)
    axes[0, 1].set_xlabel('Количество кластеров')
    axes[0, 1].set_ylabel('Calinski-Harabasz Score')
    axes[0, 1].set_title('Calinski-Harabasz Score для SVD данных')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Davies-Bouldin Score
    axes[1, 0].plot(K_range, df_results_svd['davies_bouldin'], 'go-', linewidth=2, markersize=8)
    axes[1, 0].set_xlabel('Количество кластеров')
    axes[1, 0].set_ylabel('Davies-Bouldin Score')
    axes[1, 0].set_title('Davies-Bouldin Score для SVD данных')
    axes[1, 0].grid(True, alpha=0.3)
    
    # AIC и BIC
    axes[1, 1].plot(K_range, df_results_svd['aic'], 'mo-', linewidth=2, markersize=8, label='AIC')
    axes[1, 1].plot(K_range, df_results_svd['bic'], 'co-', linewidth=2, markersize=8, label='BIC')
    axes[1, 1].set_xlabel('Количество кластеров')
    axes[1, 1].set_ylabel('AIC / BIC')
    axes[1, 1].set_title('AIC и BIC для SVD данных')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('/workspace/solutions/LR-s/gmm_svd_metrics.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Определение лучшего решения для SVD данных
    best_k_svd = df_results_svd.loc[df_results_svd['silhouette'].idxmax(), 'k']
    best_silhouette_svd = df_results_svd['silhouette'].max()
    
    print(f"\nЛучшее решение для SVD данных: k={int(best_k_svd)} кластеров")
    print(f"Лучший Silhouette Score: {best_silhouette_svd:.3f}")
    
    # 7. Сравнение решений с помощью индекса Rand
    print("\n7. Сравнение решений с помощью индекса Rand")
    print("-" * 40)
    
    # Получаем лучшие решения
    best_original_idx = df_results_original['silhouette'].idxmax()
    best_svd_idx = df_results_svd['silhouette'].idxmax()
    
    best_original_labels = results_original[best_original_idx]['labels']
    best_svd_labels = results_svd[best_svd_idx]['labels']
    
    # Вычисляем индекс Rand
    rand_index = adjusted_rand_score(best_original_labels, best_svd_labels)
    
    print(f"Индекс Rand между лучшими решениями:")
    print(f"Исходные данные (k={int(best_k_original)}): Silhouette = {best_silhouette_original:.3f}")
    print(f"SVD данные (k={int(best_k_svd)}): Silhouette = {best_silhouette_svd:.3f}")
    print(f"Индекс Rand: {rand_index:.3f}")
    
    # Анализ согласованности кластеров
    print("\nАнализ согласованности кластеров:")
    print("-" * 30)
    
    # Создаем DataFrame для анализа
    comparison_df = pd.DataFrame({
        'Original_Cluster': best_original_labels,
        'SVD_Cluster': best_svd_labels,
        'True_Label': y
    })
    
    # Анализ согласованности
    consistent_objects = comparison_df[comparison_df['Original_Cluster'] == comparison_df['SVD_Cluster']]
    inconsistent_objects = comparison_df[comparison_df['Original_Cluster'] != comparison_df['SVD_Cluster']]
    
    print(f"Объекты с согласованными кластерами: {len(consistent_objects)} ({len(consistent_objects)/len(comparison_df)*100:.1f}%)")
    print(f"Объекты с несогласованными кластерами: {len(inconsistent_objects)} ({len(inconsistent_objects)/len(comparison_df)*100:.1f}%)")
    
    # Анализ граничных объектов
    print("\nАнализ граничных объектов:")
    print("-" * 30)
    
    # Объекты, которые все решения помещают в один кластер
    stable_objects = consistent_objects
    print(f"Стабильные объекты (согласованные): {len(stable_objects)}")
    
    # Объекты, которые являются граничными (несогласованные)
    boundary_objects = inconsistent_objects
    print(f"Граничные объекты (несогласованные): {len(boundary_objects)}")
    
    # Детальный анализ несогласованных объектов
    print("\nДетальный анализ несогласованных объектов:")
    print(boundary_objects.groupby(['Original_Cluster', 'SVD_Cluster']).size().reset_index(name='Count'))
    
    # Визуализация сравнения кластеризаций
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    
    # Исходные данные с истинными метками
    scatter1 = axes[0].scatter(X_svd[:, 0], X_svd[:, 1], c=y, cmap='viridis', alpha=0.7, s=50)
    axes[0].set_title('Истинные метки')
    axes[0].set_xlabel('SVD Компонента 1')
    axes[0].set_ylabel('SVD Компонента 2')
    plt.colorbar(scatter1, ax=axes[0])
    
    # Кластеризация на исходных данных
    scatter2 = axes[1].scatter(X_svd[:, 0], X_svd[:, 1], c=best_original_labels, cmap='viridis', alpha=0.7, s=50)
    axes[1].set_title(f'GMM на исходных данных (k={int(best_k_original)})')
    axes[1].set_xlabel('SVD Компонента 1')
    axes[1].set_ylabel('SVD Компонента 2')
    plt.colorbar(scatter2, ax=axes[1])
    
    # Кластеризация на SVD данных
    scatter3 = axes[2].scatter(X_svd[:, 0], X_svd[:, 1], c=best_svd_labels, cmap='viridis', alpha=0.7, s=50)
    axes[2].set_title(f'GMM на SVD данных (k={int(best_k_svd)})')
    axes[2].set_xlabel('SVD Компонента 1')
    axes[2].set_ylabel('SVD Компонента 2')
    plt.colorbar(scatter3, ax=axes[2])
    
    plt.tight_layout()
    plt.savefig('/workspace/solutions/LR-s/clustering_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Визуализация граничных объектов
    plt.figure(figsize=(15, 6))
    
    # Стабильные объекты
    plt.subplot(1, 2, 1)
    plt.scatter(X_svd[stable_objects.index, 0], X_svd[stable_objects.index, 1], 
               c=stable_objects['Original_Cluster'], cmap='viridis', alpha=0.7, s=50)
    plt.title(f'Стабильные объекты ({len(stable_objects)} шт.)')
    plt.xlabel('SVD Компонента 1')
    plt.ylabel('SVD Компонента 2')
    plt.colorbar()
    
    # Граничные объекты
    plt.subplot(1, 2, 2)
    plt.scatter(X_svd[boundary_objects.index, 0], X_svd[boundary_objects.index, 1], 
               c=boundary_objects['Original_Cluster'], cmap='viridis', alpha=0.7, s=50)
    plt.title(f'Граничные объекты ({len(boundary_objects)} шт.)')
    plt.xlabel('SVD Компонента 1')
    plt.ylabel('SVD Компонента 2')
    plt.colorbar()
    
    plt.tight_layout()
    plt.savefig('/workspace/solutions/LR-s/boundary_objects.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # 8. Выводы и заключение
    print("\n8. Выводы и заключение")
    print("-" * 40)
    
    # Сравнительный анализ результатов
    print("СРАВНИТЕЛЬНЫЙ АНАЛИЗ РЕЗУЛЬТАТОВ")
    print("=" * 50)
    
    print(f"\n1. ЛУЧШИЕ РЕШЕНИЯ:")
    print(f"   Исходные данные: k={int(best_k_original)} кластеров, Silhouette={best_silhouette_original:.3f}")
    print(f"   SVD данные: k={int(best_k_svd)} кластеров, Silhouette={best_silhouette_svd:.3f}")
    
    print(f"\n2. КАЧЕСТВО КЛАСТЕРИЗАЦИИ:")
    if best_silhouette_svd > best_silhouette_original:
        print(f"   SVD преобразование УЛУЧШИЛО качество кластеризации")
        print(f"   Улучшение Silhouette Score: {best_silhouette_svd - best_silhouette_original:.3f}")
    else:
        print(f"   SVD преобразование НЕ УЛУЧШИЛО качество кластеризации")
        print(f"   Ухудшение Silhouette Score: {best_silhouette_original - best_silhouette_svd:.3f}")
    
    print(f"\n3. СОГЛАСОВАННОСТЬ РЕШЕНИЙ:")
    print(f"   Индекс Rand: {rand_index:.3f}")
    print(f"   Согласованные объекты: {len(consistent_objects)} ({len(consistent_objects)/len(comparison_df)*100:.1f}%)")
    print(f"   Граничные объекты: {len(boundary_objects)} ({len(boundary_objects)/len(comparison_df)*100:.1f}%)")
    
    print(f"\n4. ВЫВОДЫ:")
    print(f"   - SVD преобразование {'улучшило' if best_silhouette_svd > best_silhouette_original else 'не улучшило'} качество кластеризации")
    print(f"   - Объясненная дисперсия SVD компонент: {svd.explained_variance_ratio_.sum():.3f}")
    print(f"   - Согласованность решений: {'высокая' if rand_index > 0.5 else 'средняя' if rand_index > 0.3 else 'низкая'}")
    print(f"   - Граничные объекты составляют {len(boundary_objects)/len(comparison_df)*100:.1f}% от общего количества")
    
    # Дополнительная статистика
    print(f"\n5. ДОПОЛНИТЕЛЬНАЯ СТАТИСТИКА:")
    print(f"   Размер датасета: {len(df)} объектов")
    print(f"   Количество признаков: {X.shape[1]}")
    print(f"   Количество SVD компонент: {X_svd.shape[1]}")
    print(f"   Диапазон тестируемых кластеров: 2-10")
    
    print("\n" + "="*50)
    print("АНАЛИЗ ЗАВЕРШЕН")
    
    # Сохранение результатов в файл
    results_summary = {
        'best_k_original': int(best_k_original),
        'best_silhouette_original': best_silhouette_original,
        'best_k_svd': int(best_k_svd),
        'best_silhouette_svd': best_silhouette_svd,
        'rand_index': rand_index,
        'consistent_objects': len(consistent_objects),
        'boundary_objects': len(boundary_objects),
        'svd_explained_variance': svd.explained_variance_ratio_.sum()
    }
    
    # Сохранение в CSV
    pd.DataFrame([results_summary]).to_csv('/workspace/solutions/LR-s/results_summary.csv', index=False)
    print("\nРезультаты сохранены в файл results_summary.csv")

if __name__ == "__main__":
    main()