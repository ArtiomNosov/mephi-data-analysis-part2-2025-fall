#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Скрипт для создания визуализаций результатов кластеризации
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
    print("Создание визуализаций для лабораторной работы...")
    
    # Загрузка данных
    data = pd.read_csv('seeds_dataset_fixed.txt', sep='\t', header=None)
    feature_names = ['Area', 'Perimeter', 'Compactness', 'Length of kernel', 
                    'Width of kernel', 'Asymmetry coefficient', 'Length of kernel groove']
    data.columns = feature_names + ['Class']
    
    # Предобработка
    X = data[feature_names]
    y = data['Class']
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # 1. Визуализация распределения признаков
    print("1. Создание графиков распределения признаков...")
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    axes = axes.ravel()
    
    for i, feature in enumerate(feature_names):
        axes[i].hist(X[feature], alpha=0.7, bins=20, color='skyblue', edgecolor='black')
        axes[i].set_title(f'{feature}', fontsize=10)
        axes[i].set_xlabel('Значение')
        axes[i].set_ylabel('Частота')
    
    # Удаляем последний пустой subplot
    fig.delaxes(axes[7])
    plt.tight_layout()
    plt.savefig('распределение_признаков.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. K-Means с k=3
    print("2. Создание визуализации K-Means с k=3...")
    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(X_scaled)
    
    # PCA для 2D визуализации
    pca_2d = PCA(n_components=2)
    X_pca_2d = pca_2d.fit_transform(X_scaled)
    
    plt.figure(figsize=(12, 5))
    
    # Истинные классы
    plt.subplot(1, 2, 1)
    scatter = plt.scatter(X_pca_2d[:, 0], X_pca_2d[:, 1], c=y, cmap='viridis', alpha=0.7)
    plt.colorbar(scatter)
    plt.title('Истинные классы')
    plt.xlabel(f'PC1 ({pca_2d.explained_variance_ratio_[0]:.2%} дисперсии)')
    plt.ylabel(f'PC2 ({pca_2d.explained_variance_ratio_[1]:.2%} дисперсии)')
    
    # K-Means кластеры
    plt.subplot(1, 2, 2)
    scatter = plt.scatter(X_pca_2d[:, 0], X_pca_2d[:, 1], c=clusters, cmap='viridis', alpha=0.7)
    plt.colorbar(scatter)
    plt.title('K-Means кластеры (k=3)')
    plt.xlabel(f'PC1 ({pca_2d.explained_variance_ratio_[0]:.2%} дисперсии)')
    plt.ylabel(f'PC2 ({pca_2d.explained_variance_ratio_[1]:.2%} дисперсии)')
    
    plt.tight_layout()
    plt.savefig('kmeans_k3.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. PCA анализ
    print("3. Создание графиков PCA анализа...")
    n_components_range = range(2, 7)
    pca_results = []
    
    for n_components in n_components_range:
        pca = PCA(n_components=n_components)
        X_pca = pca.fit_transform(X_scaled)
        kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
        clusters = kmeans.fit_predict(X_pca)
        silhouette_avg = silhouette_score(X_pca, clusters)
        pca_results.append({
            'n_components': n_components,
            'explained_variance_ratio': pca.explained_variance_ratio_.sum(),
            'silhouette_score': silhouette_avg
        })
    
    pca_df = pd.DataFrame(pca_results)
    
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    # График коэффициента силуэта
    axes[0].plot(pca_df['n_components'], pca_df['silhouette_score'], 'bo-', linewidth=2, markersize=8)
    axes[0].set_xlabel('Количество главных компонентов')
    axes[0].set_ylabel('Коэффициент силуэта')
    axes[0].set_title('Зависимость коэффициента силуэта от количества PC')
    axes[0].grid(True, alpha=0.3)
    
    # График объясненной дисперсии
    axes[1].plot(pca_df['n_components'], pca_df['explained_variance_ratio'], 'ro-', linewidth=2, markersize=8)
    axes[1].set_xlabel('Количество главных компонентов')
    axes[1].set_ylabel('Объясненная дисперсия')
    axes[1].set_title('Зависимость объясненной дисперсии от количества PC')
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('pca_анализ.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 4. t-SNE анализ
    print("4. Создание графиков t-SNE анализа...")
    n_components_range = range(2, 4)
    tsne_results = []
    
    for n_components in n_components_range:
        tsne = TSNE(n_components=n_components, random_state=42, perplexity=30)
        X_tsne = tsne.fit_transform(X_scaled)
        kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
        clusters = kmeans.fit_predict(X_tsne)
        silhouette_avg = silhouette_score(X_tsne, clusters)
        tsne_results.append({
            'n_components': n_components,
            'silhouette_score': silhouette_avg
        })
    
    tsne_df = pd.DataFrame(tsne_results)
    
    plt.figure(figsize=(10, 6))
    plt.plot(tsne_df['n_components'], tsne_df['silhouette_score'], 'go-', linewidth=2, markersize=8)
    plt.xlabel('Количество компонентов t-SNE')
    plt.ylabel('Коэффициент силуэта')
    plt.title('Зависимость коэффициента силуэта от количества компонентов t-SNE')
    plt.grid(True, alpha=0.3)
    plt.savefig('tsne_анализ.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 5. Визуализация кластеров в t-SNE пространстве
    print("5. Создание визуализации кластеров в t-SNE пространстве...")
    tsne_2d = TSNE(n_components=2, random_state=42, perplexity=30)
    X_tsne_2d = tsne_2d.fit_transform(X_scaled)
    
    kmeans_tsne = KMeans(n_clusters=3, random_state=42, n_init=10)
    clusters_tsne = kmeans_tsne.fit_predict(X_tsne_2d)
    
    plt.figure(figsize=(12, 5))
    
    # Истинные классы
    plt.subplot(1, 2, 1)
    scatter = plt.scatter(X_tsne_2d[:, 0], X_tsne_2d[:, 1], c=y, cmap='viridis', alpha=0.7)
    plt.colorbar(scatter)
    plt.title('Истинные классы (t-SNE)')
    plt.xlabel('t-SNE компонент 1')
    plt.ylabel('t-SNE компонент 2')
    
    # K-Means кластеры
    plt.subplot(1, 2, 2)
    scatter = plt.scatter(X_tsne_2d[:, 0], X_tsne_2d[:, 1], c=clusters_tsne, cmap='viridis', alpha=0.7)
    plt.colorbar(scatter)
    plt.title('K-Means кластеры (t-SNE)')
    plt.xlabel('t-SNE компонент 1')
    plt.ylabel('t-SNE компонент 2')
    
    plt.tight_layout()
    plt.savefig('tsne_кластеры.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 6. Анализ оптимального k
    print("6. Создание графиков анализа оптимального k...")
    k_range = range(2, 11)
    k_results = []
    
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
    
    k_df = pd.DataFrame(k_results)
    
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    # График коэффициента силуэта
    axes[0].plot(k_df['k'], k_df['silhouette_score'], 'bo-', linewidth=2, markersize=8)
    axes[0].set_xlabel('Количество кластеров (k)')
    axes[0].set_ylabel('Коэффициент силуэта')
    axes[0].set_title('Зависимость коэффициента силуэта от k')
    axes[0].grid(True, alpha=0.3)
    axes[0].set_xticks(k_range)
    
    # График инерции (метод локтя)
    axes[1].plot(k_df['k'], k_df['inertia'], 'ro-', linewidth=2, markersize=8)
    axes[1].set_xlabel('Количество кластеров (k)')
    axes[1].set_ylabel('Инерция')
    axes[1].set_title('Метод локтя для определения оптимального k')
    axes[1].grid(True, alpha=0.3)
    axes[1].set_xticks(k_range)
    
    plt.tight_layout()
    plt.savefig('оптимальное_k.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 7. Визуализация оптимальных кластеров
    print("7. Создание визуализации оптимальных кластеров...")
    best_k = k_df.loc[k_df['silhouette_score'].idxmax(), 'k']
    kmeans_optimal = KMeans(n_clusters=best_k, random_state=42, n_init=10)
    clusters_optimal = kmeans_optimal.fit_predict(X_scaled)
    
    plt.figure(figsize=(12, 5))
    
    # Истинные классы
    plt.subplot(1, 2, 1)
    scatter = plt.scatter(X_pca_2d[:, 0], X_pca_2d[:, 1], c=y, cmap='viridis', alpha=0.7)
    plt.colorbar(scatter)
    plt.title('Истинные классы')
    plt.xlabel(f'PC1 ({pca_2d.explained_variance_ratio_[0]:.2%} дисперсии)')
    plt.ylabel(f'PC2 ({pca_2d.explained_variance_ratio_[1]:.2%} дисперсии)')
    
    # Оптимальные кластеры
    plt.subplot(1, 2, 2)
    scatter = plt.scatter(X_pca_2d[:, 0], X_pca_2d[:, 1], c=clusters_optimal, cmap='viridis', alpha=0.7)
    plt.colorbar(scatter)
    plt.title(f'Оптимальные кластеры (k={best_k})')
    plt.xlabel(f'PC1 ({pca_2d.explained_variance_ratio_[0]:.2%} дисперсии)')
    plt.ylabel(f'PC2 ({pca_2d.explained_variance_ratio_[1]:.2%} дисперсии)')
    
    plt.tight_layout()
    plt.savefig('оптимальные_кластеры.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("Все визуализации созданы успешно!")
    print("Созданные файлы:")
    print("- распределение_признаков.png")
    print("- kmeans_k3.png")
    print("- pca_анализ.png")
    print("- tsne_анализ.png")
    print("- tsne_кластеры.png")
    print("- оптимальное_k.png")
    print("- оптимальные_кластеры.png")

if __name__ == "__main__":
    main()