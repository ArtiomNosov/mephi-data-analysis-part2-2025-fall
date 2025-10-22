#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ЛР8: Визуализации для кластеризации и классификации данных
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, auc
from sklearn.decomposition import PCA
from sklearn.preprocessing import label_binarize
from itertools import cycle
import warnings
warnings.filterwarnings('ignore')

# Настройка стиля графиков
plt.style.use('default')
sns.set_palette("husl")

def create_visualizations():
    print("Создание визуализаций для ЛР8...")
    
    # Загружаем данные
    wine = datasets.load_wine()
    df = pd.DataFrame(wine.data, columns=wine.feature_names)
    df['target'] = wine.target
    
    X = df.drop('target', axis=1)
    y = df['target']
    
    # Стандартизация
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Разделение данных
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42, stratify=y)
    
    # 1. Распределение признаков
    feature_columns = df.columns[:-1]  # исключаем target
    n_features = len(feature_columns)
    n_cols = 4
    n_rows = (n_features + n_cols - 1) // n_cols  # округляем вверх
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 4*n_rows))
    axes = axes.ravel()
    
    for i, column in enumerate(feature_columns):
        axes[i].hist(df[column], bins=20, alpha=0.7)
        axes[i].set_title(f'Распределение {column}')
        axes[i].set_xlabel(column)
        axes[i].set_ylabel('Частота')
    
    # Скрываем лишние подграфики
    for i in range(n_features, len(axes)):
        axes[i].set_visible(False)
    
    plt.tight_layout()
    plt.savefig('/workspace/solutions/LR-s/LR8/feature_distributions.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Корреляционная матрица
    plt.figure(figsize=(12, 10))
    correlation_matrix = df.corr()
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0, fmt='.2f')
    plt.title('Корреляционная матрица признаков')
    plt.tight_layout()
    plt.savefig('/workspace/solutions/LR-s/LR8/correlation_matrix.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. Кластеризация
    # K-means
    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    kmeans_labels = kmeans.fit_predict(X_scaled)
    
    # Иерархическая
    hierarchical = AgglomerativeClustering(n_clusters=3)
    hierarchical_labels = hierarchical.fit_predict(X_scaled)
    
    # DBSCAN
    dbscan = DBSCAN(eps=0.5, min_samples=5)
    dbscan_labels = dbscan.fit_predict(X_scaled)
    
    # PCA для визуализации
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)
    
    # Визуализация кластеров
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Истинные классы
    scatter = axes[0, 0].scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap='viridis', alpha=0.7)
    axes[0, 0].set_title('Истинные классы')
    axes[0, 0].set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%} variance)')
    axes[0, 0].set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%} variance)')
    plt.colorbar(scatter, ax=axes[0, 0])
    
    # K-means
    scatter = axes[0, 1].scatter(X_pca[:, 0], X_pca[:, 1], c=kmeans_labels, cmap='viridis', alpha=0.7)
    axes[0, 1].set_title('K-means кластеризация')
    axes[0, 1].set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%} variance)')
    axes[0, 1].set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%} variance)')
    plt.colorbar(scatter, ax=axes[0, 1])
    
    # Иерархическая
    scatter = axes[1, 0].scatter(X_pca[:, 0], X_pca[:, 1], c=hierarchical_labels, cmap='viridis', alpha=0.7)
    axes[1, 0].set_title('Иерархическая кластеризация')
    axes[1, 0].set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%} variance)')
    axes[1, 0].set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%} variance)')
    plt.colorbar(scatter, ax=axes[1, 0])
    
    # DBSCAN
    scatter = axes[1, 1].scatter(X_pca[:, 0], X_pca[:, 1], c=dbscan_labels, cmap='viridis', alpha=0.7)
    axes[1, 1].set_title('DBSCAN кластеризация')
    axes[1, 1].set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%} variance)')
    axes[1, 1].set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%} variance)')
    plt.colorbar(scatter, ax=axes[1, 1])
    
    plt.tight_layout()
    plt.savefig('/workspace/solutions/LR-s/LR8/clustering_results.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 4. Классификация
    classifiers = {
        'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
        'SVM': SVC(random_state=42, probability=True),
        'Random Forest': RandomForestClassifier(random_state=42, n_estimators=100)
    }
    
    results = {}
    
    for name, classifier in classifiers.items():
        classifier.fit(X_train, y_train)
        y_pred = classifier.predict(X_test)
        y_pred_proba = classifier.predict_proba(X_test) if hasattr(classifier, 'predict_proba') else None
        accuracy = accuracy_score(y_test, y_pred)
        
        results[name] = {
            'classifier': classifier,
            'y_pred': y_pred,
            'y_pred_proba': y_pred_proba,
            'accuracy': accuracy
        }
    
    # Матрицы ошибок
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    for i, (name, result) in enumerate(results.items()):
        cm = confusion_matrix(y_test, result['y_pred'])
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[i])
        axes[i].set_title(f'Матрица ошибок - {name}')
        axes[i].set_xlabel('Предсказанный класс')
        axes[i].set_ylabel('Истинный класс')
    
    plt.tight_layout()
    plt.savefig('/workspace/solutions/LR-s/LR8/confusion_matrices.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 5. ROC-кривые
    y_test_bin = label_binarize(y_test, classes=[0, 1, 2])
    n_classes = y_test_bin.shape[1]
    
    plt.figure(figsize=(12, 8))
    colors = cycle(['blue', 'red', 'green'])
    
    for name, result in results.items():
        if result['y_pred_proba'] is not None:
            y_score = result['y_pred_proba']
            
            # Вычисление ROC для каждого класса
            fpr = dict()
            tpr = dict()
            roc_auc = dict()
            
            for i in range(n_classes):
                fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], y_score[:, i])
                roc_auc[i] = auc(fpr[i], tpr[i])
            
            # Средняя ROC
            fpr["micro"], tpr["micro"], _ = roc_curve(y_test_bin.ravel(), y_score.ravel())
            roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
            
            plt.plot(fpr["micro"], tpr["micro"], color=next(colors),
                    label=f'{name} (AUC = {roc_auc["micro"]:.2f})')
    
    plt.plot([0, 1], [0, 1], 'k--', lw=2, label='Случайный классификатор')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Доля ложноположительных результатов')
    plt.ylabel('Доля истинноположительных результатов')
    plt.title('ROC-кривые')
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    plt.savefig('/workspace/solutions/LR-s/LR8/roc_curves.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 6. Важность признаков
    rf_classifier = results['Random Forest']['classifier']
    feature_importance = rf_classifier.feature_importances_
    
    feature_importance_df = pd.DataFrame({
        'feature': X.columns,
        'importance': feature_importance
    }).sort_values('importance', ascending=False)
    
    plt.figure(figsize=(10, 8))
    sns.barplot(data=feature_importance_df, x='importance', y='feature')
    plt.title('Важность признаков (Random Forest)')
    plt.xlabel('Важность')
    plt.tight_layout()
    plt.savefig('/workspace/solutions/LR-s/LR8/feature_importance.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 7. Сравнение точности
    accuracy_comparison = pd.DataFrame({
        'Классификатор': list(results.keys()),
        'Точность': [result['accuracy'] for result in results.values()]
    })
    
    plt.figure(figsize=(10, 6))
    sns.barplot(data=accuracy_comparison, x='Классификатор', y='Точность')
    plt.title('Сравнение точности классификаторов')
    plt.ylabel('Точность')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('/workspace/solutions/LR-s/LR8/accuracy_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("Все визуализации созданы и сохранены в папку LR8/")
    print("Созданные файлы:")
    print("- feature_distributions.png")
    print("- correlation_matrix.png")
    print("- clustering_results.png")
    print("- confusion_matrices.png")
    print("- roc_curves.png")
    print("- feature_importance.png")
    print("- accuracy_comparison.png")

if __name__ == "__main__":
    create_visualizations()