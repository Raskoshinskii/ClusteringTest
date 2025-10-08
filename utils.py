"""
Дополнительные утилиты для Streamlit приложения иерархической кластеризации
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import silhouette_samples
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram, linkage

def create_enhanced_dendrogram(X, data, method='ward', figsize=(12, 8)):
    """
    Создание улучшенной дендрограммы с использованием matplotlib/seaborn
    """
    # Вычисление связей
    Z = linkage(X.toarray() if hasattr(X, "toarray") else X, method=method)
    
    # Создание фигуры
    fig, ax = plt.subplots(figsize=figsize)
    
    # Подготовка меток
    labels_for_dendrogram = None
    if 'Номер Z-конфигурации' in data.columns:
        labels_for_dendrogram = data['Номер Z-конфигурации'].astype(str).values
        # Ограничиваем количество меток для читаемости
        if len(labels_for_dendrogram) > 30:
            labels_for_dendrogram = None
    
    # Построение дендрограммы
    dendrogram_plot = dendrogram(
        Z,
        labels=labels_for_dendrogram,
        leaf_rotation=90,
        leaf_font_size=10,
        ax=ax,
        color_threshold=0.7*max(Z[:,2]),  # Цветовой порог
        above_threshold_color='gray'
    )
    
    # Улучшение внешнего вида
    ax.set_title(f'Дендрограмма ({method.title()} linkage)', fontsize=16, fontweight='bold')
    ax.set_xlabel('Объекты', fontsize=14)
    ax.set_ylabel('Расстояние', fontsize=14)
    ax.grid(True, alpha=0.3, linestyle='--')
    
    # Добавляем горизонтальную линию для показа возможной точки отсечения
    threshold = 0.7 * max(Z[:,2])
    ax.axhline(y=threshold, color='red', linestyle='--', alpha=0.7, 
               label=f'Порог отсечения: {threshold:.2f}')
    ax.legend()
    
    plt.tight_layout()
    return fig

def create_silhouette_plot(X, labels, n_clusters):
    """
    Создание детального графика силуэта для каждого кластера
    """
    silhouette_scores = silhouette_samples(X, labels)
    
    fig = go.Figure()
    
    y_lower = 10
    colors = px.colors.qualitative.Set1[:n_clusters]
    
    for i in range(n_clusters):
        cluster_silhouette_scores = silhouette_scores[labels == i]
        cluster_silhouette_scores.sort()
        
        size_cluster_i = cluster_silhouette_scores.shape[0]
        y_upper = y_lower + size_cluster_i
        
        fig.add_trace(go.Scatter(
            x=cluster_silhouette_scores,
            y=np.arange(y_lower, y_upper),
            mode='lines',
            fill='tozeroy',
            name=f'Кластер {i}',
            line=dict(color=colors[i % len(colors)]),
            fillcolor=colors[i % len(colors)]
        ))
        
        y_lower = y_upper + 10
    
    avg_score = silhouette_scores.mean()
    fig.add_vline(x=avg_score, line_dash="dash", line_color="red",
                  annotation_text=f"Средний силуэт: {avg_score:.3f}")
    
    fig.update_layout(
        title=f'Анализ силуэта для {n_clusters} кластеров',
        xaxis_title='Значение силуэта',
        yaxis_title='Индекс объекта',
        height=400,
        showlegend=True
    )
    
    return fig

def analyze_cluster_characteristics(data, labels, selected_features):
    """
    Анализ характеристик кластеров
    """
    data_with_clusters = data.copy()
    data_with_clusters['cluster'] = labels
    
    # Статистика по кластерам
    cluster_summary = []
    
    for cluster_id in sorted(data_with_clusters['cluster'].unique()):
        cluster_data = data_with_clusters[data_with_clusters['cluster'] == cluster_id]
        
        summary = {
            'Кластер': cluster_id,
            'Размер': len(cluster_data),
            'Процент': f"{len(cluster_data) / len(data_with_clusters) * 100:.1f}%"
        }
        
        # Добавляем статистику по числовым признакам
        for feature in selected_features:
            if feature in data.select_dtypes(include='number').columns:
                summary[f'{feature}_mean'] = cluster_data[feature].mean()
                summary[f'{feature}_std'] = cluster_data[feature].std()
        
        cluster_summary.append(summary)
    
    return pd.DataFrame(cluster_summary)

def create_cluster_comparison_plot(data, labels, selected_features):
    """
    Создание графика сравнения кластеров по признакам
    """
    data_with_clusters = data.copy()
    data_with_clusters['cluster'] = labels
    
    numeric_features = [f for f in selected_features 
                       if f in data.select_dtypes(include='number').columns]
    
    if len(numeric_features) == 0:
        return None
    
    # Создаем subplot для каждого признака
    from plotly.subplots import make_subplots
    
    cols = min(3, len(numeric_features))
    rows = (len(numeric_features) + cols - 1) // cols
    
    fig = make_subplots(
        rows=rows, cols=cols,
        subplot_titles=numeric_features,
        vertical_spacing=0.1
    )
    
    colors = px.colors.qualitative.Set1
    
    for i, feature in enumerate(numeric_features):
        row = i // cols + 1
        col = i % cols + 1
        
        for cluster_id in sorted(data_with_clusters['cluster'].unique()):
            cluster_data = data_with_clusters[data_with_clusters['cluster'] == cluster_id]
            
            fig.add_trace(
                go.Box(
                    y=cluster_data[feature],
                    name=f'Кластер {cluster_id}',
                    marker_color=colors[cluster_id % len(colors)],
                    showlegend=(i == 0)  # Показываем легенду только для первого графика
                ),
                row=row, col=col
            )
    
    fig.update_layout(
        title='Сравнение кластеров по признакам',
        height=200 * rows,
        showlegend=True
    )
    
    return fig

def export_detailed_results(data, labels, selected_features, metrics):
    """
    Подготовка детальных результатов для экспорта
    """
    results = data.copy()
    results['cluster'] = labels
    
    # Добавляем метрики
    results_summary = {
        'Общая информация': {
            'Количество объектов': len(data),
            'Количество кластеров': len(np.unique(labels)),
            'Используемые признаки': ', '.join(selected_features),
            'Silhouette Score': metrics.get('silhouette_score', 'N/A'),
            'WCSS': metrics.get('wcss', 'N/A')
        }
    }
    
    # Статистика по кластерам
    for cluster_id in sorted(np.unique(labels)):
        cluster_data = results[results['cluster'] == cluster_id]
        results_summary[f'Кластер {cluster_id}'] = {
            'Размер': len(cluster_data),
            'Процент от общего': f"{len(cluster_data) / len(results) * 100:.1f}%"
        }
        
        # Добавляем статистику по числовым признакам
        numeric_features = [f for f in selected_features 
                           if f in data.select_dtypes(include='number').columns]
        
        for feature in numeric_features:
            results_summary[f'Кластер {cluster_id}'][f'{feature}_mean'] = cluster_data[feature].mean()
            results_summary[f'Кластер {cluster_id}'][f'{feature}_std'] = cluster_data[feature].std()
    
    return results, results_summary

@st.cache_data
def load_and_cache_data(file_path):
    """
    Кэшированная загрузка данных
    """
    return pd.read_excel(file_path)

def validate_data_quality(data):
    """
    Проверка качества данных
    """
    issues = []
    
    # Проверка на пропущенные значения
    missing_data = data.isnull().sum()
    if missing_data.sum() > 0:
        issues.append(f"Обнаружены пропущенные значения: {missing_data.sum()} всего")
    
    # Проверка на дублированные строки
    duplicates = data.duplicated().sum()
    if duplicates > 0:
        issues.append(f"Обнаружены дублированные строки: {duplicates}")
    
    # Проверка на константные столбцы
    constant_cols = []
    for col in data.select_dtypes(include='number').columns:
        if data[col].nunique() <= 1:
            constant_cols.append(col)
    
    if constant_cols:
        issues.append(f"Столбцы с константными значениями: {', '.join(constant_cols)}")
    
    # Проверка размера датасета
    if len(data) < 10:
        issues.append("Слишком мало данных для надежной кластеризации (< 10 объектов)")
    
    return issues

def suggest_optimal_clusters(wcss_list, silhouette_list, k_range):
    """
    Автоматическое предложение оптимального количества кластеров
    """
    suggestions = []
    
    # Метод локтя - поиск наибольшего изменения градиента
    if len(wcss_list) > 2:
        deltas = np.diff(wcss_list)
        second_deltas = np.diff(deltas)
        if len(second_deltas) > 0:
            elbow_idx = np.argmax(second_deltas) + 2
            if elbow_idx < len(k_range):
                suggestions.append({
                    'method': 'Метод локтя',
                    'clusters': k_range[elbow_idx],
                    'reason': 'Наибольшее изменение кривизны'
                })
    
    # Максимальный силуэт
    max_silhouette_idx = np.argmax(silhouette_list)
    suggestions.append({
        'method': 'Максимальный силуэт',
        'clusters': k_range[max_silhouette_idx],
        'reason': f'Наивысший Silhouette Score: {silhouette_list[max_silhouette_idx]:.3f}'
    })
    
    # Практическое правило (sqrt(n/2))
    practical_clusters = max(2, int(np.sqrt(len(wcss_list) * 2)))
    if practical_clusters in k_range:
        suggestions.append({
            'method': 'Практическое правило',
            'clusters': practical_clusters,
            'reason': 'Эмпирическое правило √(n/2)'
        })
    
    return suggestions