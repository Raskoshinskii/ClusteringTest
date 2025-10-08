import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score
from sklearn.manifold import TSNE
from scipy.cluster.hierarchy import dendrogram, linkage
import datetime
import warnings
warnings.filterwarnings('ignore')

# Настройка страницы
st.set_page_config(
    page_title="Иерархическая кластеризация",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Заголовок приложения
st.title("🔍 Иерархическая кластеризация данных")
st.markdown("---")

# Sidebar для настроек
st.sidebar.header("⚙️ Настройки анализа")

def load_data(uploaded_file):
    """Загрузка и предварительная обработка данных"""
    if uploaded_file is not None:
        try:
            data = pd.read_excel(uploaded_file)
            return data
        except Exception as e:
            st.error(f"Ошибка при загрузке файла: {e}")
            return None
    return None

def preprocess_data(data, drop_columns=None):
    """Предобработка данных"""
    if drop_columns:
        data = data.drop(drop_columns, axis=1, errors='ignore')
    
    # Переименование целевой переменной
    if 'Количество (Норматив)' in data.columns:
        data = data.rename(columns={'Количество (Норматив)': 'target'})
    
    return data

def get_column_types(data):
    """Определение типов колонок"""
    num_col = data.select_dtypes(include='number').columns.tolist()
    cat_col = data.select_dtypes(exclude='number').columns.tolist()
    date_col = []
    
    # Поиск колонок с датами
    for col in data.columns:
        if data[col].dtype == 'object':
            try:
                pd.to_datetime(data[col], errors='raise')
                date_col.append(col)
                if col in cat_col:
                    cat_col.remove(col)
            except:
                pass
    
    return num_col, cat_col, date_col

def compute_wcss(X, labels):
    """Вычисление WCSS для метода локтя"""
    if hasattr(X, "todense"):
        X = np.asarray(X.todense())
    wcss = 0
    for cluster in np.unique(labels):
        cluster_points = X[labels == cluster]
        centroid = cluster_points.mean(axis=0)
        wcss += ((cluster_points - centroid) ** 2).sum()
    return wcss

def main():
    # Загрузка данных
    st.sidebar.subheader("📁 Загрузка данных")
    uploaded_file = st.sidebar.file_uploader(
        "Выберите Excel файл",
        type=['xlsx', 'xls'],
        help="Загрузите файл с данными для анализа"
    )

    # Загрузка данных по умолчанию
    use_sample = st.sidebar.checkbox("Использовать данные по умолчанию", value=True)
    
    if use_sample and uploaded_file is None:
        try:
            data = pd.read_excel("data/sverstal_data.xlsx")
            st.sidebar.success("✅ Загружены данные по умолчанию")
        except:
            st.sidebar.error("❌ данные по умолчанию не найдены")
            data = None
    else:
        data = load_data(uploaded_file)
    
    if data is not None:
        # Предобработка данных
        data = preprocess_data(data, ['Базовая ЕИ', 'VID_PROD_GP'])
        
        # Основная информация о данных
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Количество строк", data.shape[0])
        with col2:
            st.metric("Количество столбцов", data.shape[1])
        with col3:
            st.metric("Пропущенные значения", data.isnull().sum().sum())
        
        # Отображение данных
        with st.expander("📊 Просмотр данных", expanded=False):
            st.dataframe(data.head(10))
            
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("Статистика числовых признаков")
                st.dataframe(data.describe())
            with col2:
                st.subheader("Информация о данных")
                buffer = []
                for col in data.columns:
                    buffer.append({
                        'Столбец': col,
                        'Тип': str(data[col].dtype),
                        'Уникальных значений': data[col].nunique(),
                        'Пропуски': data[col].isnull().sum()
                    })
                st.dataframe(pd.DataFrame(buffer))
        
        # Определение типов колонок
        num_col, cat_col, date_col = get_column_types(data)
        
        # Фильтрация по дате
        st.sidebar.subheader("📅 Фильтрация по дате")
        if date_col:
            selected_date_col = st.sidebar.selectbox("Выберите колонку с датой", date_col)
            
            if selected_date_col:
                data[selected_date_col] = pd.to_datetime(data[selected_date_col])
                min_date = data[selected_date_col].min()
                max_date = data[selected_date_col].max()
                
                date_range = st.sidebar.date_input(
                    "Выберите диапазон дат",
                    value=(min_date, max_date),
                    min_value=min_date,
                    max_value=max_date
                )
                
                if len(date_range) == 2:
                    data = data[
                        (data[selected_date_col] >= pd.to_datetime(date_range[0])) &
                        (data[selected_date_col] <= pd.to_datetime(date_range[1]))
                    ]
                    st.sidebar.success(f"Отфильтровано: {len(data)} записей")
        else:
            st.sidebar.info("Колонки с датами не найдены")
        
        # Выбор признаков
        st.sidebar.subheader("🎯 Выбор признаков")
        
        # Исключаем ID колонку из числовых признаков
        if 'Номер Z-конфигурации' in num_col:
            analysis_num_col = [col for col in num_col if col != 'Номер Z-конфигурации']
        else:
            analysis_num_col = num_col
        
        feature_selection = st.sidebar.radio(
            "Режим выбора признаков",
            ["Все признаки", "Выбрать конкретные признаки"]
        )
        
        if feature_selection == "Выбрать конкретные признаки":
            all_features = analysis_num_col + cat_col
            selected_features = st.sidebar.multiselect(
                "Выберите признаки (2-3 рекомендуется)",
                all_features,
                default=all_features[:3] if len(all_features) >= 3 else all_features
            )
            
            if len(selected_features) < 2:
                st.sidebar.warning("Выберите минимум 2 признака")
                return
        else:
            selected_features = analysis_num_col + cat_col
        
        # Выбор количества кластеров
        st.sidebar.subheader("🎲 Параметры кластеризации")
        n_clusters = st.sidebar.slider(
            "Количество кластеров",
            min_value=2,
            max_value=min(20, len(data)-1),
            value=5,
            help="Выберите количество кластеров для анализа"
        )
        
        # Кнопка запуска анализа
        if st.sidebar.button("🚀 Запустить анализ", type="primary"):
            perform_clustering_analysis(data, selected_features, analysis_num_col, cat_col, n_clusters)
    else:
        st.info("👆 Загрузите файл данных или используйте данные по умолчанию")

def perform_clustering_analysis(data, selected_features, num_col, cat_col, n_clusters):
    """Выполнение кластерного анализа"""
    
    # Подготовка данных для кластеризации
    selected_num_col = [col for col in selected_features if col in num_col]
    selected_cat_col = [col for col in selected_features if col in cat_col]
    
    # Создание препроцессора
    transformers = []
    if selected_num_col:
        transformers.append(("num", StandardScaler(), selected_num_col))
    if selected_cat_col:
        transformers.append(("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), selected_cat_col))
    
    if not transformers:
        st.error("Не выбраны признаки для анализа")
        return
    
    preprocessor = ColumnTransformer(transformers=transformers)
    
    try:
        X = preprocessor.fit_transform(data)
        
        # Метод локтя и силуэт
        st.subheader("📈 Анализ оптимального количества кластеров")
        
        with st.spinner("Вычисление метрик..."):
            k_range = range(2, min(15, len(data)))
            wcss_list = []
            silhouette_list = []
            
            for k in k_range:
                model = AgglomerativeClustering(n_clusters=k, linkage="ward")
                labels = model.fit_predict(X)
                wcss_list.append(compute_wcss(X, labels))
                silhouette_list.append(silhouette_score(X, labels))
        
        # Создание графиков
        col1, col2 = st.columns(2)
        
        with col1:
            fig_elbow = go.Figure()
            fig_elbow.add_trace(go.Scatter(
                x=list(k_range),
                y=wcss_list,
                mode='lines+markers',
                name='WCSS',
                line=dict(color='blue', width=2),
                marker=dict(size=8)
            ))
            fig_elbow.update_layout(
                title='Метод локтя',
                xaxis_title='Количество кластеров (k)',
                yaxis_title='WCSS',
                height=400
            )
            st.plotly_chart(fig_elbow, use_container_width=True)
        
        with col2:
            fig_silhouette = go.Figure()
            fig_silhouette.add_trace(go.Scatter(
                x=list(k_range),
                y=silhouette_list,
                mode='lines+markers',
                name='Silhouette Score',
                line=dict(color='red', width=2),
                marker=dict(size=8)
            ))
            fig_silhouette.update_layout(
                title='Анализ силуэта',
                xaxis_title='Количество кластеров (k)',
                yaxis_title='Silhouette Score',
                height=400
            )
            st.plotly_chart(fig_silhouette, use_container_width=True)
        
        # Финальная кластеризация
        st.subheader(f"🎯 Результаты кластеризации ({n_clusters} кластеров)")
        
        final_model = AgglomerativeClustering(n_clusters=n_clusters, linkage="ward")
        final_labels = final_model.fit_predict(X)
        
        # Добавление результатов в данные
        data_with_clusters = data.copy()
        data_with_clusters['cluster'] = final_labels
        
        # Дендрограмма
        st.subheader("🌳 Дендрограмма")
        with st.spinner("Построение дендрограммы..."):
            Z = linkage(X.toarray() if hasattr(X, "toarray") else X, method="ward")
            
            # Создание дендрограммы с matplotlib/seaborn
            fig, ax = plt.subplots(figsize=(14, 8))
            
            # Подготовка меток и параметров для больших датасетов
            labels_for_dendrogram = None
            truncate_mode = None
            p = 30  # Максимальное количество листьев для отображения
            
            if 'Номер Z-конфигурации' in data.columns:
                labels_for_dendrogram = data['Номер Z-конфигурации'].astype(str).values
                
            # Для больших датасетов используем усечение
            if len(data) > 50:
                truncate_mode = 'lastp'
                labels_for_dendrogram = None  # Отключаем метки для читаемости
                st.info(f"📊 Для улучшения читаемости дендрограмма усечена до {p} кластеров")
            
            # Построение дендрограммы
            dendrogram_plot = dendrogram(
                Z,
                labels=labels_for_dendrogram,
                leaf_rotation=90,
                leaf_font_size=8,
                ax=ax,
                color_threshold=0.7*max(Z[:,2]),  # Цветовой порог для раскраски
                above_threshold_color='gray',
                truncate_mode=truncate_mode,
                p=p if truncate_mode else None
            )
            
            ax.set_title('Дендрограмма (Ward linkage)', fontsize=16, fontweight='bold')
            ax.set_xlabel('Объекты/Кластеры', fontsize=12)
            ax.set_ylabel('Расстояние', fontsize=12)
            ax.grid(True, alpha=0.3, linestyle='--')
                        
            # Улучшение внешнего вида
            plt.tight_layout()
            
            # Отображение в Streamlit
            st.pyplot(fig)
            plt.close()  # Освобождение памяти
            
            # Дополнительная информация
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Количество объектов", len(data))
            with col2:
                st.metric("Максимальное расстояние", f"{max(Z[:,2]):.2f}")
        
        # t-SNE визуализация
        st.subheader("🗺️ Визуализация в пространстве признаков (t-SNE)")
        with st.spinner("Выполнение t-SNE..."):
            # Адаптивный расчет perplexity
            perplexity = max(3, min(30, (len(data) - 1) // 3))
            tsne = TSNE(n_components=2, random_state=42, perplexity=perplexity, 
                       learning_rate='auto', init='random')
            X_embedded = tsne.fit_transform(X.toarray() if hasattr(X, "toarray") else X)
            
            # Создание DataFrame для plotly
            tsne_df = pd.DataFrame({
                'x': X_embedded[:, 0],
                'y': X_embedded[:, 1],
                'cluster': final_labels.astype(str),  # Преобразуем в строки для категорий
                'index': range(len(final_labels))
            })
            
            if 'Номер Z-конфигурации' in data.columns:
                tsne_df['id'] = data['Номер Z-конфигурации'].values
                hover_data = ['id', 'cluster']
            else:
                hover_data = ['index', 'cluster']
            
            # Создание цветовой схемы
            try:
                color_sequence = px.colors.qualitative.Set1
            except:
                # Резервная цветовая схема
                color_sequence = px.colors.qualitative.Plotly
            
            fig_tsne = px.scatter(
                tsne_df,
                x='x',
                y='y',
                color='cluster',
                title='Объекты в пространстве признаков (t-SNE)',
                labels={'x': 't-SNE 1', 'y': 't-SNE 2'},
                hover_data=hover_data,
                color_discrete_sequence=color_sequence
            )
            fig_tsne.update_layout(height=600)
            st.plotly_chart(fig_tsne, use_container_width=True)
        
        # Статистика по кластерам
        st.subheader("📊 Статистика по кластерам")
        cluster_stats = data_with_clusters.groupby('cluster').agg({
            selected_features[0]: ['count', 'mean', 'std'] if selected_features else 'count'
        }).round(2)
        
        st.dataframe(cluster_stats)
        
        # Результаты
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Количество кластеров", n_clusters)
            st.metric("Silhouette Score", f"{silhouette_score(X, final_labels):.3f}")
        with col2:
            st.metric("Использованных признаков", len(selected_features))
            st.metric("WCSS", f"{compute_wcss(X, final_labels):.0f}")
        
        # Возможность скачать результаты
        csv = data_with_clusters.to_csv(index=False)
        st.download_button(
            label="📥 Скачать результаты",
            data=csv,
            file_name=f'clustering_results_{n_clusters}_clusters.csv',
            mime='text/csv'
        )
        
    except Exception as e:
        st.error(f"Ошибка при выполнении анализа: {e}")

if __name__ == "__main__":
    main()