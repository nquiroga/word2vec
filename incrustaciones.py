import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
from collections import defaultdict, Counter
import warnings
from gensim.models import Word2Vec
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

warnings.filterwarnings('ignore')

# Configuración de la página
st.set_page_config(
    page_title="Análisis Word2Vec - Discursos Presidenciales",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Configuración para visualizaciones
plt.style.use('default')
sns.set_palette("husl")

# Períodos históricos
PERIODOS = [
    (1850, 1899, "Organización Nacional (1850-1899)"),
    (1900, 1945, "República Conservadora y Radicalismo (1900-1945)"),
    (1946, 1975, "del peronismo al peronismo (1946-1975)"),
    (1976, 2000, "Dictadura y Transición (1976-2000)"),
    (2001, 2025, "Siglo XXI (2001-2025)")
]

@st.cache_data
def cargar_datos():
    """Carga los datos de discursos presidenciales."""
    try:
        df_discursos = pd.read_csv('discursos_presidenciales_limpios.csv')
        df_discursos['anio'] = df_discursos['anio'].astype(int)
        df_discursos['presidente'] = df_discursos['presidente'].str.replace('.', '')
        return df_discursos
    except Exception as e:
        st.error(f"Error cargando datos: {e}")
        return None

@st.cache_data
def entrenar_modelos_por_periodo(_df, vector_size=300, window=10, min_count=3):
    """Entrena modelos Word2Vec para cada período histórico."""
    modelos_por_periodo = {}
    estadisticas_corpus = {}
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for i, (inicio, fin, nombre_periodo) in enumerate(PERIODOS):
        status_text.text(f"Entrenando modelo para: {nombre_periodo}")
        
        # Filtrar discursos del período
        mask = (_df['anio'] >= inicio) & (_df['anio'] <= fin)
        df_periodo = _df[mask].copy()
        
        if len(df_periodo) == 0:
            continue
            
        # Estadísticas del corpus
        estadisticas_corpus[nombre_periodo] = {
            'discursos': len(df_periodo),
            'presidentes': df_periodo['presidente'].nunique(),
            'años': df_periodo['anio'].nunique()
        }
        
        # Preparar corpus
        textos_periodo = df_periodo['texto_limpio'].tolist()
        corpus_tokens = [texto.split() for texto in textos_periodo if isinstance(texto, str)]
        
        if corpus_tokens:
            try:
                # Entrenar Word2Vec
                modelo = Word2Vec(
                    sentences=corpus_tokens,
                    vector_size=vector_size,
                    window=window,
                    min_count=min_count,
                    workers=4,
                    epochs=20,
                    seed=42
                )
                modelos_por_periodo[nombre_periodo] = modelo
            except Exception as e:
                st.warning(f"Error entrenando modelo para {nombre_periodo}: {e}")
        
        progress_bar.progress((i + 1) / len(PERIODOS))
    
    status_text.empty()
    progress_bar.empty()
    
    return modelos_por_periodo, estadisticas_corpus

def analizar_similitudes_por_periodos(palabra_objetivo, modelos_por_periodo, n_similares=10):
    """Analiza similitudes de una palabra a través de períodos históricos."""
    similitudes_por_periodo = {}
    
    for nombre_periodo, modelo in modelos_por_periodo.items():
        if palabra_objetivo in modelo.wv:
            similares = modelo.wv.most_similar(palabra_objetivo, topn=n_similares)
            similitudes_por_periodo[nombre_periodo] = similares
        else:
            similitudes_por_periodo[nombre_periodo] = []
    
    # Crear tabla de resultados
    nombres_periodos = list(modelos_por_periodo.keys())
    tabla_resultados = pd.DataFrame(
        index=range(1, n_similares + 1),
        columns=nombres_periodos
    )
    tabla_resultados.index.name = 'Ranking'
    
    # Llenar la tabla
    for periodo in nombres_periodos:
        if similitudes_por_periodo[periodo]:
            for i, (palabra, score) in enumerate(similitudes_por_periodo[periodo], 1):
                tabla_resultados.loc[i, periodo] = f"{palabra} ({score:.3f})"
        else:
            for i in range(1, n_similares + 1):
                tabla_resultados.loc[i, periodo] = "--- sin datos ---"
    
    return tabla_resultados, similitudes_por_periodo

def crear_grafico_similitudes(palabra_objetivo, modelos_por_periodo, tabla_similitudes):
    """Crea gráfico interactivo de similitudes por período."""
    datos_grafico = []
    
    # Procesar datos
    for i, col_periodo in enumerate(tabla_similitudes.columns):
        if col_periodo in modelos_por_periodo:
            modelo = modelos_por_periodo[col_periodo]
            
            for ranking in tabla_similitudes.index:
                celda = tabla_similitudes.loc[ranking, col_periodo]
                
                if isinstance(celda, str) and celda.strip() and "sin datos" not in celda.lower():
                    # Extraer palabra del formato "palabra (score)"
                    palabra = celda.split(' (')[0].strip()
                    
                    try:
                        if palabra_objetivo in modelo.wv and palabra in modelo.wv:
                            score = modelo.wv.similarity(palabra_objetivo, palabra)
                            
                            # Añadir jitter para separar puntos horizontalmente
                            x_jitter = np.random.normal(0, 0.1)
                            
                            datos_grafico.append({
                                'periodo': col_periodo,
                                'periodo_corto': col_periodo.split('(')[0].strip(),
                                'periodo_index': i + x_jitter,  # Posición con jitter
                                'ranking': ranking,
                                'palabra': palabra,
                                'similitud': score
                            })
                    except Exception:
                        continue
    
    if not datos_grafico:
        return None
    
    df_datos = pd.DataFrame(datos_grafico)
    
    # Crear gráfico con Plotly
    fig = go.Figure()
    
    # Añadir puntos con colores por período
    periodos_unicos = df_datos['periodo_corto'].unique()
    colors = px.colors.qualitative.Set3[:len(periodos_unicos)]
    
    for i, periodo in enumerate(periodos_unicos):
        datos_periodo = df_datos[df_datos['periodo_corto'] == periodo]
        
        fig.add_trace(go.Scatter(
            x=datos_periodo['periodo_index'],
            y=datos_periodo['similitud'],
            mode='markers',
            marker=dict(
                size=12,
                color=colors[i % len(colors)],
                opacity=0.8,
                line=dict(width=1, color='black')
            ),
            name=periodo,
            text=datos_periodo['palabra'],
            customdata=datos_periodo['ranking'],
            hovertemplate='<b>%{text}</b><br>' +
                         'Similitud: %{y:.3f}<br>' +
                         'Ranking: %{customdata}<br>' +
                         'Período: ' + periodo + '<extra></extra>',
            showlegend=False
        ))
    
    # Añadir línea de similitud media
    similitud_media = df_datos['similitud'].mean()
    fig.add_hline(
        y=similitud_media, 
        line_dash="dash", 
        line_color="red",
        opacity=0.7,
        annotation_text=f"Similitud Media: {similitud_media:.3f}",
        annotation_position="top right"
    )
    
    # Configurar layout
    fig.update_layout(
        title=dict(
            text=f'Similitudes Word2Vec por Período Histórico<br>Palabra: "{palabra_objetivo.upper()}"',
            x=0.5,
            font=dict(size=16)
        ),
        xaxis=dict(
            title="Período Histórico",
            tickmode='array',
            tickvals=list(range(len(periodos_unicos))),
            ticktext=periodos_unicos,
            tickangle=-45
        ),
        yaxis=dict(
            title="Similitud Coseno (Word2Vec)",
            gridcolor="lightgray",
            gridwidth=1
        ),
        width=1000,
        height=800,  # Altura aumentada
        plot_bgcolor="rgba(250,250,250,0.8)",
        margin=dict(l=60, r=60, t=100, b=120),
        hovermode='closest'  # Mejor detección del hover
    )
    
    return fig, df_datos

def main():
    st.title("📊 Análisis Semántico Word2Vec de Discursos Presidenciales")
    st.markdown("### Evolución de similitudes semánticas a través de períodos históricos argentinos")
    
    # Sidebar para configuración
    st.sidebar.header("⚙️ Configuración")
    
    # Cargar datos
    with st.spinner("Cargando datos de discursos..."):
        df_discursos = cargar_datos()
    
    if df_discursos is None:
        st.error("No se pudieron cargar los datos. Verifica la conexión a internet.")
        return
    
    # Mostrar información básica de los datos
    st.sidebar.markdown("### 📈 Información del Dataset")
    st.sidebar.info(f"""
    **Total de discursos:** {len(df_discursos):,}
    
    **Rango temporal:** {df_discursos['anio'].min()} - {df_discursos['anio'].max()}
    
    **Presidentes únicos:** {df_discursos['presidente'].nunique()}
    """)
    
    # Parámetros fijos del modelo (según código original)
    vector_size = 300
    window = 10
    min_count = 3
    n_similares = 10
    
    # Entrenar modelos
    if 'modelos_entrenados' not in st.session_state:
        with st.spinner("Entrenando modelos Word2Vec por período..."):
            modelos, estadisticas = entrenar_modelos_por_periodo(
                df_discursos, vector_size, window, min_count
            )
            st.session_state.modelos_entrenados = modelos
            st.session_state.estadisticas_corpus = estadisticas
    
    modelos = st.session_state.modelos_entrenados
    estadisticas = st.session_state.estadisticas_corpus
    
    if not modelos:
        st.error("No se pudieron entrenar los modelos Word2Vec.")
        return
    
    st.success(f"✅ Modelos entrenados exitosamente para {len(modelos)} períodos")
    
    # Input para palabra objetivo
    st.markdown("### 🎯 Análisis de Palabra")
    palabra_objetivo = st.text_input(
        "Ingresa la palabra a analizar:",
        value="crisis",
        placeholder="Ej: pueblo, democracia, libertad, crisis..."
    ).lower().strip()
    
    if palabra_objetivo:
        # Verificar disponibilidad en vocabularios
        periodos_disponibles = []
        for periodo, modelo in modelos.items():
            if palabra_objetivo in modelo.wv:
                periodos_disponibles.append(periodo)
        
        if not periodos_disponibles:
            st.warning(f"⚠️ La palabra '{palabra_objetivo}' no se encontró en ningún vocabulario de los períodos.")
            
            # Sugerir palabras similares
            st.markdown("### 💡 Sugerencias de palabras disponibles")
            todas_palabras = set()
            for modelo in modelos.values():
                todas_palabras.update(list(modelo.wv.index_to_key)[:1000])  # Top 1000 palabras por modelo
            
            palabras_sugeridas = [p for p in todas_palabras if palabra_objetivo in p][:10]
            if palabras_sugeridas:
                st.write("Palabras que contienen tu búsqueda:")
                st.write(", ".join(palabras_sugeridas))
            
            return
        
        st.info(f"📊 Palabra '{palabra_objetivo}' encontrada en {len(periodos_disponibles)} de {len(modelos)} períodos")
        
        # Analizar similitudes
        with st.spinner("Analizando similitudes semánticas..."):
            tabla, similitudes = analizar_similitudes_por_periodos(
                palabra_objetivo, modelos, n_similares
            )
        
        # Mostrar resultados en tabs
        tab1, tab2, tab3 = st.tabs(["📊 Gráfico", "📋 Tabla de Similitudes", "📈 Estadísticas"])
        
        with tab1:
            st.markdown(f"### Evolución Semántica de '{palabra_objetivo.upper()}'")
            
            fig, datos = crear_grafico_similitudes(palabra_objetivo, modelos, tabla)
            
            if fig is not None:
                st.plotly_chart(fig, use_container_width=True)
                
                # Análisis de patrones
                st.markdown("#### 🔍 Análisis de Patrones")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("**🏆 Palabra más similar por período:**")
                    for periodo in datos['periodo_corto'].unique():
                        datos_periodo = datos[datos['periodo_corto'] == periodo]
                        if not datos_periodo.empty:
                            mejor = datos_periodo.loc[datos_periodo['similitud'].idxmax()]
                            st.write(f"• **{periodo}:** {mejor['palabra']} ({mejor['similitud']:.3f})")
                
                with col2:
                    st.markdown("**🔄 Palabras recurrentes:**")
                    contador_palabras = datos['palabra'].value_counts()
                    palabras_recurrentes = contador_palabras[contador_palabras > 1]
                    
                    if not palabras_recurrentes.empty:
                        for palabra, freq in palabras_recurrentes.items():
                            st.write(f"• **{palabra}:** aparece en {freq} períodos")
                    else:
                        st.write("No hay palabras recurrentes entre períodos")
            else:
                st.error("No se pudieron generar datos para el gráfico.")
        
        with tab2:
            st.markdown(f"### Tabla de Similitudes para '{palabra_objetivo.upper()}'")
            st.markdown("*Formato: palabra (similitud_coseno)*")
            
            # Formatear tabla para mejor visualización
            tabla_display = tabla.copy()
            st.dataframe(tabla_display, use_container_width=True)
            
            # Opción de descarga
            csv = tabla.to_csv(index=True)
            st.download_button(
                label="📥 Descargar tabla como CSV",
                data=csv,
                file_name=f"similitudes_{palabra_objetivo}_{pd.Timestamp.now().strftime('%Y%m%d_%H%M')}.csv",
                mime="text/csv"
            )
        
        with tab3:
            st.markdown("### 📈 Estadísticas del Corpus por Período")
            
            # Mostrar estadísticas del corpus
            if estadisticas:
                stats_df = pd.DataFrame(estadisticas).T
                st.dataframe(stats_df, use_container_width=True)
                
                # Gráfico de distribución de discursos
                fig_stats = px.bar(
                    x=stats_df.index,
                    y=stats_df['discursos'],
                    title="Distribución de Discursos por Período",
                    labels={'x': 'Período', 'y': 'Número de Discursos'}
                )
                fig_stats.update_xaxes(tickangle=-45)
                st.plotly_chart(fig_stats, use_container_width=True)
    
    # Footer
    st.markdown("---")
    st.markdown("""
    **Metodología:** Este análisis utiliza modelos Word2Vec entrenados independientemente para cada período histórico,
    permitiendo observar la evolución semántica de las palabras en el discurso político argentino.
    
    **Fuente de datos:** Discursos presidenciales argentinos (1850-2025)
    """)

if __name__ == "__main__":
    main()
