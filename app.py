import pickle
from sklearn import preprocessing
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import plotly.express as px
import plotly.graph_objects as go
import xgboost as xgb
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

fixed_color_freq_map = {
    'Deep Orange': 0.311203,
    'Light Orange': 0.265560,
    'Orange': 0.157676,
    'Orange-Red': 0.228216,
    'Yellow-Orange': 0.037344
}

fixed_belmished_freq_map = {
    'N': 0.618257,
    'N (Minor)': 0.004149,
    'N (Split Skin)': 0.004149,
    'Y (Bruise)': 0.004149,
    'Y (Bruising)': 0.037344,
    'Y (Minor Insect Damage)': 0.024896,
    'Y (Minor)': 0.058091,
    'Y (Mold Spot)': 0.041494,
    'Y (Scars)': 0.070539,
    'Y (Split Skin)': 0.033195,
    'Y (Sunburn Patch)': 0.095436,
    'Y (Sunburn)': 0.008299
}

fixed_variety_freq_map = {
    'Ambiance': 0.045643,
    'Blood Orange': 0.008299,
    'California Valencia': 0.029046,
    'Cara Cara': 0.087137,
    'Clementine': 0.058091,
    'Clementine (Seedless)': 0.016598,
    'Hamlin': 0.020747,
    'Honey Tangerine': 0.029046,
    'Jaffa': 0.045643,
    'Midsweet (Hybrid)': 0.020747,
    'Minneola (Hybrid)': 0.049793,
    'Moro (Blood)': 0.066390,
    'Murcott (Hybrid)': 0.012448,
    'Navel': 0.066390,
    'Navel (Early Season)': 0.008299,
    'Navel (Late Season)': 0.012448,
    'Ortanique (Hybrid)': 0.053942,
    'Satsuma Mandarin': 0.053942,
    'Star Ruby': 0.074689,
    'Tangelo (Hybrid)': 0.004149,
    'Tangerine': 0.058091,
    'Temple': 0.074689,
    'Valencia': 0.045643,
    'Washington Navel': 0.058091
}

# Configuration de la page
st.set_page_config(
    page_title="🍎 Prédicteur de Qualité des Fruits orangier",
    page_icon="🍊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS personnalisé pour l'apparence
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #87CEEB, #20B2AA, #F08080);
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
        color: white;
        font-size: 2.5em;
        font-weight: bold;
        margin-bottom: 2rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
    }
    
    .fruit-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 15px;
        margin: 1rem 0;
        color: white;
        box-shadow: 0 8px 16px rgba(0,0,0,0.2);
        height: 400px;
        display: flex;
        flex-direction: column;
        justify-content: space-between;
        overflow: hidden;
        transition: all 0.3s ease;
        cursor: pointer;
    }

    .fruit-card:hover {
        transform: translateY(-10px) scale(1.02);
        box-shadow: 0 15px 30px rgba(0,0,0,0.3);
        background: linear-gradient(135deg, #20B2AA 0%, #87CEEB 100%);
    }
    
    .quality-badge {
        display: inline-block;
        padding: 0.5rem 1rem;
        border-radius: 25px;
        font-weight: bold;
        font-size: 1.2em;
        margin: 0.5rem;
    }
    
    .quality-1 { background: #F08080; color: white; }
    .quality-2 { background: rgba(240, 128, 128, 0.8); color: white; }
    .quality-3 { background: rgba(32, 178, 170, 0.7); color: white; }
    .quality-4 { background: rgba(32, 178, 170, 0.9); color: white; }
    .quality-5 { background: #20B2AA; color: white; }
    
    .prediction-result {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 15px;
        text-align: center;
        color: white;
        margin: 2rem 0;
        box-shadow: 0 8px 16px rgba(0,0,0,0.2);
    }
</style>
""", unsafe_allow_html=True)

# Données d'exemple pour l'entraînement du modèle
@st.cache_data
def load_sample_data():
    data = pd.read_csv('orange.csv')
    return pd.DataFrame(data)


# Entraîner le modèle
@st.cache_resource
def train_model():
    df = load_sample_data()

# En-tête principal
st.markdown('<div class="main-header">🟡🍊 Prédicteur de Qualité des Oranges 🍯🟡🍊</div>', unsafe_allow_html=True)

# Navigation par onglets
tab1, tab2, tab3, tab4 = st.tabs(["🍊 Description des types d'Oranges", "🌟 Échelle de Qualité","📊 Analyse des Données", "🔮 Prédiction"])

with tab1:
        st.markdown("## 🍊 Découvrez nos Variétés d'Oranges Premium")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("""
            <div class="fruit-card">
                <h3>🍊 Valencia</h3>
                <p>L'orange de jus par excellence ! Juteuse et parfaitement équilibrée,
                idéale pour les jus frais et les desserts.</p>
                <p><strong>Saison:</strong> Printemps-Été</p>
                <p><strong>Caractéristiques:</strong> Très juteuse, peu de pépins</p>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("""
            <div class="fruit-card">
                <h3>🟠 Navel</h3>
                <p>Reconnaissable par son "nombril", cette orange est parfaite à croquer.
                Chair ferme et sucrée, sans pépins.</p>
                <p><strong>Saison:</strong> Hiver-Printemps</p>
                <p><strong>Caractéristiques:</strong> Sans pépins, facile à éplucher</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
            <div class="fruit-card">
                <h3>🔴 Blood Orange (Sanguine)</h3>
                <p>Avec sa chair rouge distinctive, elle offre un goût unique
                légèrement acidulé et très parfumé.</p>
                <p><strong>Saison:</strong> Hiver</p>
                <p><strong>Caractéristiques:</strong> Chair rouge, goût intense</p>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("""
            <div class="fruit-card">
                <h3>🟡 Cara Cara</h3>
                <p>Orange rose à la chair délicate et sucrée, moins acide
                que les oranges traditionnelles.</p>
                <p><strong>Saison:</strong> Hiver-Printemps</p>
                <p><strong>Caractéristiques:</strong> Chair rose, très douce</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown("""
            <div class="fruit-card">
                <h3>🟠 Clementine</h3>
                <p>Petite et facile à éplucher, cette mandarine est parfaite
                pour les enfants et les collations rapides.</p>
                <p><strong>Saison:</strong> Automne-Hiver</p>
                <p><strong>Caractéristiques:</strong> Facile à éplucher, segments détachables</p>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("""
            <div class="fruit-card">
                <h3>🍯 Honey Tangerine</h3>
                <p>Comme son nom l'indique, cette variété est particulièrement
                sucrée avec un arôme délicat de miel.</p>
                <p><strong>Saison:</strong> Hiver</p>
                <p><strong>Caractéristiques:</strong> Très sucrée, parfum de miel</p>
            </div>
            """, unsafe_allow_html=True)
with tab2:
    st.markdown("## 🌟 Échelle de Qualité des Oranges")
    
    st.markdown("""
    Notre système d'évaluation utilise une échelle de **1 à 5 étoiles** pour classifier la qualité de nos fruits :
    """)
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.markdown("""
        <div class="quality-badge quality-5">⭐⭐⭐⭐⭐ Qualité 5</div><br>
        <div class="quality-badge quality-4">⭐⭐⭐⭐ Qualité 4</div><br>
        <div class="quality-badge quality-3">⭐⭐⭐ Qualité 3</div><br>
        <div class="quality-badge quality-2">⭐⭐ Qualité 2</div><br>
        <div class="quality-badge quality-1">⭐ Qualité 1</div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        **🌟 Qualité 5 - Excellence Premium**
        - Apparence parfaite, sans défaut
        - Maturité optimale
        - Saveur exceptionnelle
        - Texture idéale
        
        **⭐ Qualité 4 - Très Bonne**
        - Quelques imperfections mineures
        - Bonne maturité
        - Saveur agréable
        
        **⭐ Qualité 3 - Bonne**
        - Qualité standard acceptable
        - Maturité correcte
        - Quelques défauts visibles
        
        **⭐ Qualité 2 - Passable**
        - Défauts visibles
        - Maturité inégale
        - Goût moins prononcé
        
        **⭐ Qualité 1 - Faible**
        - Nombreux défauts
        - Problèmes de maturité
        - Qualité inférieure
        """)
    
    # Graphique de distribution
    st.markdown("### 📊 Distribution des Qualités")
    df_sample = load_sample_data()
    quality_counts = df_sample['Quality (1-5)'].value_counts().sort_index()
    
    fig = px.bar(
        x=quality_counts.index,
        y=quality_counts.values,
        labels={'x': 'Niveau de Qualité', 'y': 'Nombre d\'oranges'},
        title="Distribution des Niveaux de Qualité",
        color=quality_counts.values,
        color_continuous_scale='Viridis'
    )
    st.plotly_chart(fig, use_container_width=True)
with tab3:
    df_sample = load_sample_data()
    st.markdown("## 📊 Analyse Approfondie des Données")
    
    # Métriques générales
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Échantillons Total", len(df_sample))
    with col2:
        st.metric("Qualité Moyenne", f"{df_sample['Quality (1-5)'].mean():.2f}")
    with col3:
        st.metric("Variétés", df_sample['Variety'].nunique())
    with col4:
        st.metric("Couleurs", df_sample['Color'].nunique())

    # 2. Corrélations entre variables
    st.markdown("### 🔗 Matrice de Corrélation")
    
    # Préparer les données numériques pour la corrélation
    df_corr = df_sample.copy()
    
    numeric_cols = ['Size (cm)', 'Weight (g)', 'Softness (1-5)', 'Brix (Sweetness)', 'pH (Acidity)',
                   'HarvestTime (days)', 'Ripeness (1-5)', 'Quality (1-5)']
    
    correlation_matrix = df_corr[numeric_cols].corr()
    
    fig_corr = px.imshow(
        correlation_matrix,
        text_auto=True,
        aspect="auto",
        title="Matrice de Corrélation entre les Variables",
        color_continuous_scale='RdBu'
    )
    fig_corr.update_layout(height=600)
    st.plotly_chart(fig_corr, use_container_width=True)
    
    # 3. Distribution par qualité
    st.markdown("### 📈 Distribution des Caractéristiques par Niveau de Qualité")
    
    # Sélecteur de variable à analyser
    analysis_var = st.selectbox(
        "Choisissez une variable à analyser :",
        ['Brix (Sweetness)', 'Ripeness (1-5)', 'Softness (1-5)', 'Size (cm)', 'Weight (g)', 'pH (Acidity)']
    )
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Box plot
        fig_box = px.box(
            df_sample,
            x='Quality (1-5)',
            y=analysis_var,
            title=f"Distribution de {analysis_var} par Qualité",
            color='Quality (1-5)',
            color_discrete_sequence=px.colors.qualitative.Set2
        )
        st.plotly_chart(fig_box, use_container_width=True)
    
    with col2:
        # Violin plot
        fig_violin = px.violin(
            df_sample,
            x='Quality (1-5)',
            y=analysis_var,
            title=f"Densité de {analysis_var} par Qualité",
            color='Quality (1-5)',
            color_discrete_sequence=px.colors.qualitative.Set2
        )
        st.plotly_chart(fig_violin, use_container_width=True)
    
    # 4. Analyse par variété
    st.markdown("### 🍎 Analyse par Variété de Types")
    
    variety_quality = df_sample.groupby(['Variety', 'Quality (1-5)']).size().unstack(fill_value=0)
    variety_quality_pct = variety_quality.div(variety_quality.sum(axis=1), axis=0) * 100
    
    fig_variety = px.bar(
        variety_quality_pct.reset_index().melt(id_vars='Variety'),
        x='Variety',
        y='value',
        color='Quality (1-5)',
        title="Répartition de la Qualité par Variété (%)",
        labels={'value': 'Pourcentage', 'Quality (1-5)': 'Qualité'}
    )
    fig_variety.update_xaxes(tickangle=45)
    st.plotly_chart(fig_variety, use_container_width=True)
    
    # 5. Impact des imperfections
    st.markdown("### 🔍 Impact des Imperfections sur la Qualité")
    
    col1, col2 = st.columns(2)
    
    with col1:
        blemish_quality = df_sample.groupby(['Blemishes (Y/N)', 'Quality (1-5)']).size().unstack(fill_value=0)
        blemish_quality_pct = blemish_quality.div(blemish_quality.sum(axis=1), axis=0) * 100
        
        fig_blemish = px.bar(
            blemish_quality_pct.reset_index().melt(id_vars='Blemishes (Y/N)'),
            x='Blemishes (Y/N)',
            y='value',
            color='Quality (1-5)',
            title="Impact des Imperfections sur la Qualité",
            labels={'value': 'Pourcentage', 'Blemishes (Y/N)': 'Imperfections'}
        )
        st.plotly_chart(fig_blemish, use_container_width=True)
    
    with col2:
        # Statistiques descriptives
        st.markdown("#### 📊 Statistiques Clés")
        
        avg_quality_no_blemish = df_sample[df_sample['Blemishes (Y/N)'] == 'N']['Quality (1-5)'].mean()
        avg_quality_with_blemish = df_sample[df_sample['Blemishes (Y/N)'] == 'Y (Sunburn Patch)']['Quality (1-5)'].mean()
        
        st.markdown(f"""
        <div class="metric-card">
            <strong>Qualité moyenne sans imperfections:</strong> {avg_quality_no_blemish} ⭐
        </div>
        """, unsafe_allow_html=True)
    
    # 6. Insights et recommandations
    st.markdown("### 💡 Insights et Recommandations")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class="insight-box">
            <h4>🎯 Facteurs Critiques Identifiés</h4>
            <ul>
                <li><strong>Taux de sucre (Brix)</strong> : Impact majeur sur la qualité perçue</li>
                <li><strong>Maturité</strong> : Timing crucial pour une qualité optimale</li>
                <li><strong>Imperfections</strong> : Réduction significative de la qualité</li>
                <li><strong>Tendreté</strong> : Indicateur important de fraîcheur</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="insight-box">
            <h4>📈 Recommandations d'Amélioration</h4>
            <ul>
                <li><strong>Optimiser</strong> le timing de récolte selon la maturité</li>
                <li><strong>Contrôler</strong> rigoureusement les imperfections</li>
                <li><strong>Surveiller</strong> le taux de sucre des Oranges</li>
                <li><strong>Adapter</strong> les critères selon les variétés</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
with tab4:
    st.markdown("## 🔮 Prédiction de la Qualité")
    st.markdown("Remplissez le formulaire ci-dessous pour prédire la qualité de votre Orange :")
    
    with st.form("prediction_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### 📏 Caractéristiques Physiques")
            size = st.number_input("🍎 Taille (cm)", value=0, step=1)
            weight = st.number_input("⚖️ Poids (g)", value=0, step=1)

            # size = st.slider("🍎 Taille (cm)", 3.0, 15.0, 8.0, 0.1)
            # weight = st.slider("⚖️ Poids (g)", 50, 400, 150, 5)
            softness = st.selectbox("🤏 Tendreté (1-5)", [1, 1.5,2,2.5, 3,3.5, 4,4.5, 5], index=2)
            color_options = df_sample['Color'].unique().tolist()
            default_idx = color_options.index('Orange') if 'Orange' in color_options else (0 if color_options else 0)
            color = st.selectbox("🎨 Couleur", color_options, index=default_idx)
            blemishes_options = df_sample['Blemishes (Y/N)'].unique().tolist()
            default_idx_blemishes = blemishes_options.index('N (Minor)') if 'N (Minor)' in blemishes_options else (0 if blemishes_options else 0)
            blemishes = st.selectbox("🔍 Imperfections", blemishes_options, index=default_idx_blemishes)
        with col2:
            st.markdown("### 🧪 Caractéristiques Gustatives")
            brix = st.number_input("🍯 Taux de sucre (Brix)", value=0, step=1)
            ph = st.number_input("⚗️ Acidité (pH)", value=0, step=1)
            
            st.markdown("### 🌱 Informations de Culture")
            harvest_time = st.number_input("📅 Temps avant récolte (jours)", value=0, step=1)
            ripeness = st.selectbox("🍃 Maturité (1-5)", [1,1.5, 2,2.5, 3,3.5, 4,4.5, 5], index=2)
            variety_options = df_sample['Variety'].unique().tolist()
            default_idx = variety_options.index('Orange') if 'Orange' in variety_options else (0 if variety_options else 0)
            variety = st.selectbox("🍇 Variété", variety_options, index=default_idx)
        
        submitted = st.form_submit_button("🚀 Prédire la Qualité", use_container_width=True)
        
        if submitted:
            # Préparation des données pour la prédiction
            input_data = pd.DataFrame({
                'Size (cm)': [size],
                'Weight (g)': [weight],
                'Softness (1-5)': [softness],
                'Color': fixed_color_freq_map[color],
                'Blemishes (Y/N)': fixed_belmished_freq_map[blemishes],
                'Brix (Sweetness)': [brix],
                'pH (Acidity)': [ph],
                'HarvestTime (days)': [harvest_time],
                'Ripeness (1-5)': [ripeness],
                'Variety': fixed_variety_freq_map[variety], #[le_variety.transform([variety])[0]]
            })
            
            x_array = np.array([size, weight, brix, ph, softness, harvest_time, ripeness, fixed_color_freq_map[color], fixed_variety_freq_map[variety], fixed_belmished_freq_map[blemishes]])
            normalized_arr = preprocessing.normalize([x_array])
            # transformation en DMATRIX
            dpredict = xgb.DMatrix(normalized_arr)
            pickled_model1 = pickle.load(open('xgboost_model.pkl', 'rb'))
            raw_prediction = pickled_model1.predict(dpredict)
            # st.table(raw_prediction)
            # je vais récupérer la classes ayant la plus forte probabilité, parce que xgboost prédit un tableau avec des probabilités correspondant à chaque class
            # et j'ajoute 1 parce que np compte les indices à partir de 0, et l'utilisateur de comprendra pas donc c'est mieux d'aller de 1-5 au lieu de 0-4
            
            max_idx = np.argmax(raw_prediction) + 1
            max_val = np.max(raw_prediction)

            # Prédiction            
            # Descriptions des qualités
            quality_descriptions = {
                1: "Ce fruit présente une qualité faible avec des défauts significatifs. Il pourrait être utilisé pour la transformation mais n'est pas idéal pour la consommation directe.",
                2: "Ce fruit a une qualité passable avec quelques défauts visibles. Acceptable pour certains usages mais pas optimal.",
                3: "Ce fruit présente une qualité standard correcte. Il répond aux critères de base pour la vente au détail.",
                4: "Ce fruit a une très bonne qualité avec des caractéristiques attrayantes. Excellent choix pour les consommateurs.",
                5: "Ce fruit présente une qualité premium exceptionnelle ! Parfait en tous points, il représente l'excellence de nos produits."
            }
            
            # Affichage du résultat
            stars = "⭐" * max_idx
            quality_colors = {1: "#FF6B6B", 2: "#FFA726", 3: "#FFEB3B", 4: "#66BB6A", 5: "#4CAF50"}
            
            st.markdown(f"""
            <div class="prediction-result">
                <h2>🎯 Résultat de la Prédiction</h2>
                <h1 style="font-size: 3em; margin: 1rem 0;">{stars}</h1>
                <h3>Qualité Prédite: <span style="color: {quality_colors[max_idx]};">{max_idx}/5</span></h3>
                <div style="margin: 1rem 0; padding: 1rem; background-color: rgba(0,123,255,0.1); border-radius: 8px; border-left: 4px solid #007bff;">
                    <h4 style="margin: 0 0 0.5rem 0; color: white;">🎯 Niveau de Confiance</h4>
                    <p style="font-size: 1.2em; font-weight: bold; margin: 0; color: black;">
                        {max_val:.1%} de certitude
                    </p>
                    <div style="background-color: #e9ecef; border-radius: 10px; height: 20px; margin-top: 0.5rem; overflow: hidden;">
                        <div style="background-color: {quality_colors[max_idx]}; height: 100%; width: {max_val:.1%}; border-radius: 10px; transition: width 0.3s ease;"></div>
                    </div>
                </div>
                <hr style="margin: 2rem 0;">
                <p style="font-size: 1.1em; line-height: 1.6; text-align: left;">
                    {quality_descriptions[max_idx]}
                </p>
            </div>
            """, unsafe_allow_html=True)
            classes = [f"Classe {i+1}" for i in range(len(raw_prediction[0]))]
            probabilities = raw_prediction[0]  # Extract probabilities from the first (and only) prediction
            fig = px.bar(
                x=classes, 
                y=probabilities,
                title='Probabilités par Classe de Qualité',
                labels={'x': 'Classes de Qualité', 'y': 'Probabilité'},
                color=classes,
                color_continuous_scale='Viridis'
            )
            fig.update_layout(
                xaxis_title="Classes de Qualité",
                yaxis_title="Probabilité",
                showlegend=False
            )
            st.plotly_chart(fig)


# Sidebar avec informations
with st.sidebar:
    st.markdown("## 🍎 À Propos")
    st.markdown("""
    Cette application utilise l'intelligence artificielle pour prédire la qualité des Oranges
     
    basée sur leurs caractéristiques physiques, gustatives et de culture.
    
    ### 🔧 Fonctionnalités:
    - **Description des types d'oranges** avec informations détaillées
    - **Échelle de qualité** de 1 à 5 étoiles
    - **Prédiction intelligente** avec niveau de confiance
    - **Visualisations interactives**
    
    ### 🧠 Modèle:
    Xg boost avec 5 classes
    
    ### 📊 Données:
    Plus de 100 échantillons d'entraînement
    """)
    
    st.markdown("---")
    st.markdown("### 🎯 Conseils d'utilisation")
    st.info("""
    Pour une prédiction optimale:
    - Mesurez précisément la taille et le poids
    - Évaluez objectivement la tendreté et la maturité
    - Vérifiez soigneusement la présence d'imperfections
    """)

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; padding: 2rem;">
    🍎 Développé avec ❤️ pour l'industrie fruitière 🍊<br>
    Optimisez vos processus de tri et de classification !
</div>
""", unsafe_allow_html=True)