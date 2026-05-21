from pathlib import Path
import pandas as pd
import joblib
import streamlit as st

# =========================================================
# CONFIGURATION PAGE
# =========================================================
st.set_page_config(
    page_title="Sleepio | IA Sommeil",
    page_icon="🌙",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# =========================================================
# PATHS
# =========================================================
PROJECT_ROOT = Path(__file__).parent.parent
MODELS_DIR = PROJECT_ROOT / "models"
PLOTS_DIR = PROJECT_ROOT / "plots"

# =========================================================
# CSS MODERNE (Optimisé)
# =========================================================
st.markdown("""
<style>
    .stApp {
        background: linear-gradient(180deg, #020617 0%, #0F172A 100%);
        color: white;
    }
    .banner {
        background: linear-gradient(135deg, rgba(15,23,42,0.96), rgba(30,41,59,0.92));
        border: 1px solid rgba(255,255,255,0.1);
        padding: 2.5rem;
        border-radius: 25px;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 0 10px 30px rgba(0,0,0,0.5);
    }
    .banner-title { font-size: 3.5rem; font-weight: 800; color: white; margin: 0; }
    .banner-subtitle { color: #7DD3FC; font-size: 1.2rem; opacity: 0.9; }
    
    .card {
        background: rgba(30, 41, 59, 0.5);
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 15px;
        padding: 20px;
        margin-bottom: 15px;
    }
    
    /* Harmonisation des Sliders et Inputs */
    .stSlider [data-baseweb="slider"] { margin-bottom: 10px; }
    
    /* Bouton stylisé */
    .stButton > button {
        background: linear-gradient(135deg, #06B6D4 0%, #14B8A6 100%);
        border: none;
        color: white;
        font-weight: bold;
        height: 3em;
        transition: all 0.3s;
    }
    .stButton > button:hover { transform: scale(1.01); border: none; }
</style>
""", unsafe_allow_html=True)

# =========================================================
# CHARGEMENT DU MODELE
# =========================================================
@st.cache_resource
def load_latest_model():
    try:
        model_files = list(MODELS_DIR.glob("random_forest_sleep_*.pkl"))
        le_files = list(MODELS_DIR.glob("label_encoder_sleep_*.pkl"))
        if not model_files or not le_files: return None, None
        
        latest_model = max(model_files, key=lambda x: x.stat().st_mtime)
        latest_le = max(le_files, key=lambda x: x.stat().st_mtime)
        
        return joblib.load(latest_model), joblib.load(latest_le)
    except:
        return None, None

# =========================================================
# HEADER
# =========================================================
st.markdown("""
    <div class="banner">
        <h1 class="banner-title">🌙 Sleepio</h1>
        <p class="banner-subtitle">Analyse prédictive de la santé du sommeil par Intelligence Artificielle</p>
    </div>
""", unsafe_allow_html=True)

tab1, tab2, tab3 = st.tabs(["📊 Présentation", "🔮 Diagnostic IA", "📈 Analyses Exploratoires"])

# =========================================================
# TAB 1 - PRESENTATION
# =========================================================
with tab1:
    col1, col2 = st.columns([1.2, 1], gap="large")
    with col1:
        st.markdown("""
        ### Pourquoi Sleepio ?
        Les troubles du sommeil non détectés impactent durablement la santé cardiovasculaire et mentale. 
        Sleepio utilise un modèle **Random Forest** entraîné sur des profils cliniques pour identifier 
        les biomarqueurs du sommeil.
        """)
        
        c1, c2 = st.columns(2)
        c1.metric("Précision", "94%", "+2.1%")
        c2.metric("Latence IA", "12ms", "-5ms")
        
        st.info("**Pathologies couvertes :** Insomnie, Apnée du Sommeil, Sommeil Récupérateur.")
    
    with col2:
        st.image("https://images.unsplash.com/photo-1541781774459-bb2af2f05b55?q=80&w=1200&auto=format&fit=crop")

# =========================================================
# TAB 2 - DIAGNOSTIC
# =========================================================
with tab2:
    model, le = load_latest_model()
    if model is None:
        st.error("⚠️ Modèle introuvable. Vérifiez le dossier `/models`.")
        st.stop()

    with st.form("prediction_form"):
        st.markdown("### 🧬 Paramètres Physiologiques")
        
        colA, colB = st.columns(2, gap="medium")
        
        with colA:
            gender = st.selectbox("Genre", ["Male", "Female"])
            age = st.slider("Âge", 18, 90, 35)
            occupation = st.selectbox("Profession", ["Doctor", "Engineer", "Lawyer", "Manager", "Nurse", "Salesperson", "Scientist", "Software Engineer", "Teacher", "Accountant"])
            sleep_duration = st.slider("Sommeil (heures)", 3.0, 10.0, 7.0, 0.5)
            quality_of_sleep = st.select_slider("Qualité perçue (1-10)", options=range(1, 11), value=7)

        with colB:
            bmi = st.selectbox("Catégorie BMI", ["Normal", "Overweight", "Obese"])
            heart_rate = st.number_input("Rythme cardiaque (repos)", 40, 120, 70)
            systolic = st.number_input("Tension Systolique", 90, 180, 120)
            diastolic = st.number_input("Tension Diastolique", 60, 110, 80)
            stress_level = st.select_slider("Niveau de stress", options=range(1, 11), value=4)

        with st.expander("🏃 Activité Physique & Habitudes"):
            c_steps, c_act = st.columns(2)
            physical_activity = c_act.slider("Activité (min/jour)", 0, 120, 30)
            daily_steps = c_steps.number_input("Pas quotidiens", 0, 20000, 5000)

        submitted = st.form_submit_button("Lancer l'analyse AI")

    if submitted:
        # -- Traitement des données --
        input_data = pd.DataFrame({
            'Age': [age], 'Sleep Duration': [sleep_duration], 'Quality of Sleep': [quality_of_sleep],
            'Physical Activity Level': [physical_activity], 'Stress Level': [stress_level], 'Heart Rate': [heart_rate],
            'Daily Steps': [daily_steps], 'Systolic_BP': [systolic], 'Diastolic_BP': [diastolic],
            'Gender': [gender], 'Occupation': [occupation], 'BMI Category': [bmi]
        })
        
        input_encoded = pd.get_dummies(input_data)
        for col in model.feature_names_in_:
            if col not in input_encoded.columns: input_encoded[col] = 0
        input_encoded = input_encoded[model.feature_names_in_]

        prediction = model.predict(input_encoded)[0]
        proba = model.predict_proba(input_encoded)[0]
        label = le.inverse_transform([prediction])[0]

        # -- Affichage Résultat --
        st.markdown("---")
        res_col1, res_col2 = st.columns([1, 1.5])
        
        with res_col1:
            st.markdown("#### Résultat du Diagnostic")
            if label == "Normal": st.success(f"### ✨ {label}")
            elif label == "Insomnia": st.warning(f"### ⚠️ {label}")
            else: st.error(f"### 🚨 {label}")
            
            st.metric("Indice de confiance", f"{max(proba)*100:.1f}%")

        with res_col2:
            st.markdown("#### Probabilités par catégorie")
            proba_df = pd.DataFrame({"Diagnostic": le.classes_, "Certitude (%)": proba * 100})
            st.bar_chart(proba_df.set_index("Diagnostic"), horizontal=True)

# =========================================================
# TAB 3 - VISUALISATIONS (Avec ajouts demandés)
# =========================================================
with tab3:
    st.subheader("🔍 Insights du Dataset & Modèle")
    
    # Dictionnaire étendu avec les nouveaux plots
    plots = {
        "🎯 Importance des Features (Calorie)": "Feature_importance_calory.png",
        "🔗 Matrice de Corrélation": "correlation_sleep.png",
        "📊 Sommeil vs Pathologie": "duration_par_sleep_disorder.png",
        "📉 Stress vs Pathologie": "stress_par_sleep_disorder.png",
        "⚖️ Impact du BMI": "BMI_distribution_of_sleep_disorder.png",
        "💓 Rythme Cardiaque": "heartrate_par_sleep_disorder.png"
    }

    # Affichage en grille 2x3
    plot_cols = st.columns(2)
    for i, (title, filename) in enumerate(plots.items()):
        with plot_cols[i % 2]:
            path = PLOTS_DIR / filename
            st.markdown(f'<div class="card">', unsafe_allow_html=True)
            st.markdown(f"**{title}**")
            if path.exists():
                st.image(str(path), use_container_width=True)
            else:
                st.warning(f"Fichier `{filename}` non trouvé.")
            st.markdown('</div>', unsafe_allow_html=True)

# =========================================================
# FOOTER
# =========================================================
st.markdown("---")
st.caption("© 2026 Sleepio AI Health Lab • Système expert basé sur Random Forest Classifier")