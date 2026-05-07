"""
🌙 Sleepio
Application IA de prédiction des troubles du sommeil
UI moderne • thème nocturne • images fonctionnelles
"""

# =========================================================
# IMPORTS
# =========================================================

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
# CSS MODERNE
# =========================================================

st.markdown("""
<style>

/* =======================================================
GLOBAL
======================================================= */

html, body, [class*="css"] {
    font-family: 'Segoe UI', sans-serif;
}

.stApp {
    background:
        linear-gradient(
            180deg,
            #020617 0%,
            #0F172A 100%
        );
    color: white;
}

/* =======================================================
BANNER
======================================================= */

.banner {

    background:
        linear-gradient(
            135deg,
            rgba(15,23,42,0.96),
            rgba(30,41,59,0.92)
        );

    border: 1px solid rgba(255,255,255,0.05);

    padding: 3rem;
    border-radius: 30px;

    text-align: center;

    margin-bottom: 2rem;

    box-shadow:
        0 0 30px rgba(0,255,200,0.12);
}

.banner-title {

    font-size: 4rem;
    font-weight: 800;

    color: white;

    margin-bottom: 0.3rem;
}

.banner-sub {

    font-size: 1.2rem;

    color: #CBD5E1;
}

/* =======================================================
TITRES
======================================================= */

h1, h2, h3 {
    color: #7DD3FC !important;
}

/* =======================================================
CARDS
======================================================= */

.card {

    background:
        rgba(15,23,42,0.7);

    border:
        1px solid rgba(255,255,255,0.06);

    border-radius: 20px;

    padding: 1.5rem;

    margin-bottom: 1rem;

    backdrop-filter: blur(10px);
}

/* =======================================================
BOUTONS
======================================================= */

.stButton > button,
.stFormSubmitButton > button {

    width: 100%;

    border: none;
    border-radius: 15px;

    padding: 0.8rem 1rem;

    background:
        linear-gradient(
            135deg,
            #06B6D4,
            #14B8A6
        );

    color: white;

    font-size: 1rem;
    font-weight: 700;

    transition: 0.25s ease;

    box-shadow:
        0 0 20px rgba(20,184,166,0.25);
}

.stButton > button:hover,
.stFormSubmitButton > button:hover {

    transform:
        translateY(-2px)
        scale(1.02);

    box-shadow:
        0 0 30px rgba(34,211,238,0.55);
}

/* =======================================================
INPUTS
======================================================= */

.stSelectbox div[data-baseweb="select"],
.stNumberInput input {

    background-color: #111827 !important;
    color: white !important;

    border-radius: 12px !important;
}

/* =======================================================
TABS
======================================================= */

.stTabs [data-baseweb="tab-list"] {
    gap: 12px;
}

.stTabs [data-baseweb="tab"] {

    background:
        rgba(255,255,255,0.03);

    border-radius: 12px;

    padding: 10px 20px;
}

.stTabs [aria-selected="true"] {

    background:
        linear-gradient(
            135deg,
            #0891B2,
            #14B8A6
        ) !important;
}

/* =======================================================
METRICS
======================================================= */

[data-testid="metric-container"] {

    background:
        rgba(15,23,42,0.8);

    border-radius: 16px;

    padding: 1rem;

    border:
        1px solid rgba(255,255,255,0.05);
}

/* =======================================================
IMAGES
======================================================= */

img {
    border-radius: 20px;
}

/* =======================================================
FOOTER
======================================================= */

footer {
    visibility: hidden;
}

</style>
""", unsafe_allow_html=True)

# =========================================================
# CHARGEMENT DU MODELE
# =========================================================

@st.cache_resource
def load_latest_model():

    model_files = list(
        MODELS_DIR.glob("random_forest_sleep_*.pkl")
    )

    if not model_files:
        return None, None

    latest_model = max(
        model_files,
        key=lambda x: x.stat().st_mtime
    )

    model = joblib.load(latest_model)

    le_files = list(
        MODELS_DIR.glob("label_encoder_sleep_*.pkl")
    )

    le = None

    if le_files:

        latest_le = max(
            le_files,
            key=lambda x: x.stat().st_mtime
        )

        le = joblib.load(latest_le)

    return model, le

# =========================================================
# BANNER
# =========================================================

st.markdown("""
    <div class="banner">
        <h1 class="banner-title">🌙 Sleepio</h1>
        <p class="banner-subtitle">
            Intelligence Artificielle pour la détection des troubles du sommeil
        </p>
    </div>
""", unsafe_allow_html=True)


# =========================================================
# TABS
# =========================================================

tab1, tab2, tab3 = st.tabs([
    "📊 Présentation",
    "🔮 Prédiction",
    "📈 Visualisations"
])

# =========================================================
# TAB 1 - PRESENTATION
# =========================================================

with tab1:

    col1, col2 = st.columns([1.5, 1])

    with col1:

        st.markdown('<div class="card">', unsafe_allow_html=True)

        st.subheader("🧠 Objectif du Projet")

        st.write("""
        Cette application prédit les troubles du sommeil
        grâce au Machine Learning.

        Types détectés :

        - ✅ Sommeil normal
        - ⚠️ Insomnie
        - 🚨 Apnée du sommeil

        Les prédictions utilisent des données
        physiologiques et comportementales.
        """)

        st.metric(
            "Précision du modèle",
            "94%",
            "+3%"
        )

        st.markdown('</div>', unsafe_allow_html=True)

    with col2:

        st.image(
            "https://images.unsplash.com/photo-1541781774459-bb2af2f05b55?q=80&w=1200&auto=format&fit=crop",
            use_container_width=True
        )

# =========================================================
# TAB 2 - PREDICTION
# =========================================================

with tab2:

    st.subheader("🔮 Analyse personnalisée")

    model, le = load_latest_model()

    if model is None:

        st.error("❌ Aucun modèle trouvé dans /models")
        st.stop()

    st.success("✅ Modèle chargé avec succès")

    with st.form("prediction_form"):

        colA, colB = st.columns(2)

        with colA:

            gender = st.selectbox(
                "Genre",
                ["Male", "Female"]
            )

            age = st.slider(
                "Âge",
                20, 70, 42
            )

            occupation = st.selectbox(
                "Profession",
                [
                    "Doctor",
                    "Engineer",
                    "Lawyer",
                    "Manager",
                    "Nurse",
                    "Sales Representative",
                    "Salesperson",
                    "Scientist",
                    "Software Engineer",
                    "Teacher",
                    "Accountant"
                ]
            )

            sleep_duration = st.slider(
                "Durée du sommeil (h)",
                4.0, 9.0, 6.8, 0.1
            )

            quality_of_sleep = st.slider(
                "Qualité du sommeil",
                1, 10, 6
            )

        with colB:

            stress_level = st.slider(
                "Niveau de stress",
                1, 10, 6
            )

            physical_activity = st.slider(
                "Activité physique (min)",
                30, 120, 60
            )

            bmi = st.selectbox(
                "Catégorie BMI",
                [
                    "Normal",
                    "Overweight",
                    "Obese"
                ]
            )

            heart_rate = st.slider(
                "Fréquence cardiaque",
                60, 110, 78
            )

            daily_steps = st.slider(
                "Pas quotidiens",
                2000, 15000, 7000
            )

            systolic = st.number_input(
                "Pression systolique",
                90, 180, 125
            )

            diastolic = st.number_input(
                "Pression diastolique",
                60, 120, 82
            )

        submitted = st.form_submit_button(
            "🚀 Lancer la prédiction"
        )

    # =====================================================
    # PREDICTION
    # =====================================================

    if submitted:

        input_data = pd.DataFrame({

            'Age': [age],

            'Sleep Duration': [sleep_duration],

            'Quality of Sleep': [quality_of_sleep],

            'Physical Activity Level': [physical_activity],

            'Stress Level': [stress_level],

            'Heart Rate': [heart_rate],

            'Daily Steps': [daily_steps],

            'Systolic_BP': [systolic],

            'Diastolic_BP': [diastolic],

            'Gender': [gender],

            'Occupation': [occupation],

            'BMI Category': [bmi]
        })

        input_encoded = pd.get_dummies(
            input_data,
            columns=[
                'Gender',
                'Occupation',
                'BMI Category'
            ]
        )

        if hasattr(model, 'feature_names_in_'):

            model_features = model.feature_names_in_

            for col in model_features:

                if col not in input_encoded.columns:
                    input_encoded[col] = 0

            input_encoded = input_encoded[
                model_features
            ]

        prediction = model.predict(
            input_encoded
        )[0]

        prediction_proba = model.predict_proba(
            input_encoded
        )[0]

        pred_label = le.inverse_transform(
            [prediction]
        )[0]

        st.divider()

        st.subheader("🧾 Résultat")

        if pred_label == "Normal":

            st.success(
                f"✅ {pred_label}"
            )

        elif pred_label == "Insomnia":

            st.warning(
                f"⚠️ {pred_label}"
            )

        else:

            st.error(
                f"🚨 {pred_label}"
            )

        # ================================================
        # BAR CHART
        # ================================================

        proba_df = pd.DataFrame({

            "Trouble": le.classes_,

            "Probabilité":
                (prediction_proba * 100).round(2)
        })

        st.bar_chart(
            proba_df.set_index("Trouble")
        )

        st.info(
            f"Confiance du modèle : "
            f"{prediction_proba.max()*100:.1f}%"
        )

# =========================================================
# TAB 3 - VISUALISATIONS
# =========================================================

with tab3:

    st.subheader(
        "📈 Analyses & Visualisations"
    )

    plots = {

        "Durée du sommeil":
            "duration_par_sleep_disorder.png",

        "Qualité du sommeil":
            "quality_par_sleep_disorder.png",

        "Stress":
            "stress_par_sleep_disorder.png",

        "Fréquence cardiaque":
            "heartrate_par_sleep_disorder.png",

        "Âge":
            "age_par_sleep_disorder.png",

        "Activité physique":
            "activity_par_sleep_disorder.png",

        "Pas quotidiens":
            "steps_par_sleep_disorder.png",

        "Pression systolique":
            "systolic_BP_par_sleep_disorder.png",

        "Distribution BMI":
            "BMI_distribution_of_sleep_disorder.png",

        "Répartition genre":
            "gender_distribution_of_sleep_disorder.png"
    }

    cols = st.columns(2)

    for i, (title, filename) in enumerate(
        plots.items()
    ):

        with cols[i % 2]:

            file_path = PLOTS_DIR / filename

            st.markdown(
                '<div class="card">',
                unsafe_allow_html=True
            )

            if file_path.exists():

                st.image(
                    str(file_path),
                    caption=title,
                    use_container_width=True
                )

            else:

                st.warning(
                    f"Image manquante : {filename}"
                )

            st.markdown(
                '</div>',
                unsafe_allow_html=True
            )

# =========================================================
# FOOTER
# =========================================================

st.divider()

st.caption(
    "🌙 Sleepio • Machine Learning • "
    "Random Forest • Dashboard IA Santé • 2026"
)