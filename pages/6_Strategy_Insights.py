"""
Page 6: Strategy Insights & Machine Learning
Kaggriculture - Agricultural Simulation Analysis & Strategy Platform
"""
import streamlit as st
import pandas as pd

from config import APP_TITLE, CUSTOM_CSS
from src.state_manager import render_sidebar_status_and_filters
from src.strategy_insights import StrategyInsights
from src.visualization import Visualizer

st.set_page_config(page_title=f"Strategy Insights | {APP_TITLE}", page_icon="🌱", layout="wide")
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

filtered_df, filtered_actions_df = render_sidebar_status_and_filters()

st.title("💡 Strategic Insights & Predictive Intelligence")
st.markdown("<p style='color:#9CA3AF;'>Evidence-based operational rules derived from high-performing farming episodes and unsupervised machine learning clustering.</p>", unsafe_allow_html=True)

if filtered_df.empty:
    st.warning("⚠️ No simulation data available to evaluate strategy patterns.")
else:
    strat = StrategyInsights(filtered_df, filtered_actions_df)

    tab_rules, tab_ml_predict, tab_ml_cluster = st.tabs([
        "🧠 Evidence-Based Strategy Rules",
        "🔮 Score Prediction & Feature Leverage",
        "🎯 Unsupervised Strategy Clustering"
    ])

    with tab_rules:
        st.markdown("### 📜 **Empirical Findings from High-Performing Matches**")
        st.markdown("<span style='color:#9CA3AF;'>Derived by comparing top-quartile (Q4) versus bottom-quartile (Q1) final score cohorts.</span>", unsafe_allow_html=True)

        insights = strat.derive_evidence_based_insights()

        for ins in insights:
            with st.container():
                st.markdown(f"""
                <div class="kag-card">
                    <div class="kag-card-header">
                        <h4 style="margin:0; color:#34D399; font-size:1.1rem;">🌱 {ins.get('title', 'Insight')}</h4>
                        <span class="kag-badge-gold">{ins.get('metric', '')}</span>
                    </div>
                    <p style="color:#ECFDF5; font-size:0.95rem; margin-bottom:12px;">{ins.get('observation', '')}</p>
                    <div style="background:#0B1914; border:1px solid #1E3E33; padding:12px 14px; border-radius:8px; margin-bottom:10px;">
                        <span style="color:#34D399; font-weight:600;">Interpretation:</span> 
                        <span style="color:#F4F6F0;">{ins.get('interpretation', '')}</span>
                    </div>
                    <div style="font-size:0.8rem; color:#9CA3AF; display:flex; justify-content:space-between;">
                        <span><b>Sample Analyzed:</b> {ins.get('sample_size', 'N/A')}</span>
                        <span style="color:#64748B;"><i>Limitation: {ins.get('limitation', '')}</i></span>
                    </div>
                </div>
                """, unsafe_allow_html=True)

    with tab_ml_predict:
        st.markdown("### 🔮 **Score Prediction Model (Supervised Random Forest)**")
        st.markdown("<span style='color:#9CA3AF;'>Quantifies how much each farming activity (harvests, sales, land expansion) predicts final episode scores.</span>", unsafe_allow_html=True)

        if st.button("🚀 Train Machine Learning Regressor", key="train_rf"):
            with st.spinner("Training Random Forest Regressor and preventing target leakage..."):
                res = strat.train_score_predictor()
                if "error" in res:
                    st.warning(res["error"])
                else:
                    m1, m2, m3, m4 = st.columns(4)
                    m1.metric("Model R² Score", f"{res['r2']}")
                    m2.metric("Mean Absolute Error (MAE)", f"${res['mae']:,.2f}")
                    m3.metric("Root Mean Squared Error", f"${res['rmse']:,.2f}")
                    m4.metric("Train / Test Records", f"{res['train_size']} / {res['test_size']}")

                    st.markdown("#### ⚖️ **Feature Importance Ranking**")
                    st.dataframe(res["feature_importances"], use_container_width=True, hide_index=True)

                    st.markdown(f"""
                    <div style="background:rgba(16, 185, 129, 0.1); border:1px solid #10B981; padding:12px; border-radius:6px; margin-top:14px; font-size:0.85rem; color:#ECFDF5;">
                        <b>Scientific Note:</b> {res['disclaimer']}
                    </div>
                    """, unsafe_allow_html=True)
        else:
            st.info("Click the button above to run feature importance and score prediction modeling.")

    with tab_ml_cluster:
        st.markdown("### 🎯 **Farming Strategy Archetypes (K-Means & 2D PCA)**")
        st.markdown("<span style='color:#9CA3AF;'>Discovers organic strategic archetypes among competing agents without prior labeling.</span>", unsafe_allow_html=True)

        n_clusters = st.slider("Select Target Clusters (Archetypes)", min_value=2, max_value=5, value=3)

        if st.button("🧩 Discover Strategy Clusters", key="run_kmeans"):
            with st.spinner("Running K-Means clustering and PCA dimensionality reduction..."):
                cluster_res = strat.run_strategy_clustering(n_clusters=n_clusters)
                if "error" in cluster_res:
                    st.warning(cluster_res["error"])
                else:
                    st.plotly_chart(Visualizer.plot_clusters_scatter(cluster_res), use_container_width=True)

                    st.markdown("#### 🏷️ **Archetype Profiles**")
                    c_sum_df = pd.DataFrame(cluster_res["summaries"])
                    st.dataframe(
                        c_sum_df,
                        use_container_width=True,
                        column_config={
                            "cluster_id": "Cluster #",
                            "name": "Archetype Descriptor",
                            "count": "Episodes",
                            "avg_score": st.column_config.NumberColumn("Avg Score ($)", format="$%.2f"),
                            "avg_harvests": "Avg Harvests",
                            "avg_sales": st.column_config.NumberColumn("Avg Sales ($)", format="$%.2f"),
                            "avg_quadrants": "Avg Land Quadrants",
                            "avg_livestock": "Avg Livestock Animals"
                        },
                        hide_index=True
                    )
        else:
            st.info("Click the button above to partition episodes into unsupervised strategy archetypes.")
