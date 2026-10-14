"""
Strategy Insights & Machine Learning Module for Kaggriculture.
Derives evidence-based farming insights, strategy clusters, and predictive models
from actual simulation data.
"""
from typing import Dict, Any, List, Optional, Tuple
import pandas as pd
import numpy as np


class StrategyInsights:
    """Extracts data-driven strategic rules, behavioral patterns, and ML models."""

    def __init__(self, episodes_df: pd.DataFrame, actions_df: Optional[pd.DataFrame] = None):
        self.df = episodes_df.copy()
        self.actions_df = actions_df.copy() if actions_df is not None else None

    def derive_evidence_based_insights(self) -> List[Dict[str, Any]]:
        """
        Derives structured insights comparing top quartile (high-performing) vs bottom quartile episodes.
        Every insight includes:
        - Observation
        - Supporting Metric
        - Sample Size
        - Limitations
        - Practical Interpretation
        """
        insights: List[Dict[str, Any]] = []

        if len(self.df) < 8:
            return [{
                "title": "Insufficient Sample Size",
                "observation": f"The dataset currently contains only {len(self.df)} records.",
                "metric": "N < 8 episodes",
                "sample_size": len(self.df),
                "limitation": "Reliable statistical pattern extraction requires at least 8 completed episodes.",
                "interpretation": "Upload additional simulation episodes to activate automated strategy rule extraction."
            }]

        # Analyze scores distribution to split high vs low
        score_col = "score_a" if "score_a" in self.df.columns else None
        if not score_col:
            return [{
                "title": "Missing Score Attributes",
                "observation": "No numerical score column detected in the dataset.",
                "metric": "N/A",
                "sample_size": len(self.df),
                "limitation": "Score values are required to categorize high vs low performance.",
                "interpretation": "Map score attributes in Data Management."
            }]

        # Pool agent episodes
        pooled_records = []
        for _, row in self.df.iterrows():
            if "score_a" in row and pd.notna(row["score_a"]):
                pooled_records.append({
                    "score": float(row["score_a"]),
                    "harvests": float(row.get("crops_harvested_a", 0) or 0),
                    "sales": float(row.get("market_sales_a", 0) or 0),
                    "quadrants": float(row.get("land_quadrants_a", 1) or 1),
                    "livestock": float(row.get("livestock_count_a", 0) or 0),
                    "actions": float(row.get("actions_a_count", 0) or 0)
                })
            if "score_b" in row and pd.notna(row["score_b"]):
                pooled_records.append({
                    "score": float(row["score_b"]),
                    "harvests": float(row.get("crops_harvested_b", 0) or 0),
                    "sales": float(row.get("market_sales_b", 0) or 0),
                    "quadrants": float(row.get("land_quadrants_b", 1) or 1),
                    "livestock": float(row.get("livestock_count_b", 0) or 0),
                    "actions": float(row.get("actions_b_count", 0) or 0)
                })

        p_df = pd.DataFrame(pooled_records)
        if len(p_df) < 8:
            return insights

        q75 = p_df["score"].quantile(0.75)
        q25 = p_df["score"].quantile(0.25)

        high_perf = p_df[p_df["score"] >= q75]
        low_perf = p_df[p_df["score"] <= q25]

        n_high = len(high_perf)
        n_low = len(low_perf)

        # 1. Harvest Volume Impact
        if p_df["harvests"].sum() > 0:
            avg_h_high = high_perf["harvests"].mean()
            avg_h_low = low_perf["harvests"].mean()
            diff_h = avg_h_high - avg_h_low
            pct_h = ((diff_h / avg_h_low) * 100) if avg_h_low > 0 else 0
            insights.append({
                "title": "Crop Harvest Volume & Yield Pacing",
                "observation": f"Top quartile agents harvested an average of {avg_h_high:.1f} crops compared to {avg_h_low:.1f} crops for bottom quartile agents.",
                "metric": f"+{diff_h:.1f} harvests ({pct_h:+.1f}%) in high performers",
                "sample_size": f"{n_high} high vs {n_low} low records (total {len(p_df)})",
                "limitation": "Causal direction requires checking whether early farm expansion or seed luck enabled higher harvest yields.",
                "interpretation": "High-tier agents maintain active crop cycles and minimize idle crop bed turns."
            })

        # 2. Market Sales Timing & Revenue
        if p_df["sales"].sum() > 0:
            avg_s_high = high_perf["sales"].mean()
            avg_s_low = low_perf["sales"].mean()
            diff_s = avg_s_high - avg_s_low
            insights.append({
                "title": "Market Liquidation & Capital Conversion",
                "observation": f"High performers generated ${avg_s_high:,.2f} in market sales versus ${avg_s_low:,.2f} for lower performers.",
                "metric": f"${diff_s:+,.2f} revenue differential",
                "sample_size": f"{n_high} high vs {n_low} low records",
                "limitation": "Market price fluctuations during the 30-day season can amplify returns independently of bot logic.",
                "interpretation": "Converting perishable crops into liquid bank balance before end-of-season cutoff is vital for peak scores."
            })

        # 3. Land Expansion Threshold
        if p_df["quadrants"].nunique() > 1:
            avg_q_high = high_perf["quadrants"].mean()
            avg_q_low = low_perf["quadrants"].mean()
            insights.append({
                "title": "Farm Quadrant Scaling Strategy",
                "observation": f"Winning agents operated across an average of {avg_q_high:.1f} land quadrants compared to {avg_q_low:.1f} for bottom agents.",
                "metric": f"{avg_q_high:.1f} vs {avg_q_low:.1f} average quadrants",
                "sample_size": f"{n_high} high vs {n_low} low records",
                "limitation": "Purchasing quadrants incurs upfront capital expenditure which can backfire if done too late in the season.",
                "interpretation": "Early expansion allows compounding crop yields if the farmhand staffing or watering throughput can keep pace."
            })

        # 4. Livestock Diversification
        if p_df["livestock"].sum() > 0:
            avg_l_high = high_perf["livestock"].mean()
            avg_l_low = low_perf["livestock"].mean()
            insights.append({
                "title": "Livestock Portfolio Diversification",
                "observation": f"Top quartile agents maintained {avg_l_high:.1f} livestock animals on average vs {avg_l_low:.1f} for bottom agents.",
                "metric": f"{avg_l_high:.1f} vs {avg_l_low:.1f} animals",
                "sample_size": f"{n_high} high vs {n_low} low records",
                "limitation": "Livestock requires ongoing feed purchases; drought or high feed costs can penalize over-reliance.",
                "interpretation": "A balanced mix of crops and livestock stabilizes income against crop spoilage and weed blights."
            })

        # 5. Granular Action Insights (if available)
        if self.actions_df is not None and not self.actions_df.empty:
            act_counts = self.actions_df["action"].value_counts()
            top_action = act_counts.index[0]
            insights.append({
                "title": "Dominant Operational Priority",
                "observation": f"The single most frequent action across the simulation was '{top_action}' ({act_counts.iloc[0]} occurrences).",
                "metric": f"{((act_counts.iloc[0] / len(self.actions_df)) * 100):.1f}% of all actions",
                "sample_size": f"{len(self.actions_df)} recorded action events",
                "limitation": "Frequency alone does not measure efficiency or turn economics.",
                "interpretation": f"Automating and optimizing the '{top_action}' execution loop yields the highest leverage on match outcomes."
            })

        return insights

    def run_strategy_clustering(self, n_clusters: int = 3) -> Dict[str, Any]:
        """
        Performs K-Means clustering on episode feature vectors to discover distinct farming archetypes.
        Uses PCA for 2D visualization coordinates.
        """
        from sklearn.cluster import KMeans
        from sklearn.preprocessing import StandardScaler
        from sklearn.decomposition import PCA

        features = []
        records = []
        for _, row in self.df.iterrows():
            if "score_a" in row and pd.notna(row["score_a"]):
                records.append({
                    "episode_id": row.get("episode_id", "N/A"),
                    "agent": row.get("agent_a", "Agent A"),
                    "score": float(row["score_a"]),
                    "harvests": float(row.get("crops_harvested_a", 0) or 0),
                    "sales": float(row.get("market_sales_a", 0) or 0),
                    "quadrants": float(row.get("land_quadrants_a", 1) or 1),
                    "livestock": float(row.get("livestock_count_a", 0) or 0),
                    "actions": float(row.get("actions_a_count", 0) or 0)
                })

        if len(records) < n_clusters * 3:
            return {"error": f"Need at least {n_clusters * 3} records for meaningful clustering. Found {len(records)}."}

        feat_df = pd.DataFrame(records)
        num_cols = ["score", "harvests", "sales", "quadrants", "livestock", "actions"]
        X = feat_df[num_cols].values

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        clusters = kmeans.fit_predict(X_scaled)
        feat_df["cluster"] = clusters

        # 2D PCA projection
        pca = PCA(n_components=2, random_state=42)
        coords = pca.fit_transform(X_scaled)
        feat_df["pca_x"] = coords[:, 0]
        feat_df["pca_y"] = coords[:, 1]

        # Archetype labeling
        cluster_summaries = []
        cluster_names = [
            "Industrial Crop Specialist",
            "Balanced Agri-Enterprise",
            "Livestock & Market Arbitrage",
            "Conservative Subsistence",
            "High-Throughput Expander"
        ]

        for k in range(n_clusters):
            c_sub = feat_df[feat_df["cluster"] == k]
            name = cluster_names[k % len(cluster_names)]
            cluster_summaries.append({
                "cluster_id": k,
                "name": name,
                "count": len(c_sub),
                "avg_score": round(float(c_sub["score"].mean()), 2),
                "avg_harvests": round(float(c_sub["harvests"].mean()), 1),
                "avg_sales": round(float(c_sub["sales"].mean()), 2),
                "avg_quadrants": round(float(c_sub["quadrants"].mean()), 1),
                "avg_livestock": round(float(c_sub["livestock"].mean()), 1)
            })

        return {
            "cluster_df": feat_df,
            "summaries": cluster_summaries,
            "explained_variance": [round(float(v), 3) for v in pca.explained_variance_ratio_]
        }

    def train_score_predictor(self) -> Dict[str, Any]:
        """
        Trains an interpretable Random Forest Regressor to predict episode score from farming actions.
        Prevents target leakage and reports test evaluation metrics.
        """
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

        records = []
        for _, row in self.df.iterrows():
            if "score_a" in row and pd.notna(row["score_a"]):
                records.append({
                    "score": float(row["score_a"]),
                    "crops_harvested": float(row.get("crops_harvested_a", 0) or 0),
                    "market_sales": float(row.get("market_sales_a", 0) or 0),
                    "land_quadrants": float(row.get("land_quadrants_a", 1) or 1),
                    "livestock_count": float(row.get("livestock_count_a", 0) or 0),
                    "total_actions": float(row.get("actions_a_count", 0) or 0)
                })

        if len(records) < 20:
            return {"error": "Score prediction model requires at least 20 observations to prevent overfitting."}

        data = pd.DataFrame(records)
        X = data[["crops_harvested", "market_sales", "land_quadrants", "livestock_count", "total_actions"]]
        y = data["score"]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

        model = RandomForestRegressor(n_estimators=100, max_depth=4, random_state=42)
        model.fit(X_train, y_train)

        preds = model.predict(X_test)
        r2 = round(float(r2_score(y_test, preds)), 3)
        mae = round(float(mean_absolute_error(y_test, preds)), 2)
        rmse = round(float(np.sqrt(mean_squared_error(y_test, preds))), 2)

        # Feature importances
        importances = pd.DataFrame({
            "Feature": X.columns,
            "Importance (%)": (model.feature_importances_ * 100).round(2)
        }).sort_values(by="Importance (%)", ascending=False)

        return {
            "model": model,
            "r2": r2,
            "mae": mae,
            "rmse": rmse,
            "train_size": len(X_train),
            "test_size": len(X_test),
            "feature_importances": importances,
            "disclaimer": "Predicted scores demonstrate empirical relationships in past matches; they do not guarantee future competitive outcomes."
        }
