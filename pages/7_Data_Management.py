"""
Page 7: Data Management & Ingestion
Kaggriculture - Agricultural Simulation Analysis & Strategy Platform
"""
import streamlit as st
import pandas as pd
from pathlib import Path

from config import APP_TITLE, CUSTOM_CSS, SAMPLE_CSV_PATH, SAMPLE_JSON_PATH
from src.state_manager import render_sidebar_status_and_filters
from src.data_loader import DataLoader
from src.data_validator import DataValidator
from src.data_cleaner import DataCleaner
from src.schema_mapper import SchemaMapper, ALIAS_MAP

st.set_page_config(page_title=f"Data Management | {APP_TITLE}", page_icon="🌱", layout="wide")
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

filtered_df, filtered_actions_df = render_sidebar_status_and_filters()

st.title("🗄️ Dataset Ingestion & Schema Management")
st.markdown("<p style='color:#9CA3AF;'>Upload official Kaggle simulation files, inspect raw schemas, run validation suites, and apply safe non-destructive cleaning transformations.</p>", unsafe_allow_html=True)

# Tabs
tab_upload, tab_schema, tab_clean, tab_samples = st.tabs([
    "📥 Ingestion & Upload",
    "🗺️ Schema Mapping & Validation",
    "🧼 Non-Destructive Cleaning",
    "📦 Sample Benchmarks & Templates"
])

with tab_upload:
    st.markdown("### 📤 **Upload Kaggriculture Files**")
    st.markdown("""
    Support formats: **CSV**, **JSON**, **JSONL**, **Parquet**.
    Supports standard tabular episode files or nested Kaggle simulation replay files.
    """)

    uploaded_files = st.file_uploader(
        "Upload File(s)",
        type=["csv", "json", "jsonl", "parquet"],
        accept_multiple_files=True,
        help="Select one or more files from your local system."
    )

    if uploaded_files:
        if st.button("⚡ Ingest & Process Uploaded Files", use_container_width=True):
            try:
                with st.spinner("Parsing and standardizing datasets..."):
                    if len(uploaded_files) == 1:
                        ep_df, act_df, meta = DataLoader.load_from_file(uploaded_files[0])
                    else:
                        ep_df, act_df, meta = DataLoader.load_multiple_files(uploaded_files)

                    if not ep_df.empty:
                        st.session_state["episodes_df"] = ep_df
                        st.session_state["actions_df"] = act_df
                        st.session_state["dataset_metadata"] = {
                            "source": "User Upload",
                            "filename": meta.get("source_filename") or f"{len(uploaded_files)} files",
                            "is_sample": False
                        }
                        st.session_state["validation_report"] = DataValidator.validate_dataset(ep_df)
                        st.success(f"Successfully loaded {len(ep_df):,} episodes from {len(uploaded_files)} file(s)!")
                        st.rerun()
                    else:
                        st.error("No valid simulation episodes could be extracted from the uploaded file(s).")
            except Exception as e:
                st.error(f"Error reading uploaded file: {str(e)}")

    st.markdown("---")
    st.markdown("### 🌐 **Kaggle API Automated Download**")
    st.markdown("""
    If you have configured your Kaggle API key (`kaggle.json` or environment variables `KAGGLE_USERNAME` and `KAGGLE_KEY`),
    you can fetch the competition episodes dataset directly:
    """)
    dataset_ref = st.text_input("Kaggle Dataset Handle", value="kaggle/kaggriculture-episodes-2026-09-08")
    if st.button("⬇️ Download via Kaggle API"):
        with st.spinner("Downloading dataset from Kaggle..."):
            success, msg = DataLoader.download_from_kaggle(dataset_ref=dataset_ref)
            if success:
                st.success(msg)
            else:
                st.info(msg)

with tab_schema:
    st.markdown("### 🔍 **Schema Inspection & Column Mapping**")
    active_df = st.session_state.get("episodes_df", pd.DataFrame())

    if not active_df.empty:
        s1, s2, s3 = st.columns(3)
        s1.metric("Total Rows", f"{len(active_df):,}")
        s2.metric("Total Columns", f"{len(active_df.columns)}")
        s3.metric("Duplicate IDs", f"{active_df.duplicated(subset=['episode_id']).sum() if 'episode_id' in active_df.columns else 0}")

        st.markdown("#### 🗺️ **Canonical Column Mapping Status**")
        current_mappings = SchemaMapper.detect_column_mappings(active_df)
        mapping_records = []
        for canonical, source in current_mappings.items():
            status = "Mapped ✅" if source else "Unavailable ⚠️"
            mapping_records.append({
                "Canonical Field": canonical,
                "Source Column in Data": source or "None",
                "Status": status
            })
        st.dataframe(pd.DataFrame(mapping_records), use_container_width=True, hide_index=True)

        st.markdown("#### 🛡️ **Validation Diagnostics**")
        val_report = st.session_state.get("validation_report", {})
        issues = val_report.get("issues", [])
        if issues:
            for iss in issues:
                color = "#EF4444" if iss["severity"] == "ERROR" else ("#F59E0B" if iss["severity"] == "WARNING" else "#38BDF8")
                st.markdown(f"<span style='color:{color}; font-weight:bold;'>[{iss['severity']}]</span> {iss['message']}", unsafe_allow_html=True)
        else:
            st.success("All schema integrity checks passed.")
    else:
        st.info("No active dataset loaded to inspect.")

with tab_clean:
    st.markdown("### 🧼 **Non-Destructive Data Cleaning Operations**")
    st.markdown("<p style='color:#9CA3AF;'>Transformations create a working cleaned copy and preserve original uploaded records.</p>", unsafe_allow_html=True)

    active_df = st.session_state.get("episodes_df", pd.DataFrame())
    if not active_df.empty:
        cleaner = DataCleaner(active_df)

        c_col1, c_col2 = st.columns(2)
        with c_col1:
            st.markdown("#### 1. Deduplication")
            if st.button("🧹 Remove Duplicate Episode IDs"):
                cleaned, removed = cleaner.remove_duplicates()
                st.session_state["episodes_df"] = cleaned
                st.session_state["cleaner_audit_log"].extend(cleaner.get_audit_log())
                st.success(f"Removed {removed} duplicate rows.")
                st.rerun()

            st.markdown("#### 2. Missing Value Imputation")
            num_strat = st.selectbox("Numerical Imputation", ["median", "mean", "zero"])
            if st.button("🩹 Impute Missing Values"):
                cleaned = cleaner.handle_missing_values(num_strategy=num_strat)
                st.session_state["episodes_df"] = cleaned
                st.session_state["cleaner_audit_log"].extend(cleaner.get_audit_log())
                st.success("Imputed missing values.")
                st.rerun()

        with c_col2:
            st.markdown("#### 3. Outlier Trimming")
            out_method = st.selectbox("Outlier Method", ["IQR", "ZScore"])
            if st.button("✂️ Trim Score Outliers"):
                cleaned, removed = cleaner.filter_score_outliers(method=out_method.lower(), threshold=1.5)
                st.session_state["episodes_df"] = cleaned
                st.session_state["cleaner_audit_log"].extend(cleaner.get_audit_log())
                st.success(f"Trimmed {removed} outlier matches.")
                st.rerun()

            st.markdown("#### 4. Reset & Rollback")
            if st.button("↩️ Reset to Original Uploaded State"):
                if SAMPLE_CSV_PATH.exists():
                    ep_df, _, _ = DataLoader.load_from_file(SAMPLE_CSV_PATH)
                    st.session_state["episodes_df"] = ep_df
                    st.session_state["cleaner_audit_log"].append("Reverted dataset to baseline state.")
                    st.success("Dataset reset.")
                    st.rerun()

        st.markdown("#### 📜 **Transformation Audit Trail**")
        audit_log = st.session_state.get("cleaner_audit_log", [])
        if audit_log:
            for log_entry in audit_log:
                st.markdown(f"- `{log_entry}`")
        else:
            st.info("No cleaning transformations have been applied yet.")

        # Download Cleaned Dataset
        st.markdown("#### 💾 **Download Working Dataset**")
        csv_data = active_df.to_csv(index=False)
        st.download_button(
            label="📥 Download Cleaned Dataset (CSV)",
            data=csv_data,
            file_name="kaggriculture_cleaned_episodes.csv",
            mime="text/csv",
            use_container_width=True
        )
    else:
        st.info("Upload or load a dataset first.")

with tab_samples:
    st.markdown("### 📦 **Sample Benchmarks & Template Datasets**")
    st.markdown("""
    If you don't yet have the Kaggle dataset downloaded, you can load or download
    pre-packaged simulation benchmarks clearly marked as **SAMPLE DATA**:
    """)

    samp1, samp2 = st.columns(2)
    with samp1:
        st.markdown("""
        <div class="kag-card">
            <h4 style="color:#34D399; margin:0 0 8px 0;">Sample Episodes CSV (120 Matches)</h4>
            <p style="color:#9CA3AF; font-size:0.85rem;">Realistic simulation matches between 8 benchmark agents including scores, harvests, and market sales.</p>
        </div>
        """, unsafe_allow_html=True)
        if st.button("🔄 Reload Sample Benchmark Episodes"):
            ep_df, _, meta = DataLoader.load_from_file(SAMPLE_CSV_PATH)
            st.session_state["episodes_df"] = ep_df
            st.session_state["dataset_metadata"] = {
                "source": "Sample Benchmark Dataset",
                "filename": SAMPLE_CSV_PATH.name,
                "is_sample": True
            }
            st.success("Sample episodes active!")
            st.rerun()

    with samp2:
        st.markdown("""
        <div class="kag-card">
            <h4 style="color:#34D399; margin:0 0 8px 0;">Sample Replay JSON Shard (1,200 Actions)</h4>
            <p style="color:#9CA3AF; font-size:0.85rem;">Granular turn-by-turn action replay stream with crop planting, watering, market trades, and rewards.</p>
        </div>
        """, unsafe_allow_html=True)
        if st.button("🔄 Reload Sample Replay Shards"):
            _, act_df, _ = DataLoader.load_from_file(SAMPLE_JSON_PATH)
            st.session_state["actions_df"] = act_df
            st.success("Sample replay stream loaded!")
            st.rerun()
