"""Page 2 – Data Explorer: preview, structure, quality checks and downloads (Module 1)."""
import pandas as pd
import streamlit as st

from config import DATASET_FILES
from src.data_loader import find_dataset_files, summarise_dataset
from ui import apply_styles, cached_dataset, dataset_folder, page_header, sidebar_data_status

st.set_page_config(page_title="Data Explorer · SmartSales AI", layout="wide")
apply_styles()
sidebar_data_status()
page_header("Data Explorer", "Inspect every dataset file: structure, quality and summary statistics.")

present = find_dataset_files(dataset_folder())
if not present:
    st.error("No dataset files were found in the dataset folder. Check the path in the sidebar.")
    st.stop()

options = {DATASET_FILES[k]: k for k in present}
choice = st.selectbox("Dataset", list(options.keys()))
key = options[choice]
df = cached_dataset(key, dataset_folder())
summary = summarise_dataset(df)

c1, c2, c3, c4 = st.columns(4)
c1.metric("Rows", f"{summary['rows']:,}")
c2.metric("Columns", summary["columns"])
c3.metric("Memory", f"{summary['memory_mb']} MB")
c4.metric("Duplicate rows", f"{summary['duplicate_rows']:,}")

if summary["date_range"]:
    start, end = summary["date_range"]
    st.markdown(f'<div class="ss-note">Date range: <b>{start:%d %b %Y}</b> to <b>{end:%d %b %Y}</b></div>',
                unsafe_allow_html=True)

tab_preview, tab_columns, tab_quality, tab_stats, tab_download = st.tabs(
    ["Preview", "Columns", "Missing and duplicates", "Statistics", "Download"]
)

with tab_preview:
    n = st.slider("Rows to show", 5, 50, 10)
    st.markdown("**First rows**")
    st.dataframe(df.head(n), width="stretch", hide_index=True)
    st.markdown("**Last rows**")
    st.dataframe(df.tail(n), width="stretch", hide_index=True)

with tab_columns:
    info = pd.DataFrame({
        "column": summary["column_names"],
        "dtype": [summary["dtypes"][c] for c in summary["column_names"]],
        "missing": [summary["missing_values"][c] for c in summary["column_names"]],
        "role": ["numeric" if c in summary["numeric_columns"] else "categorical/date" for c in summary["column_names"]],
    })
    st.dataframe(info, width="stretch", hide_index=True)

with tab_quality:
    missing = pd.DataFrame({"column": list(summary["missing_values"].keys()),
                            "missing": list(summary["missing_values"].values())})
    missing["percent"] = (missing["missing"] / summary["rows"] * 100).round(3)
    st.markdown("**Missing values per column**")
    st.dataframe(missing, width="stretch", hide_index=True)
    if summary["duplicate_rows"]:
        st.markdown(f'<div class="ss-warning">{summary["duplicate_rows"]:,} fully duplicated rows found.</div>',
                    unsafe_allow_html=True)
    else:
        st.success("No fully duplicated rows.")

with tab_stats:
    if summary["numeric_columns"]:
        st.dataframe(df[summary["numeric_columns"]].describe().T, width="stretch")
    if summary["categorical_columns"]:
        st.markdown("**Categorical columns: distinct values**")
        st.dataframe(pd.DataFrame({
            "column": summary["categorical_columns"],
            "unique_values": [df[c].nunique() for c in summary["categorical_columns"]],
        }), hide_index=True, width="stretch")

with tab_download:
    csv_bytes = df.to_csv(index=False).encode("utf-8")
    st.download_button("Download this dataset as CSV", csv_bytes, file_name=choice, mime="text/csv")
    st.caption("The download is a copy. The original file in the dataset folder is never changed.")
