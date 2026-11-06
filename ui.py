"""Shared Streamlit helpers: styling, cached data access and page headers.

Kept separate from src/ so the ML and data code never imports Streamlit.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

from config import APP_NAME, DATA_DIR, TAGLINE
from src.data_loader import load_dataset, missing_dataset_files, find_dataset_files

ASSETS_DIR = Path(__file__).resolve().parent / "assets"


def apply_styles() -> None:
    """Inject the custom stylesheet used across every page."""
    css_path = ASSETS_DIR / "styles.css"
    if css_path.exists():
        st.markdown(f"<style>{css_path.read_text(encoding='utf-8')}</style>", unsafe_allow_html=True)


def page_header(title: str, subtitle: str = "") -> None:
    st.markdown(
        f"""<div class="ss-header"><div class="ss-brand">{APP_NAME}</div>
        <h1>{title}</h1><p>{subtitle or TAGLINE}</p></div>""",
        unsafe_allow_html=True,
    )


def kpi_card(label: str, value: str, note: str = "") -> None:
    st.markdown(
        f"""<div class="ss-card"><div class="ss-kpi-label">{label}</div>
        <div class="ss-kpi-value">{value}</div><div class="ss-kpi-note">{note}</div></div>""",
        unsafe_allow_html=True,
    )


@st.cache_data(show_spinner="Loading dataset...")
def cached_dataset(key: str, data_dir: str) -> pd.DataFrame:
    return load_dataset(key, data_dir)


def dataset_folder() -> str:
    """Folder chosen in the sidebar, defaulting to the configured data directory."""
    return st.session_state.get("data_dir", str(DATA_DIR))


def sidebar_data_status() -> None:
    """Show which dataset files are available and let the user point at another folder."""
    with st.sidebar:
        st.markdown(f"**{APP_NAME}**")
        st.caption(TAGLINE)
        st.divider()
        folder = st.text_input("Dataset folder", value=dataset_folder(), help="Folder containing the Kaggle CSV files.")
        st.session_state["data_dir"] = folder
        present = find_dataset_files(folder)
        missing = missing_dataset_files(folder)
        st.markdown(f"Files found: **{len(present)} of 7**")
        if missing:
            st.warning("Missing: " + ", ".join(missing))
        else:
            st.success("All dataset files found")


def require_files(keys: list[str]) -> bool:
    """Show a clear message and return False when required files are absent."""
    present = find_dataset_files(dataset_folder())
    absent = [k for k in keys if k not in present]
    if absent:
        from config import DATASET_FILES
        names = ", ".join(DATASET_FILES[k] for k in absent)
        st.error(f"This page needs: {names}. Place the file(s) in the dataset folder set in the sidebar.")
        return False
    return True
