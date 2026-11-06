"""SmartSales AI – main dashboard (entry point).

Run with:  streamlit run app.py
"""
import pandas as pd
import plotly.express as px
import streamlit as st

from config import APP_NAME, DATASET_FILES
from src.data_loader import find_dataset_files
from ui import apply_styles, cached_dataset, dataset_folder, kpi_card, page_header, sidebar_data_status

st.set_page_config(page_title=APP_NAME, page_icon=None, layout="wide", initial_sidebar_state="expanded")
apply_styles()
sidebar_data_status()
page_header("Executive Dashboard", "Overview of the store network, auxiliary signals and forecast readiness.")

present = find_dataset_files(dataset_folder())

# ---- KPI row -------------------------------------------------------------------------------
c1, c2, c3, c4 = st.columns(4)
if "stores" in present:
    stores = cached_dataset("stores", dataset_folder())
    with c1:
        kpi_card("Total stores", f"{stores['store_nbr'].nunique():,}", "From stores.csv")
    with c2:
        kpi_card("Cities covered", f"{stores['city'].nunique():,}", "Unique cities in stores.csv")
if "test" in present:
    test = cached_dataset("test", dataset_folder())
    with c3:
        kpi_card("Product families", f"{test['family'].nunique():,}", "From test.csv")
if "transactions" in present:
    tx = cached_dataset("transactions", dataset_folder())
    with c4:
        kpi_card("Transactions recorded", f"{tx['transactions'].sum():,.0f}",
                 f"{tx['date'].min():%d %b %Y} to {tx['date'].max():%d %b %Y}")

st.markdown("")
s1, s2, s3 = st.columns(3)
with s1:
    kpi_card("Historical sales", "Needs train.csv", "Sales KPIs are calculated only from train.csv")
with s2:
    kpi_card("Forecasted sales", "Needs train.csv", "Forecasts need historical sales for training")
with s3:
    kpi_card("Best performing store", "Needs train.csv", "Ranked from actual sales")

# ---- Transactions trend ----------------------------------------------------------------------
if "transactions" in present:
    st.subheader("Daily transactions across the store network")
    daily = tx.groupby("date", as_index=False)["transactions"].sum()
    fig = px.line(daily, x="date", y="transactions",
                  labels={"date": "Date", "transactions": "Total transactions per day"})
    fig.update_layout(height=340, margin=dict(l=10, r=10, t=20, b=10), hovermode="x unified")
    st.plotly_chart(fig, width="stretch")
    st.caption("Transactions show store traffic. They are available only up to the last transaction date and are not used as a future feature.")

# ---- Oil price --------------------------------------------------------------------------------
if "oil" in present:
    oil = cached_dataset("oil", dataset_folder())
    st.subheader("Daily oil price (WTI)")
    fig = px.line(oil, x="date", y="dcoilwtico", labels={"date": "Date", "dcoilwtico": "Price (USD per barrel)"})
    fig.update_layout(height=300, margin=dict(l=10, r=10, t=20, b=10))
    st.plotly_chart(fig, width="stretch")
    missing_oil = int(oil["dcoilwtico"].isna().sum())
    st.markdown(
        f'<div class="ss-note">{missing_oil} oil price rows are blank in the source file. '
        "They are forward-filled from the previous trading day, so no future price is used.</div>",
        unsafe_allow_html=True,
    )

# ---- Dataset readiness ------------------------------------------------------------------------
st.subheader("Dataset readiness")
readiness = pd.DataFrame(
    [{"File": name, "Status": "Available" if key in present else "Missing",
      "Purpose": {
          "train": "Historical sales (required for forecasting and sales KPIs)",
          "test": "Future dates to forecast",
          "stores": "Store city, state, type and cluster",
          "oil": "Daily oil price",
          "holidays": "Holidays and events",
          "transactions": "Daily store transactions",
          "sample_submission": "Kaggle submission format",
      }[key]}
     for key, name in DATASET_FILES.items()]
)
st.dataframe(readiness, hide_index=True, width="stretch")

if "train" not in present:
    st.warning(
        "train.csv is not in the dataset folder. It holds the historical sales that the forecasting models learn from. "
        "Add it to data/raw/ (or set SMARTSALES_DATA_DIR) to unlock the sales KPIs and forecasting pages."
    )
