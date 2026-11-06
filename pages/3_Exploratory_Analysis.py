"""Page 3 – Exploratory analysis of the auxiliary datasets (oil, holidays, transactions, stores).

Sales-based EDA (Module 3 A–G) needs train.csv and is shown once that file is present.
"""
import plotly.express as px
import streamlit as st

from src.data_loader import find_dataset_files
from src.preprocessing import clean_oil, build_holiday_flags
from ui import apply_styles, cached_dataset, dataset_folder, page_header, require_files, sidebar_data_status

st.set_page_config(page_title="Exploratory Analysis · SmartSales AI", layout="wide")
apply_styles()
sidebar_data_status()
page_header("Exploratory Analysis", "Store traffic, oil prices, holidays and store profile.")

present = find_dataset_files(dataset_folder())
if not present:
    st.error("No dataset files were found. Check the dataset folder in the sidebar.")
    st.stop()

tab_tx, tab_oil, tab_hol, tab_store = st.tabs(["Transactions", "Oil price", "Holidays", "Stores"])

with tab_tx:
    if require_files(["transactions"]):
        tx = cached_dataset("transactions", dataset_folder())
        stores = cached_dataset("stores", dataset_folder()) if "stores" in present else None

        f1, f2 = st.columns(2)
        with f1:
            start, end = st.date_input("Date range", value=(tx["date"].min().date(), tx["date"].max().date()),
                                       min_value=tx["date"].min().date(), max_value=tx["date"].max().date())
        with f2:
            store_choice = st.selectbox("Store", ["All stores"] + sorted(tx["store_nbr"].unique().tolist()))

        view = tx[(tx["date"].dt.date >= start) & (tx["date"].dt.date <= end)]
        if store_choice != "All stores":
            view = view[view["store_nbr"] == store_choice]

        daily = view.groupby("date", as_index=False)["transactions"].sum()
        fig = px.line(daily, x="date", y="transactions",
                      labels={"date": "Date", "transactions": "Transactions per day"},
                      title="Daily transactions")
        fig.update_layout(height=360, margin=dict(l=10, r=10, t=50, b=10))
        st.plotly_chart(fig, width="stretch")

        order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
        weekday = view.assign(day=view["date"].dt.day_name()).groupby("day")["transactions"].mean()
        weekday = weekday.reindex(order).rename("transactions").reset_index()
        fig = px.bar(weekday, x="day", y="transactions", title="Average transactions by weekday",
                     labels={"day": "Weekday", "transactions": "Average transactions"})
        fig.update_layout(height=320, margin=dict(l=10, r=10, t=50, b=10))
        st.plotly_chart(fig, width="stretch")

        if stores is not None:
            by_city = (view.merge(stores[["store_nbr", "city"]], on="store_nbr", how="left")
                       .groupby("city", as_index=False)["transactions"].sum()
                       .sort_values("transactions", ascending=False).head(15))
            fig = px.bar(by_city, x="city", y="transactions", title="Top 15 cities by total transactions",
                         labels={"city": "City", "transactions": "Total transactions"})
            fig.update_layout(height=340, margin=dict(l=10, r=10, t=50, b=10))
            st.plotly_chart(fig, width="stretch")

with tab_oil:
    if require_files(["oil"]):
        oil_raw = cached_dataset("oil", dataset_folder())
        oil, report = clean_oil(oil_raw)
        fig = px.line(oil, x="date", y="dcoilwtico", labels={"date": "Date", "dcoilwtico": "USD per barrel"},
                      title="WTI crude oil price (forward-filled)")
        fig.update_layout(height=360, margin=dict(l=10, r=10, t=50, b=10))
        st.plotly_chart(fig, width="stretch")
        c1, c2, c3 = st.columns(3)
        c1.metric("Blank price rows in source", report["oil_missing_before_fill"])
        c2.metric("Leading rows back-filled", report["oil_leading_rows_backfilled"])
        c3.metric("Rows still missing", report["oil_missing_after_fill"])

with tab_hol:
    if require_files(["holidays"]):
        hol = cached_dataset("holidays", dataset_folder())
        counts = hol.groupby(["type", "locale"]).size().reset_index(name="count")
        fig = px.bar(counts, x="type", y="count", color="locale", barmode="group",
                     title="Holiday and event records by type and locale",
                     labels={"type": "Type", "count": "Records", "locale": "Locale"})
        fig.update_layout(height=340, margin=dict(l=10, r=10, t=50, b=10))
        st.plotly_chart(fig, width="stretch")
        flags = build_holiday_flags(hol)
        st.markdown(
            f'<div class="ss-note">After removing transferred holidays, {int(flags["is_national_holiday"].sum())} '
            f"national holiday dates and {int(flags['is_event'].sum())} national event dates are flagged. "
            "Holidays that were moved are represented by their Transfer row.</div>",
            unsafe_allow_html=True,
        )
        st.dataframe(hol.sort_values("date", ascending=False).head(200), hide_index=True, width="stretch")

with tab_store:
    if require_files(["stores"]):
        stores = cached_dataset("stores", dataset_folder())
        c1, c2 = st.columns(2)
        with c1:
            fig = px.histogram(stores, x="type", title="Stores by type", labels={"type": "Store type", "count": "Stores"})
            fig.update_layout(height=320, margin=dict(l=10, r=10, t=50, b=10))
            st.plotly_chart(fig, width="stretch")
        with c2:
            fig = px.histogram(stores, x="cluster", title="Stores by cluster", labels={"cluster": "Cluster"})
            fig.update_layout(height=320, margin=dict(l=10, r=10, t=50, b=10))
            st.plotly_chart(fig, width="stretch")
        st.dataframe(stores, hide_index=True, width="stretch")
