"""Page 8 – Inventory planning driven by user-supplied stock values and a demand estimate."""
import plotly.graph_objects as go
import streamlit as st

from src.inventory_planning import InventoryInputs, calculate_inventory_plan, explain_plan
from ui import apply_styles, page_header, sidebar_data_status

st.set_page_config(page_title="Inventory Planning · SmartSales AI", layout="wide")
apply_styles()
sidebar_data_status()
page_header("Inventory Planning", "Turn a demand estimate and your stock position into reorder decisions.")

st.markdown(
    '<div class="ss-note">The Kaggle dataset does not contain stock levels, so enter your own figures. '
    "Demand can be typed in directly or taken from a forecast you generate elsewhere.</div>",
    unsafe_allow_html=True,
)

with st.form("inventory_form"):
    left, right = st.columns(2)
    with left:
        current_stock = st.number_input("Current stock (units)", min_value=0.0, value=500.0, step=10.0)
        avg_daily_demand = st.number_input("Average daily demand (units/day)", min_value=0.0, value=40.0, step=1.0,
                                           help="Use the average of a forecast, or your own estimate.")
        lead_time_days = st.number_input("Supplier lead time (days)", min_value=0, value=7, step=1)
    with right:
        safety_stock = st.number_input("Safety stock (units)", min_value=0.0, value=60.0, step=5.0)
        min_stock_level = st.number_input("Minimum stock level (units)", min_value=0.0, value=100.0, step=5.0)
        planning_horizon = st.selectbox("Planning horizon (days)", [7, 14, 30, 60], index=2)
        overstock_days = st.number_input("Overstock threshold (days of cover)", min_value=1, value=60, step=5)
    submitted = st.form_submit_button("Calculate inventory plan", type="primary")

if submitted:
    try:
        plan = calculate_inventory_plan(InventoryInputs(
            current_stock=current_stock,
            avg_daily_demand=avg_daily_demand,
            lead_time_days=int(lead_time_days),
            safety_stock=safety_stock,
            min_stock_level=min_stock_level,
            planning_horizon_days=int(planning_horizon),
            overstock_days_threshold=int(overstock_days),
        ))
    except ValueError as exc:
        st.error(f"Please correct the inputs: {exc}")
        st.stop()

    st.success(f"Recommendation: {plan['action']}")
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Reorder point", f"{plan['reorder_point']:,.0f} units")
    k2.metric("Recommended order", f"{plan['recommended_order_quantity']:,.0f} units")
    k3.metric("Estimated shortage", f"{plan['estimated_stock_shortage']:,.0f} units")
    k4.metric("Days of cover", "No demand" if plan["days_of_cover"] is None else f"{plan['days_of_cover']:,.1f} days")

    st.subheader("Stock position against the reorder point")
    fig = go.Figure()
    fig.add_bar(x=["Current stock", "Reorder point", "Target level"],
                y=[current_stock, plan["reorder_point"], plan["target_stock_level"]],
                marker_color=["#0F766E", "#B45309", "#334155"])
    fig.update_layout(height=340, yaxis_title="Units", margin=dict(l=10, r=10, t=20, b=10))
    st.plotly_chart(fig, width="stretch")

    st.subheader("How this was calculated")
    for line in explain_plan(plan):
        st.markdown(f"- {line}")
    with st.expander("Formulas"):
        st.markdown(
            """
- **Expected lead-time demand** = average daily demand x lead time
- **Reorder point** = expected lead-time demand + safety stock
- **Target stock level** = average daily demand x planning horizon + safety stock
- **Recommended order** = max(0, max(target level, minimum stock) - current stock)
- **Estimated shortage** = max(0, expected lead-time demand - current stock)
- **Days of cover** = current stock / average daily demand
            """
        )
    st.caption("These are planning estimates based on the demand you entered. They are not guaranteed outcomes.")
