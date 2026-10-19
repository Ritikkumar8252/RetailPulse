from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st
from pathlib import Path

import numpy as np

# ---------------- PAGE CONFIGURATION ----------------

st.set_page_config(
    page_title="RetailPulse | Business Analytics",
    page_icon="📊",
    layout="wide"
)

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data" / "processed"
OUTPUT_DIR = BASE_DIR / "outputs"


# ---------------- LOAD DATA ----------------

@st.cache_data
def load_sales_data():
    file_path = DATA_DIR / "retailpulse_positive_sales.csv"

    if not file_path.exists():
        raise FileNotFoundError(
            f"Sales dataset not found at: {file_path}"
        )

    df = pd.read_csv(
        file_path,
        parse_dates=["InvoiceDate"],
        dtype={
            "InvoiceNo": "string",
            "StockCode": "string",
            "Country": "string"
        }
    )

    df["Revenue"] = pd.to_numeric(
        df["Revenue"], errors="coerce"
    )

    df["Quantity"] = pd.to_numeric(
        df["Quantity"], errors="coerce"
    )

    return df

@st.cache_data
def load_customer_segments():
    file_path = OUTPUT_DIR / "retailpulse_customer_segments.csv"

    if not file_path.exists():
        raise FileNotFoundError(
            f"Customer segments file not found: {file_path}"
        )

    df = pd.read_csv(file_path)

    # The CustomerID may have been saved as the CSV index.
    if "CustomerID" not in df.columns:
        if "Unnamed: 0" in df.columns:
            df = df.rename(columns={"Unnamed: 0": "CustomerID"})
        else:
            df = df.rename(columns={df.columns[0]: "CustomerID"})

    df["Recency"] = pd.to_numeric(df["Recency"], errors="coerce")
    df["Frequency"] = pd.to_numeric(df["Frequency"], errors="coerce")
    df["Monetary"] = pd.to_numeric(df["Monetary"], errors="coerce")

    return df

@st.cache_data
def load_product_demand():
    file_path = OUTPUT_DIR / "retailpulse_daily_product_demand.csv"

    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    df = pd.read_csv(file_path, parse_dates=["Date"])
    df["StockCode"] = df["StockCode"].astype(str)

    return df


@st.cache_data
def load_product_summary():
    file_path = OUTPUT_DIR / "retailpulse_product_summary.csv"

    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    df = pd.read_csv(file_path)

    if "StockCode" not in df.columns:
        df = df.rename(columns={df.columns[0]: "StockCode"})

    df["StockCode"] = df["StockCode"].astype(str)

    return df


@st.cache_data
def load_model_evaluation():
    file_path = OUTPUT_DIR / "retailpulse_product_model_evaluation.csv"

    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    return pd.read_csv(file_path)


try:
    sales_df = load_sales_data()

except Exception as error:
    st.error(f"Could not load the sales dataset: {error}")
    st.stop()

try:
    customer_df = load_customer_segments()

except Exception as error:
    st.error(f"Could not load customer segments: {error}")
    st.stop()

try:
    product_demand_df = load_product_demand()
    product_summary_df = load_product_summary()
    model_evaluation_df = load_model_evaluation()

except Exception as error:
    st.error(f"Could not load forecasting data: {error}")
    st.stop()
# ---------------- SIDEBAR ----------------

st.sidebar.title("RetailPulse")
st.sidebar.caption("AI-Powered Customer Analytics")

page = st.sidebar.radio(
    "Navigation",
    [
        "Business Overview",
        "Sales Analytics",
        "Customer Segmentation",
        "Demand Forecasting",
        "Model Evaluation",
        "Business Recommendations",
        "Data Quality"
    ]
)


# ---------------- BUSINESS OVERVIEW ----------------

if page == "Business Overview":

    st.title("RetailPulse Business Overview")

    st.write(
        "Monitor business performance and explore sales trends."
    )

    total_revenue = sales_df["Revenue"].sum()
    total_orders = sales_df["InvoiceNo"].nunique()
    unique_customers = sales_df["CustomerID"].nunique()
    total_units = sales_df["Quantity"].sum()

    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "Total Revenue",
        f"£{total_revenue:,.2f}"
    )

    col2.metric(
        "Total Orders",
        f"{total_orders:,}"
    )

    col3.metric(
        "Known Customers",
        f"{unique_customers:,}"
    )

    col4.metric(
        "Units Sold",
        f"{total_units:,.0f}"
    )

    st.divider()

    monthly_revenue = (
        sales_df
        .dropna(subset=["InvoiceDate"])
        .assign(
            Month=lambda df: (
                df["InvoiceDate"].dt.to_period("M").astype(str)
            )
        )
        .groupby("Month", as_index=False)["Revenue"]
        .sum()
    )

    fig = px.line(
        monthly_revenue,
        x="Month",
        y="Revenue",
        markers=True,
        title="Monthly Revenue Trend"
    )

    fig.update_layout(
        xaxis_title="Month",
        yaxis_title="Revenue (£)"
    )

    st.plotly_chart(
        fig,
        width="stretch"
    )


# ---------------- SALES ANALYTICS ----------------

elif page == "Sales Analytics":

    st.title("Sales Analytics")
    st.write("Explore sales performance using interactive filters.")

    # Sidebar filters
    st.sidebar.subheader("Sales Filters")

    valid_dates = sales_df["InvoiceDate"].dropna()

    min_date = valid_dates.min().date()
    max_date = valid_dates.max().date()

    selected_dates = st.sidebar.date_input(
        "Select date range",
        value=(min_date, max_date),
        min_value=min_date,
        max_value=max_date
    )

    countries = sorted(
        sales_df["Country"].dropna().unique().tolist()
    )

    selected_countries = st.sidebar.multiselect(
        "Select countries",
        options=countries,
        default=["United Kingdom"]
        if "United Kingdom" in countries else countries
    )

    # Filter the sales data
    filtered_sales = sales_df.copy()

    if len(selected_dates) == 2:
        start_date, end_date = selected_dates

        filtered_sales = filtered_sales[
            filtered_sales["InvoiceDate"].dt.date.between(
                start_date, end_date
            )
        ]
    else:
        st.info("Select both a start date and an end date.")

    filtered_sales = filtered_sales[
        filtered_sales["Country"].isin(selected_countries)
    ]

    if filtered_sales.empty:
        st.warning(
            "No sales found for these filters. "
            "Try another date range or country."
        )

    else:
        # KPI cards
        col1, col2, col3 = st.columns(3)

        col1.metric(
            "Filtered Revenue",
            f"£{filtered_sales['Revenue'].sum():,.2f}"
        )

        col2.metric(
            "Filtered Orders",
            f"{filtered_sales['InvoiceNo'].nunique():,}"
        )

        col3.metric(
            "Filtered Units Sold",
            f"{filtered_sales['Quantity'].sum():,.0f}"
        )

        st.divider()

        # Top products
        top_products = (
            filtered_sales
            .groupby("StockCode", as_index=False)
            .agg(
                Revenue=("Revenue", "sum"),
                Units_Sold=("Quantity", "sum")
            )
            .nlargest(10, "Revenue")
        )

        fig_products = px.bar(
            top_products,
            x="Revenue",
            y="StockCode",
            orientation="h",
            title="Top 10 Products by Revenue"
        )

        fig_products.update_layout(
            yaxis={"categoryorder": "total ascending"},
            xaxis_title="Revenue (£)",
            yaxis_title="Product Code"
        )

        st.plotly_chart(
            fig_products,
            width="stretch"
        )

        # Revenue by country
        country_revenue = (
            filtered_sales
            .groupby("Country", as_index=False)["Revenue"]
            .sum()
            .nlargest(10, "Revenue")
        )

        fig_country = px.bar(
            country_revenue,
            x="Country",
            y="Revenue",
            title="Revenue by Country"
        )

        fig_country.update_layout(
            xaxis_title="Country",
            yaxis_title="Revenue (£)"
        )

        st.plotly_chart(
            fig_country,
            width="stretch"
        )

        # Filtered records
        st.subheader("Filtered Sales Records")

        st.dataframe(
            filtered_sales.head(500),
            width="stretch",
            hide_index=True
        )

        # Download filtered data
        csv_data = filtered_sales.to_csv(
            index=False
        ).encode("utf-8")

        st.download_button(
            "Download Filtered Sales CSV",
            data=csv_data,
            file_name="retailpulse_filtered_sales.csv",
            mime="text/csv"
        )

# ---------------- CUSTOMER SEGMENTATION ----------------

elif page == "Customer Segmentation":

    st.title("Customer Segmentation")
    st.write(
        "Explore customer behavior using Recency, Frequency, "
        "and Monetary (RFM) analysis."
    )

    # Check required columns
    required_columns = [
        "CustomerID",
        "Recency",
        "Frequency",
        "Monetary",
        "Cluster",
        "Segment"
    ]

    missing_columns = [
        col for col in required_columns
        if col not in customer_df.columns
    ]

    if missing_columns:
        st.error(
            f"Missing columns in customer dataset: {missing_columns}"
        )
        st.stop()

    # Summary metrics
    total_customers = customer_df["CustomerID"].nunique()
    total_segments = customer_df["Segment"].nunique()
    avg_monetary = customer_df["Monetary"].mean()
    median_recency = customer_df["Recency"].median()

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Segmented Customers", f"{total_customers:,}")
    col2.metric("Customer Segments", f"{total_segments}")
    col3.metric("Average Customer Spend", f"£{avg_monetary:,.2f}")
    col4.metric("Median Recency", f"{median_recency:.0f} days")

    st.divider()

    # Segment distribution
    segment_counts = (
        customer_df["Segment"]
        .value_counts()
        .rename_axis("Segment")
        .reset_index(name="Customers")
    )

    col1, col2 = st.columns(2)

    fig_distribution = px.bar(
        segment_counts,
        x="Segment",
        y="Customers",
        color="Segment",
        title="Customer Distribution by Segment",
        text="Customers"
    )

    fig_distribution.update_layout(
        showlegend=False,
        xaxis_title="Customer Segment",
        yaxis_title="Number of Customers"
    )

    col1.plotly_chart(
        fig_distribution,
        width="stretch"
    )

    # Monetary comparison
    segment_monetary = (
        customer_df.groupby("Segment", as_index=False)
        .agg(
            Average_Spend=("Monetary", "mean"),
            Median_Spend=("Monetary", "median")
        )
        .sort_values("Median_Spend", ascending=False)
    )

    fig_monetary = px.bar(
        segment_monetary,
        x="Segment",
        y="Median_Spend",
        color="Segment",
        title="Median Customer Spend by Segment"
    )

    fig_monetary.update_layout(
        showlegend=False,
        xaxis_title="Customer Segment",
        yaxis_title="Median Spend (£)"
    )

    col2.plotly_chart(
        fig_monetary,
        width="stretch"
    )

    st.divider()

    # RFM profiles
    st.subheader("Segment RFM Profiles")

    rfm_profiles = (
        customer_df.groupby("Segment", as_index=False)
        .agg(
            Customers=("CustomerID", "nunique"),
            Median_Recency=("Recency", "median"),
            Median_Frequency=("Frequency", "median"),
            Median_Monetary=("Monetary", "median")
        )
    )

    st.dataframe(
        rfm_profiles.round(2),
        width="stretch",
        hide_index=True
    )

    st.divider()

    # Customer filter
    st.subheader("Explore Individual Customers")

    available_segments = sorted(
        customer_df["Segment"].dropna().unique().tolist()
    )

    selected_segments = st.multiselect(
        "Select customer segments",
        options=available_segments,
        default=available_segments
    )

    filtered_customers = customer_df[
        customer_df["Segment"].isin(selected_segments)
    ].copy()

    st.write(
        f"Customers displayed: {len(filtered_customers):,}"
    )

    st.dataframe(
        filtered_customers[
            [
                "CustomerID",
                "Recency",
                "Frequency",
                "Monetary",
                "Cluster",
                "Segment"
            ]
        ].sort_values("Monetary", ascending=False),
        width="stretch",
        hide_index=True
    )

    # Download filtered customers
    csv_data = filtered_customers.to_csv(index=False).encode("utf-8")

    st.download_button(
        label="Download Customer Segments CSV",
        data=csv_data,
        file_name="retailpulse_filtered_customer_segments.csv",
        mime="text/csv"
    )

# ---------------- DEMAND FORECASTING ----------------

elif page == "Demand Forecasting":

    st.title("📈 Product Demand Forecasting")

    st.write(
        "Select a product and forecast horizon to estimate "
        "future daily demand."
    )

    # Check whether the product demand data is available
    if product_demand_df.empty:

        st.error("Product demand data is empty.")

    else:

        # Ensure Date is in datetime format
        product_demand_df["Date"] = pd.to_datetime(
            product_demand_df["Date"],
            errors="coerce"
        )

        product_demand_df.dropna(
            subset=["Date"],
            inplace=True
        )

        # Product selection
        available_products = sorted(
            product_demand_df["StockCode"]
            .dropna()
            .astype(str)
            .unique()
            .tolist()
        )

        col1, col2 = st.columns(2)

        with col1:

            selected_product = st.selectbox(
                "Select Product",
                options=available_products
            )

        with col2:

            forecast_days = st.selectbox(
                "Forecast Horizon",
                options=[7, 14, 30],
                index=0,
                format_func=lambda x: f"Next {x} days"
            )

        st.divider()

        # ------------------------------------------------
        # Prepare complete daily demand series
        # ------------------------------------------------

        product_data = product_demand_df[
            product_demand_df["StockCode"].astype(str)
            == selected_product
        ].copy()

        product_data["Units_Sold"] = pd.to_numeric(
            product_data["Units_Sold"],
            errors="coerce"
        ).fillna(0)

        # Aggregate in case a product has multiple rows per date
        product_data = (
            product_data
            .groupby("Date")["Units_Sold"]
            .sum()
            .sort_index()
        )

        if product_data.empty:

            st.warning("No demand history is available for this product.")

        else:

            # Fill missing calendar dates with zero demand.
            # This assumes missing dates represent no recorded sales.
            all_dates = pd.date_range(
                start=product_data.index.min(),
                end=product_data.index.max(),
                freq="D"
            )

            daily_demand = product_data.reindex(
                all_dates,
                fill_value=0
            ).astype(float)

            daily_demand.index.name = "Date"

            # Forecast only after the last observed date
            last_date = daily_demand.index.max()

            future_dates = pd.date_range(
                start=last_date + pd.Timedelta(days=1),
                periods=forecast_days,
                freq="D"
            )

            # --------------------------------------------
            # Generate forecast
            # --------------------------------------------

            forecast_method = ""

            try:

                from statsmodels.tsa.holtwinters import (
                    ExponentialSmoothing
                )

                if len(daily_demand) >= 28:

                    model = ExponentialSmoothing(
                        daily_demand,
                        trend="add",
                        seasonal="add",
                        seasonal_periods=7,
                        initialization_method="estimated"
                    )

                    fitted_model = model.fit(
                        optimized=True
                    )

                    forecast_values = fitted_model.forecast(
                        forecast_days
                    ).values

                    forecast_method = (
                        "Holt-Winters Exponential Smoothing"
                    )

                else:

                    # Not enough history for a weekly seasonal model.
                    # Use the recent average as a simple baseline.
                    recent_average = daily_demand.tail(
                        min(7, len(daily_demand))
                    ).mean()

                    forecast_values = [
                        recent_average
                    ] * forecast_days

                    forecast_method = (
                        "Recent Average Baseline "
                        "(limited history)"
                    )

            except Exception as error:

                st.warning(
                    "The advanced forecasting model could not run. "
                    "Using the recent-average baseline instead."
                )

                recent_average = daily_demand.tail(
                    min(7, len(daily_demand))
                ).mean()

                forecast_values = [
                    recent_average
                ] * forecast_days

                forecast_method = "Recent Average Baseline"

            # Demand cannot be negative
            forecast_values = [
                max(0, float(value))
                for value in forecast_values
            ]

            forecast_df = pd.DataFrame({
                "Date": future_dates,
                "Forecast_Units": forecast_values
            })

            forecast_df["Forecast_Units"] = (
                forecast_df["Forecast_Units"].round(2)
            )

            # --------------------------------------------
            # KPI cards
            # --------------------------------------------

            total_historical_units = daily_demand.sum()

            recent_7_day_average = daily_demand.tail(
                min(7, len(daily_demand))
            ).mean()

            forecast_total = forecast_df[
                "Forecast_Units"
            ].sum()

            metric1, metric2, metric3 = st.columns(3)

            metric1.metric(
                "Historical Units Sold",
                f"{total_historical_units:,.0f}"
            )

            metric2.metric(
                "Recent Daily Average",
                f"{recent_7_day_average:,.2f}"
            )

            metric3.metric(
                f"Forecast Units ({forecast_days} days)",
                f"{forecast_total:,.0f}"
            )

            st.caption(
                f"Forecast method: {forecast_method}"
            )

            st.info(
                "These are model-generated estimates, not guaranteed "
                "sales. Forecast quality depends on the product's "
                "historical demand patterns."
            )

            st.divider()

            # --------------------------------------------
            # Historical demand chart
            # --------------------------------------------

            st.subheader("Historical Daily Demand")

            history_df = daily_demand.reset_index()

            history_df.columns = [
                "Date",
                "Units_Sold"
            ]

            fig_history = px.line(
                history_df,
                x="Date",
                y="Units_Sold",
                title=f"Historical Demand: {selected_product}"
            )

            fig_history.update_layout(
                xaxis_title="Date",
                yaxis_title="Units Sold"
            )

            st.plotly_chart(
                fig_history,
                width="stretch"
            )

            # --------------------------------------------
            # Historical vs forecast chart
            # --------------------------------------------

            st.subheader("Historical vs Forecast Demand")

            recent_history = history_df.tail(90).copy()

            recent_history["Type"] = "Historical"

            forecast_chart_df = forecast_df.rename(
                columns={
                    "Forecast_Units": "Units_Sold"
                }
            ).copy()

            forecast_chart_df["Type"] = "Forecast"

            chart_df = pd.concat(
                [
                    recent_history,
                    forecast_chart_df
                ],
                ignore_index=True
            )

            fig_forecast = px.line(
                chart_df,
                x="Date",
                y="Units_Sold",
                color="Type",
                title=(
                    f"{selected_product}: "
                    f"Historical and Future Demand"
                ),
                markers=True
            )

            fig_forecast.update_layout(
                xaxis_title="Date",
                yaxis_title="Units Sold"
            )

            st.plotly_chart(
                fig_forecast,
                width="stretch"
            )

            # --------------------------------------------
            # Forecast table
            # --------------------------------------------

            st.subheader("Forecast Details")

            st.dataframe(
                forecast_df,
                width="stretch",
                hide_index=True
            )

            # --------------------------------------------
            # Download forecast
            # --------------------------------------------

            csv_data = forecast_df.to_csv(
                index=False
            ).encode("utf-8")

            st.download_button(
                label="Download Forecast CSV",
                data=csv_data,
                file_name=(
                    f"retailpulse_forecast_{selected_product}.csv"
                ),
                mime="text/csv"
            )

# ---------------- MODEL EVALUATION ----------------

elif page == "Model Evaluation":

    st.title("📊 Forecast Model Evaluation")

    st.write(
        "Compare forecasting models using historical evaluation "
        "results across 20 products."
    )

    # Load saved model evaluation summary
    try:
        model_summary = pd.read_csv(
            "outputs/retailpulse_product_model_summary.csv"
        )

    except FileNotFoundError:

        st.error(
            "Model summary file not found. "
            "Check the outputs folder."
        )

        st.stop()

    # Validate required columns
    required_columns = [
        "Model",
        "Products_Evaluated",
        "Average_MAE",
        "Median_MAE",
        "Average_RMSE",
        "Median_RMSE"
    ]

    missing_columns = [
        col for col in required_columns
        if col not in model_summary.columns
    ]

    if missing_columns:

        st.error(
            f"Missing columns in model summary: {missing_columns}"
        )

        st.stop()

    # Ensure metric columns are numeric
    metric_columns = [
        "Products_Evaluated",
        "Average_MAE",
        "Median_MAE",
        "Average_RMSE",
        "Median_RMSE"
    ]

    for col in metric_columns:
        model_summary[col] = pd.to_numeric(
            model_summary[col],
            errors="coerce"
        )

    model_summary = model_summary.dropna(
        subset=[
            "Average_MAE",
            "Average_RMSE"
        ]
    )

    if model_summary.empty:

        st.warning("No valid model evaluation results are available.")

        st.stop()

    # Lower MAE and RMSE are better
    best_mae_model = model_summary.loc[
        model_summary["Average_MAE"].idxmin()
    ]

    best_rmse_model = model_summary.loc[
        model_summary["Average_RMSE"].idxmin()
    ]

    # ---------------- KPI CARDS ----------------

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Models Evaluated",
        f"{len(model_summary)}"
    )

    col2.metric(
        "Products Evaluated",
        f"{int(model_summary['Products_Evaluated'].max())}"
    )

    col3.metric(
        "Best Average MAE",
        f"{best_mae_model['Average_MAE']:.2f}",
        help=f"Best model: {best_mae_model['Model']}"
    )

    st.divider()

    # ---------------- BEST MODELS ----------------

    st.subheader("Best Performing Models")

    col1, col2 = st.columns(2)

    with col1:

        st.success(
            f"**Lowest Average MAE**\n\n"
            f"{best_mae_model['Model']}\n\n"
            f"MAE: {best_mae_model['Average_MAE']:.2f}"
        )

    with col2:

        st.success(
            f"**Lowest Average RMSE**\n\n"
            f"{best_rmse_model['Model']}\n\n"
            f"RMSE: {best_rmse_model['Average_RMSE']:.2f}"
        )

    st.caption(
        "Lower MAE and RMSE indicate better predictions. "
        "These results summarize the saved evaluation experiment."
    )

    st.divider()

    # ---------------- MAE COMPARISON ----------------

    st.subheader("Average MAE Comparison")

    mae_sorted = model_summary.sort_values(
        "Average_MAE",
        ascending=True
    )

    fig_mae = px.bar(
        mae_sorted,
        x="Model",
        y="Average_MAE",
        title="Average Mean Absolute Error by Model",
        text_auto=".2f"
    )

    fig_mae.update_layout(
        xaxis_title="Forecasting Model",
        yaxis_title="Average MAE",
        xaxis_tickangle=-25
    )

    st.plotly_chart(
        fig_mae,
        width="stretch"
    )

    # ---------------- RMSE COMPARISON ----------------

    st.subheader("Average RMSE Comparison")

    rmse_sorted = model_summary.sort_values(
        "Average_RMSE",
        ascending=True
    )

    fig_rmse = px.bar(
        rmse_sorted,
        x="Model",
        y="Average_RMSE",
        title="Average Root Mean Squared Error by Model",
        text_auto=".2f"
    )

    fig_rmse.update_layout(
        xaxis_title="Forecasting Model",
        yaxis_title="Average RMSE",
        xaxis_tickangle=-25
    )

    st.plotly_chart(
        fig_rmse,
        width="stretch"
    )

    st.divider()

    # ---------------- FULL RESULTS TABLE ----------------

    st.subheader("Complete Evaluation Results")

    display_df = model_summary.sort_values(
        "Average_MAE",
        ascending=True
    ).copy()

    st.dataframe(
        display_df.round(2),
        width="stretch",
        hide_index=True
    )

    # ---------------- DOWNLOAD RESULTS ----------------

    csv_data = model_summary.to_csv(
        index=False
    ).encode("utf-8")

    st.download_button(
        label="Download Model Evaluation Results",
        data=csv_data,
        file_name="retailpulse_model_evaluation_summary.csv",
        mime="text/csv"
    )

    # ---------------- INTERPRETATION ----------------

    st.subheader("Business Interpretation")

    st.markdown(
        f"""
        - **Best model by average MAE:** {best_mae_model['Model']}.
        - **Best model by average RMSE:** {best_rmse_model['Model']}.
        - **Evaluation coverage:** {int(model_summary['Products_Evaluated'].max())} products.
        - **Recommendation:** Use these results to select a candidate
          forecasting model, then validate it on future, unseen data
          before using its predictions for inventory decisions.
        """
    )
# ---------------- BUSINESS RECOMMENDATIONS ----------------

elif page == "Business Recommendations":

    st.title("💡 Business Recommendations")

    st.write(
        "Turn RetailPulse customer and product analytics into "
        "actionable business recommendations."
    )

    # ----------------------------------------
    # CUSTOMER RECOMMENDATIONS
    # ----------------------------------------

    st.header("1. Customer Retention")

    if customer_df.empty:

        st.warning("Customer segmentation data is unavailable.")

    else:

        required_customer_columns = [
            "Recency",
            "Frequency",
            "Monetary"
        ]

        missing_columns = [
            col for col in required_customer_columns
            if col not in customer_df.columns
        ]

        if missing_columns:

            st.warning(
                "Customer data is missing required columns: "
                f"{missing_columns}"
            )

        else:

            customer_df = customer_df.copy()

            segment_column = next(
                (
                    col for col in customer_df.columns
                    if col.lower() in [
                        "segment",
                        "customer_segment",
                        "segment_name"
                    ]
                ),
                None
            )

            if segment_column:

                at_risk = customer_df[
                    customer_df[segment_column]
                    .astype(str)
                    .str.contains(
                        "inactive|at.risk",
                        case=False,
                        regex=True
                    )
                ]

                col1, col2 = st.columns(2)

                col1.metric(
                    "Customers in At-Risk Segment",
                    f"{len(at_risk):,}"
                )

                col2.metric(
                    "At-Risk Segment Historical Monetary Value",
                    f"£{at_risk['Monetary'].sum():,.2f}"
                )

                st.markdown(
                    """
                    **Recommended actions**

                    - Run a re-engagement email campaign.
                    - Offer relevant incentives to selected customers.
                    - Analyse previous purchases before making offers.
                    - Measure repeat purchases after the campaign.
                    """
                )

                st.caption(
                    "Historical monetary value is not the same as "
                    "future revenue or recoverable revenue."
                )

            else:

                st.info(
                    "The customer segment column was not detected. "
                    "Check the column names in your customer CSV."
                )

    st.divider()

    # ----------------------------------------
    # PRODUCT RECOMMENDATIONS
    # ----------------------------------------

    st.header("2. Product Performance")

    if product_summary_df.empty:

        st.warning("Product summary data is unavailable.")

    else:

        product_df = product_summary_df.copy()

        # Standardise numeric fields when present
        for col in [
            "Revenue",
            "Total_Revenue",
            "Units_Sold",
            "Total_Units",
            "Transactions"
        ]:

            if col in product_df.columns:

                product_df[col] = pd.to_numeric(
                    product_df[col],
                    errors="coerce"
                ).fillna(0)

        revenue_column = next(
            (
                col for col in product_df.columns
                if col.lower() in [
                    "revenue",
                    "total_revenue"
                ]
            ),
            None
        )

        units_column = next(
            (
                col for col in product_df.columns
                if col.lower() in [
                    "units_sold",
                    "total_units",
                    "quantity"
                ]
            ),
            None
        )

        if revenue_column:

            top_revenue_products = (
                product_df
                .sort_values(
                    revenue_column,
                    ascending=False
                )
                .head(10)
            )

            st.subheader("Top Products by Historical Revenue")

            st.dataframe(
                top_revenue_products,
                width="stretch",
                hide_index=True
            )

            st.markdown(
                """
                **Recommended actions**

                - Review stock availability for high-revenue products.
                - Monitor product-level demand before replenishment.
                - Investigate whether revenue depends heavily on a few
                  products.
                """
            )

        else:

            st.info(
                "A revenue column was not detected in the product summary."
            )

        if units_column:

            high_demand_products = (
                product_df
                .sort_values(
                    units_column,
                    ascending=False
                )
                .head(10)
            )

            st.subheader("Top Products by Historical Units Sold")

            st.dataframe(
                high_demand_products,
                width="stretch",
                hide_index=True
            )

            st.caption(
                "High historical unit sales can help prioritise "
                "inventory monitoring, but do not guarantee future demand."
            )

    st.divider()

    # ----------------------------------------
    # FORECASTING RECOMMENDATIONS
    # ----------------------------------------

    st.header("3. Forecasting Strategy")

    st.markdown(
        """
        Based on the saved experiment across 20 products:

        - **Holt-Winters** achieved the lowest average MAE and RMSE.
        - The **7-Day Moving Average** was a competitive alternative.
        - Model performance should be checked for each product before
          selecting a production forecasting method.

        **Recommended actions**

        1. Prioritise products with sufficient sales history.
        2. Compare forecast errors before using predictions for inventory.
        3. Monitor forecast performance as new sales data arrives.
        4. Avoid automatically increasing stock based only on historical
           revenue or a single forecast.
        """
    )

    st.success(
        "Recommendation: Combine customer segmentation, product "
        "performance, and validated demand forecasts when planning "
        "retention campaigns and inventory."
    )

# ---------------- DATA QUALITY ----------------

elif page == "Data Quality":

    st.title("🔍 Data Quality & Validation")

    st.write(
        "Monitor the quality, completeness, and consistency "
        "of the RetailPulse sales dataset."
    )

    # Work with a copy to avoid modifying the original dataset
    quality_df = sales_df.copy()

    # ----------------------------------------
    # DATA PREPARATION
    # ----------------------------------------

    quality_df["InvoiceDate"] = pd.to_datetime(
        quality_df["InvoiceDate"],
        errors="coerce"
    )

    quality_df["Quantity"] = pd.to_numeric(
        quality_df["Quantity"],
        errors="coerce"
    )

    quality_df["UnitPrice"] = pd.to_numeric(
        quality_df["UnitPrice"],
        errors="coerce"
    )

    quality_df["Revenue"] = pd.to_numeric(
        quality_df["Revenue"],
        errors="coerce"
    )

    total_rows = len(quality_df)
    total_columns = len(quality_df.columns)

    missing_cells = int(
        quality_df.isna().sum().sum()
    )

    duplicate_rows = int(
        quality_df.duplicated().sum()
    )

    # ----------------------------------------
    # OVERVIEW METRICS
    # ----------------------------------------

    st.subheader("Dataset Overview")

    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "Total Records",
        f"{total_rows:,}"
    )

    col2.metric(
        "Total Columns",
        f"{total_columns:,}"
    )

    col3.metric(
        "Missing Cells",
        f"{missing_cells:,}"
    )

    col4.metric(
        "Duplicate Rows",
        f"{duplicate_rows:,}"
    )

    st.divider()

    # ----------------------------------------
    # MISSING VALUES
    # ----------------------------------------

    st.subheader("1. Missing Values Analysis")

    missing_summary = pd.DataFrame({
        "Column": quality_df.columns,
        "Missing_Count": quality_df.isna().sum().values
    })

    missing_summary["Missing_Percentage"] = (
        missing_summary["Missing_Count"]
        / max(total_rows, 1)
        * 100
    )

    missing_summary = missing_summary.sort_values(
        "Missing_Count",
        ascending=False
    )

    missing_summary = missing_summary[
        missing_summary["Missing_Count"] > 0
    ]

    if missing_summary.empty:

        st.success(
            "No missing values detected in the loaded dataset."
        )

    else:

        fig_missing = px.bar(
            missing_summary,
            x="Column",
            y="Missing_Count",
            title="Missing Values by Column",
            text_auto=True
        )

        fig_missing.update_layout(
            xaxis_title="Column",
            yaxis_title="Missing Values",
            xaxis_tickangle=-30
        )

        st.plotly_chart(
            fig_missing,
            width="stretch"
        )

        st.dataframe(
            missing_summary.round(2),
            width="stretch",
            hide_index=True
        )

    st.divider()

    # ----------------------------------------
    # DATA VALIDATION CHECKS
    # ----------------------------------------

    st.subheader("2. Data Validation Checks")

    validation_results = []

    def add_check(name, failed_count, explanation):

        validation_results.append({
            "Validation Check": name,
            "Failed Records": int(failed_count),
            "Status": (
                "PASS"
                if failed_count == 0
                else "CHECK"
            ),
            "Explanation": explanation
        })

    add_check(
        "Missing InvoiceDate",
        quality_df["InvoiceDate"].isna().sum(),
        "Invoice dates should be valid."
    )

    add_check(
        "Missing InvoiceNo",
        quality_df["InvoiceNo"].isna().sum(),
        "Invoice numbers identify transactions."
    )

    add_check(
        "Missing StockCode",
        quality_df["StockCode"].isna().sum(),
        "Product codes should be present."
    )

    add_check(
        "Missing Country",
        quality_df["Country"].isna().sum(),
        "Country is needed for geographic analysis."
    )

    add_check(
        "Missing Quantity",
        quality_df["Quantity"].isna().sum(),
        "Quantity should be numeric and available."
    )

    add_check(
        "Missing UnitPrice",
        quality_df["UnitPrice"].isna().sum(),
        "Unit price should be numeric and available."
    )

    add_check(
        "Missing Revenue",
        quality_df["Revenue"].isna().sum(),
        "Revenue should be available for sales analysis."
    )

    add_check(
        "Non-positive Quantity",
        (quality_df["Quantity"] <= 0).sum(),
        "Check transactions with zero or negative quantities."
    )

    add_check(
        "Non-positive UnitPrice",
        (quality_df["UnitPrice"] <= 0).sum(),
        "Check transactions with zero or negative prices."
    )

    add_check(
        "Negative Revenue",
        (quality_df["Revenue"] < 0).sum(),
        "Investigate negative revenue records."
    )

    # Check whether revenue equals quantity multiplied by unit price.
    expected_revenue = (
        quality_df["Quantity"]
        * quality_df["UnitPrice"]
    )

    revenue_mismatch = (
        quality_df["Revenue"].notna()
        & expected_revenue.notna()
        & ~np.isclose(
            quality_df["Revenue"],
            expected_revenue,
            rtol=1e-5,
            atol=0.01
        )
    )

    add_check(
        "Revenue Calculation Mismatch",
        revenue_mismatch.sum(),
        "Revenue should equal Quantity multiplied by UnitPrice."
    )

    validation_df = pd.DataFrame(
        validation_results
    )

    passed_checks = int(
        (validation_df["Status"] == "PASS").sum()
    )

    total_checks = len(validation_df)

    col1, col2 = st.columns(2)

    col1.metric(
        "Validation Checks Passed",
        f"{passed_checks}/{total_checks}"
    )

    col2.metric(
        "Checks Requiring Attention",
        f"{total_checks - passed_checks}"
    )

    st.dataframe(
        validation_df,
        width="stretch",
        hide_index=True
    )

    st.caption(
        "CHECK means records need investigation. It does not "
        "automatically mean the dataset is unusable."
    )

    st.divider()

    # ----------------------------------------
    # DUPLICATE RECORDS
    # ----------------------------------------

    st.subheader("3. Duplicate Record Analysis")

    st.write(
        "Duplicate detection identifies records with identical "
        "values across all loaded columns."
    )

    if duplicate_rows == 0:

        st.success(
            "No exact duplicate rows were detected."
        )

    else:

        st.warning(
            f"{duplicate_rows:,} exact duplicate rows were detected."
        )

        duplicate_preview = quality_df[
            quality_df.duplicated(keep=False)
        ].head(100)

        st.dataframe(
            duplicate_preview,
            width="stretch",
            hide_index=True
        )

    st.divider()

    # ----------------------------------------
    # DATE COVERAGE
    # ----------------------------------------

    st.subheader("4. Transaction Date Coverage")

    valid_dates = quality_df["InvoiceDate"].dropna()

    if valid_dates.empty:

        st.warning(
            "No valid transaction dates are available."
        )

    else:

        date_col1, date_col2, date_col3 = st.columns(3)

        date_col1.metric(
            "First Transaction",
            valid_dates.min().strftime("%d %b %Y")
        )

        date_col2.metric(
            "Last Transaction",
            valid_dates.max().strftime("%d %b %Y")
        )

        date_col3.metric(
            "Days Covered",
            f"{(valid_dates.max().normalize() - valid_dates.min().normalize()).days + 1:,}"
        )

        st.info(
            "The dataset ends on 9 December 2011. December is "
            "therefore a partial month, so monthly comparisons "
            "should account for incomplete coverage."
        )

    st.divider()

    # ----------------------------------------
    # DOWNLOAD QUALITY REPORT
    # ----------------------------------------

    st.subheader("5. Download Quality Report")

    quality_report = validation_df.to_csv(
        index=False
    ).encode("utf-8")

    st.download_button(
        label="Download Validation Report",
        data=quality_report,
        file_name="retailpulse_data_quality_report.csv",
        mime="text/csv"
    )