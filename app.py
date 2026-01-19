import streamlit as st
import pandas as pd
import numpy as np
from scipy.stats import beta
import plotly.express as px

# --------------------
# CONFIG
# --------------------
st.set_page_config(
    page_title="Thompson Sampling Call Optimizer",
    layout="wide"
)

# Your live Google Sheets CSV URL
CSV_URL = "https://docs.google.com/spreadsheets/d/e/2PACX-1vTenqgW-kUxb15j20c86F-g343y1ulD3-dkWV05xVw1BnDGonYwcriK8iSlLD71UPrCNejVuJFI01jT/pub?gid=868983311&single=true&output=csv"

# --------------------
# DATA LOADING
# --------------------
@st.cache_data
def load_data(csv_url: str) -> pd.DataFrame:
    df = pd.read_csv(csv_url)
    # Normalize column names
    df.columns = [c.strip().lower() for c in df.columns]

    # Map expected names
    if 'customer' not in df.columns and 'acctrefno' in df.columns:
        df.rename(columns={'acctrefno': 'customer'}, inplace=True)
    if 'hour' not in df.columns and 'callhr' in df.columns:
        df.rename(columns={'callhr': 'hour'}, inplace=True)

    # Keep only required columns if they exist
    keep_cols = [c for c in ['customer', 'hour', 'success', 'failure'] if c in df.columns]
    df = df[keep_cols].copy()

    # Ensure types
    df['customer'] = df['customer'].astype(str)
    df['hour'] = pd.to_numeric(df['hour'], errors='coerce')
    df['success'] = pd.to_numeric(df['success'], errors='coerce').fillna(0).astype(int)
    df['failure'] = pd.to_numeric(df['failure'], errors='coerce').fillna(0).astype(int)

    df = df.dropna(subset=['hour'])
    df['hour'] = df['hour'].astype(int)

    return df

df = load_data(CSV_URL)

# --------------------
# TITLE / HEADER
# --------------------
st.markdown(
    "<h1 style='margin-bottom:0.2rem;'>Thompson Sampling Call Optimizer</h1>",
    unsafe_allow_html=True
)
st.markdown(
    "<p style='color:#4b5563;margin-top:0;'>Real-time AI-Powered Call Timing Engine</p>",
    unsafe_allow_html=True
)
st.caption(
    f"Analyzing {df['customer'].nunique()} customers • {len(df)} call patterns"
)

# --------------------
# CUSTOMER SELECTION CARD
# --------------------
st.markdown("---")

customers = sorted(df['customer'].unique())
default_customer = customers[0] if customers else None

col_sel, _ = st.columns([1, 3])
with col_sel:
    selected_customer = st.selectbox(
        "Select Customer ID",
        customers,
        index=customers.index(default_customer) if default_customer in customers else 0,
        format_func=lambda x: f"Customer {x}"
    )

cust_df = df[df['customer'] == selected_customer].copy()

# --------------------
# THOMPSON SAMPLING LOGIC
# --------------------
def thompson_sampling_for_customer(cdf: pd.DataFrame) -> pd.DataFrame:
    if cdf.empty:
        return cdf.assign(
            alpha=[],
            beta=[],
            probability=[],
            total_attempts=[],
            success_rate=[]
        )

    cdf = cdf.copy()
    cdf['alpha'] = cdf['success'] + 1
    cdf['beta'] = cdf['failure'] + 1

    # Sample from Beta(alpha, beta)
    cdf['probability'] = beta.rvs(cdf['alpha'], cdf['beta'])

    cdf['total_attempts'] = cdf['success'] + cdf['failure']
    cdf['success_rate'] = cdf['total_attempts'].replace(0, np.nan)
    cdf['success_rate'] = cdf['success'] / cdf['success_rate']
    cdf['success_rate'] = cdf['success_rate'].fillna(0.0)

    cdf = cdf.sort_values('probability', ascending=False)
    return cdf

rec_df = thompson_sampling_for_customer(cust_df)

total_calls = int(rec_df['total_attempts'].sum()) if not rec_df.empty else 0
total_success = int(rec_df['success'].sum()) if not rec_df.empty else 0
overall_rate = (total_success / total_calls * 100) if total_calls > 0 else 0.0

best_row = rec_df.iloc[0] if not rec_df.empty else None
best_hour_display = f"{int(best_row['hour'])}:00" if best_row is not None else "N/A"
best_conf_display = f"{best_row['probability']*100:.1f}%" if best_row is not None else "0.0%"

# --------------------
# KPI CARDS (TOP CARDS)
# --------------------
st.markdown("---")

k1, k2, k3, k4 = st.columns(4)

with k1:
    st.markdown(
        """
        <div style="
            background: linear-gradient(to bottom right, #22c55e, #16a34a);
            border-radius: 0.75rem;
            padding: 1.25rem;
            color: white;
            box-shadow: 0 10px 15px -3px rgba(22,163,74,0.3);
        ">
            <p style="font-size:0.8rem; color:#bbf7d0; margin:0;">Best Time to Call</p>
            <p style="font-size:2rem; font-weight:700; margin:0.25rem 0 0;">{hour}</p>
            <p style="font-size:0.7rem; color:#bbf7d0; margin:0.25rem 0 0;">Hour of day</p>
        </div>
        """.format(hour=best_hour_display),
        unsafe_allow_html=True
    )

with k2:
    st.markdown(
        """
        <div style="
            background: linear-gradient(to bottom right, #3b82f6, #4f46e5);
            border-radius: 0.75rem;
            padding: 1.25rem;
            color: white;
            box-shadow: 0 10px 15px -3px rgba(59,130,246,0.3);
        ">
            <p style="font-size:0.8rem; color:#bfdbfe; margin:0;">Confidence</p>
            <p style="font-size:2rem; font-weight:700; margin:0.25rem 0 0;">{conf}</p>
            <p style="font-size:0.7rem; color:#bfdbfe; margin:0.25rem 0 0;">Thompson sampling</p>
        </div>
        """.format(conf=best_conf_display),
        unsafe_allow_html=True
    )

with k3:
    st.markdown(
        """
        <div style="
            background: linear-gradient(to bottom right, #a855f7, #7c3aed);
            border-radius: 0.75rem;
            padding: 1.25rem;
            color: white;
            box-shadow: 0 10px 15px -3px rgba(124,58,237,0.3);
        ">
            <p style="font-size:0.8rem; color:#e9d5ff; margin:0;">Total Attempts</p>
            <p style="font-size:2rem; font-weight:700; margin:0.25rem 0 0;">{calls}</p>
            <p style="font-size:0.7rem; color:#e9d5ff; margin:0.25rem 0 0;">Historical calls</p>
        </div>
        """.format(calls=total_calls),
        unsafe_allow_html=True
    )

with k4:
    st.markdown(
        """
        <div style="
            background: linear-gradient(to bottom right, #f97316, #ea580c);
            border-radius: 0.75rem;
            padding: 1.25rem;
            color: white;
            box-shadow: 0 10px 15px -3px rgba(234,88,12,0.3);
        ">
            <p style="font-size:0.8rem; color:#fed7aa; margin:0;">Success Rate</p>
            <p style="font-size:2rem; font-weight:700; margin:0.25rem 0 0;">{rate}</p>
            <p style="font-size:0.7rem; color:#fed7aa; margin:0.25rem 0 0;">Historical pickup</p>
        </div>
        """.format(rate=f"{overall_rate:.1f}%"),
        unsafe_allow_html=True
    )

if total_calls < 5:
    st.warning(
        f"Low Data Warning: Only {total_calls} call attempts recorded for this customer. "
        f"Recommendations will improve with more data."
    )

# --------------------
# CHARTS ROW
# --------------------
st.markdown("---")
col1, col2 = st.columns(2)

# Probability distribution chart
with col1:
    st.subheader("Thompson Sampling Probabilities")
    if not rec_df.empty:
        prob_df = rec_df[['hour', 'probability']].copy()
        prob_df.set_index('hour', inplace=True)
        st.bar_chart(prob_df, use_container_width=True)
    else:
        st.info("No data available for this customer.")

# Learning data (α, β) chart with legend on top
with col2:
    st.subheader("Learning Data (α, β Parameters)")
    if not rec_df.empty:
        sf_df = rec_df[['hour', 'success', 'failure']].copy()

        fig = px.bar(
            sf_df,
            x='hour',
            y=['success', 'failure'],
            barmode='group',
            labels={'value': 'Count', 'hour': 'Hour', 'variable': 'Metric'},
            title="Learning Data (α, β Parameters)"
        )

        fig.update_layout(
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.12,
                xanchor="center",
                x=0.5
            )
        )

        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No data available for this customer.")

# --------------------
# RANKED RECOMMENDATIONS TABLE
# --------------------
st.subheader(f"Ranked Recommendations for Customer {selected_customer}")

if not rec_df.empty:
    display_df = rec_df[['hour', 'success', 'failure', 'total_attempts', 'success_rate', 'probability']].copy()
    display_df['success_rate'] = (display_df['success_rate'] * 100).round(1)
    display_df['probability'] = (display_df['probability'] * 100).round(1)

    display_df.rename(columns={
        'hour': 'Hour',
        'success': 'Success (α)',
        'failure': 'Failure (β)',
        'total_attempts': 'Total Attempts',
        'success_rate': 'Historical Rate (%)',
        'probability': 'Sampled Probability (%)'
    }, inplace=True)

    st.dataframe(display_df, use_container_width=True)
else:
    st.info("No recommendations available (no data).")

# --------------------
# ALL CUSTOMERS SUMMARY
# --------------------
st.subheader("All Customers Summary")

@st.cache_data
def summarize_all_customers(df_all: pd.DataFrame) -> pd.DataFrame:
    out_rows = []

    for cust in df_all['customer'].unique():
        cdf = df_all[df_all['customer'] == cust].copy()
        r = thompson_sampling_for_customer(cdf)
        if r.empty:
            continue
        best = r.iloc[0]
        total = int(r['total_attempts'].sum())
        success = int(r['success'].sum())
        rate = (success / total * 100) if total > 0 else 0.0

        out_rows.append({
            'Customer': cust,
            'Best Hour': int(best['hour']),
            'Confidence (%)': best['probability'] * 100,
            'Total Calls': total,
            'Success Rate (%)': rate
        })

    summary = pd.DataFrame(out_rows)
    if not summary.empty:
        summary['Data Quality'] = pd.cut(
            summary['Total Calls'],
            bins=[-1, 4, 9, 1e9],
            labels=['Low', 'Fair', 'Good']
        )
    return summary

summary_df = summarize_all_customers(df)

if not summary_df.empty:
    st.dataframe(summary_df, use_container_width=True)
else:
    st.info("No customer summary available (no data).")

# --------------------
# METHODOLOGY
# --------------------
st.markdown("### Thompson Sampling Methodology")

st.write(
    "- **Data Collection**: For each customer-hour, the system tracks successes and failures "
    "from historical call outcomes (success → α, failure → β)."
)
st.write(
    "- **Bayesian Sampling**: For each hour, it samples a probability from Beta(α+1, β+1), "
    "which naturally balances exploration vs exploitation."
)
st.write(
    "- **Dynamic Decision**: The recommended hour is the one with the highest sampled probability "
    "for the selected customer."
)
