import streamlit as st
import pandas as pd
import numpy as np
from scipy.stats import beta

st.set_page_config(page_title="Thompson Sampling Call Optimizer", layout="wide")

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
    return df

# Your live Google Sheets CSV URL
CSV_URL = "https://docs.google.com/spreadsheets/d/e/2PACX-1vTenqgW-kUxb15j20c86F-g343y1ulD3-dkWV05xVw1BnDGonYwcriK8iSlLD71UPrCNejVuJFI01jT/pub?gid=868983311&single=true&output=csv"
df = load_data(CSV_URL)

st.title("Thompson Sampling Call Optimizer")
st.caption(f"Analyzing {df['customer'].nunique()} customers • {len(df)} call patterns")

customers = sorted(df['customer'].unique())
selected_customer = st.selectbox("Select Customer ID", customers, index=0)

cust_df = df[df['customer'] == selected_customer].copy()

def thompson_sampling_for_customer(cdf: pd.DataFrame) -> pd.DataFrame:
    # alpha = success + 1, beta = failure + 1
    cdf = cdf.copy()
    cdf['alpha'] = cdf['success'] + 1
    cdf['beta'] = cdf['failure'] + 1
    # sample from Beta distribution
    cdf['probability'] = beta.rvs(cdf['alpha'], cdf['beta'])
    cdf['total_attempts'] = cdf['success'] + cdf['failure']
    cdf['success_rate'] = cdf['success'] / cdf['total_attempts'].replace(0, np.nan)
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
# KPI CARDS (only customization)
# --------------------
k1, k2, k3, k4 = st.columns(4)

with k1:
    st.markdown(
        f"""
        <div style="
            background-color:#f0fdf4;
            border-radius:0.75rem;
            padding:1.1rem;
            border:1px solid #bbf7d0;
        ">
            <p style="font-size:0.8rem;color:#15803d;margin:0;">Best Time to Call</p>
            <p style="font-size:1.8rem;font-weight:700;margin:0.25rem 0 0;color:#14532d;">{best_hour_display}</p>
            <p style="font-size:0.75rem;color:#15803d;margin:0.25rem 0 0;">Hour of day</p>
        </div>
        """,
        unsafe_allow_html=True
    )

with k2:
    st.markdown(
        f"""
        <div style="
            background-color:#eff6ff;
            border-radius:0.75rem;
            padding:1.1rem;
            border:1px solid #bfdbfe;
        ">
            <p style="font-size:0.8rem;color:#1d4ed8;margin:0;">Confidence</p>
            <p style="font-size:1.8rem;font-weight:700;margin:0.25rem 0 0;color:#1d4ed8;">{best_conf_display}</p>
            <p style="font-size:0.75rem;color:#1d4ed8;margin:0.25rem 0 0;">Thompson sampling</p>
        </div>
        """,
        unsafe_allow_html=True
    )

with k3:
    st.markdown(
        f"""
        <div style="
            background-color:#faf5ff;
            border-radius:0.75rem;
            padding:1.1rem;
            border:1px solid #e9d5ff;
        ">
            <p style="font-size:0.8rem;color:#7e22ce;margin:0;">Total Attempts</p>
            <p style="font-size:1.8rem;font-weight:700;margin:0.25rem 0 0;color:#6b21a8;">{total_calls}</p>
            <p style="font-size:0.75rem;color:#7e22ce;margin:0.25rem 0 0;">Historical calls</p>
        </div>
        """,
        unsafe_allow_html=True
    )

with k4:
    st.markdown(
        f"""
        <div style="
            background-color:#fff7ed;
            border-radius:0.75rem;
            padding:1.1rem;
            border:1px solid #fed7aa;
        ">
            <p style="font-size:0.8rem;color:#c2410c;margin:0;">Success Rate</p>
            <p style="font-size:1.8rem;font-weight:700;margin:0.25rem 0 0;color:#9a3412;">{overall_rate:.1f}%</p>
            <p style="font-size:0.75rem;color:#c2410c;margin:0.25rem 0 0;">Historical pickup</p>
        </div>
        """,
        unsafe_allow_html=True
    )

if total_calls < 5:
    st.warning(f"Low Data: Only {total_calls} call attempts recorded. Recommendations will improve with more data.")

# --------------------
# Charts (same as original)
# --------------------
col1, col2 = st.columns(2)

with col1:
    st.subheader("Thompson Sampling Probabilities")
    if not rec_df.empty:
        st.bar_chart(
            rec_df.set_index('hour')['probability'],
            use_container_width=True
        )
    else:
        st.info("No data available for this customer.")

with col2:
    st.subheader("Learning Data (α, β Parameters)")
    if not rec_df.empty:
        sf_df = rec_df[['hour', 'success', 'failure']].set_index('hour')
        st.bar_chart(sf_df, use_container_width=True)
    else:
        st.info("No data available for this customer.")

# --------------------
# Ranked recommendations (same as original)
# --------------------
st.subheader(f"Ranked Recommendations for Customer {selected_customer}")
display_cols = ['hour', 'success', 'failure', 'total_attempts', 'success_rate', 'probability']

if not rec_df.empty:
    st.dataframe(
        rec_df[display_cols].assign(
            success_rate=lambda x: (x['success_rate'] * 100).round(1),
            probability=lambda x: (x['probability'] * 100).round(1)
        ),
        use_container_width=True
    )
else:
    st.info("No recommendations available (no data).")

# --------------------
# All customers summary (same as original)
# --------------------
st.subheader("All Customers Summary")

def summarize_all_customers(df_all: pd.DataFrame) -> pd.DataFrame:
    out = []
    for cust in df_all['customer'].unique():
        cdf = df_all[df_all['customer'] == cust].copy()
        r = thompson_sampling_for_customer(cdf)
        if r.empty:
            continue
        best = r.iloc[0]
        total = int(r['total_attempts'].sum())
        success = int(r['success'].sum())
        rate = (success / total * 100) if total > 0 else 0.0
        out.append({
            'customer': cust,
            'best_hour': int(best['hour']),
            'confidence': best['probability'] * 100,
            'total_calls': total,
            'success_rate': rate
        })
    return pd.DataFrame(out)

summary_df = summarize_all_customers(df)

if not summary_df.empty:
    summary_df['data_quality'] = pd.cut(
        summary_df['total_calls'],
        bins=[-1, 4, 9, 1e9],
        labels=['Low', 'Fair', 'Good']
    )
    st.dataframe(summary_df, use_container_width=True)
else:
    st.info("No customer summary available (no data).")

st.markdown("**Methodology**")
st.write(
    "For each customer-hour, the system tracks successes & failures as α and β, "
    "samples from Beta(α+1, β+1) for Thompson Sampling, and recommends the hour "
    "with the highest sampled probability."
)
