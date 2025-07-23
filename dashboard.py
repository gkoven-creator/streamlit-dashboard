import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import re

def normalize_company(s):
    s = str(s).lower()
    s = s.replace('&', 'and')
    s = re.sub(r'[^a-z0-9]', '', s)  # remove all non-alphanumeric characters
    return s

def prettify_company(s):
    # Insert spaces before capital letters and title-case
    s = re.sub(r'([a-z])([A-Z])', r'\1 \2', s)
    s = re.sub(r'([a-z])([0-9])', r'\1 \2', s)
    s = re.sub(r'([0-9])([a-zA-Z])', r'\1 \2', s)
    return s.title().replace('And', 'And')

# Load data
def load_data():
    df = pd.read_csv('goisheet - Sheet3 (1).csv')
    # Clean up column names
    df.columns = [c.strip() for c in df.columns]
    # Parse payment amount with correct locale
    def parse_payment_amount(val):
        if pd.isna(val):
            return None
        s = str(val).replace(' ', '')
        if '€' in s:
            s = s.replace('€', '').replace('.', '').replace(',', '.')
        else:
            s = s.replace(',', '')
        try:
            return float(s)
        except Exception:
            return None
    df['Payment Amount (payment currency)'] = df['Payment Amount (payment currency)'].apply(parse_payment_amount)
    # Fill missing values and normalize case for grouping/display
    for col in ['Company', 'Normalized Name', 'Country', 'Bucket']:
        if col in df.columns:
            if col == 'Company':
                df[col] = df[col].fillna('Unknown').astype(str).str.strip()
                df['Company_Normalized'] = df[col].apply(normalize_company)
            else:
                df[col] = df[col].fillna('Unknown').astype(str).str.strip().str.title()
    # Add Period column for all uses
    df['Period'] = df['Payment Cycle Period (Calculated)'].apply(parse_period)
    return df

def parse_period(date_str):
    if pd.isna(date_str):
        return None
    s = str(date_str).strip()
    # Try YYYY-MM or YYYY/MM
    m = re.match(r'(\d{4})[-/](\d{2})', s)
    if m:
        return f"{m.group(1)}-{m.group(2)}"
    # Try YYYY-MM-DD or YYYY/MM/DD
    m = re.match(r'(\d{4})[-/](\d{2})[-/](\d{2})', s)
    if m:
        return f"{m.group(1)}-{m.group(2)}"
    # Try Month YYYY
    m = re.match(r'([A-Za-z]+)\s+(\d{4})', s)
    if m:
        try:
            dt = datetime.strptime(f"{m.group(2)}-{m.group(1)}", "%Y-%B")
            return dt.strftime("%Y-%m")
        except Exception:
            try:
                dt = datetime.strptime(f"{m.group(2)}-{m.group(1)}", "%Y-%b")
                return dt.strftime("%Y-%m")
            except Exception:
                return None
    # Try parsing as date
    for fmt in ("%Y-%m-%d", "%Y/%m/%d", "%Y-%m", "%Y/%m"):
        try:
            dt = datetime.strptime(s, fmt)
            return dt.strftime("%Y-%m")
        except Exception:
            continue
    return None

def filter_last_12_months(df):
    # Standardize period to YYYY-MM
    df['Period'] = df['Payment Cycle Period (Calculated)'].apply(parse_period)
    # Remove rows with invalid period
    df = df[df['Period'].notna()]
    # Filter to last 12 months
    last_month = datetime.today().replace(day=1)
    first_month = (last_month - pd.DateOffset(months=12)).strftime('%Y-%m')
    df = df[df['Period'] >= first_month]
    return df

def filter_data(df, company_norm, worker, country):
    if company_norm != 'All':
        df = df[df['Company_Normalized'] == company_norm]
    if worker != 'All':
        df = df[df['Normalized Name'] == worker]
    if country != 'All':
        df = df[df['Country'] == country]
    return df

# Number formatting for large values
def human_format(num):
    if pd.isna(num):
        return ''
    num = float(num)
    for unit in ['', 'K', 'M', 'B', 'T']:
        if abs(num) < 1000:
            return f"{num:.1f}{unit}" if unit else f"{num:.0f}"
        num /= 1000.0
    return f"{num:.1f}P"

def format_total_amount(x):
    if isinstance(x, (int, float)) and pd.notna(x):
        return f"{x:,.0f}"
    return x  # e.g., 'n/a'

st.title('Goi Revenue, Profit, and Costs Dashboard')

st.info('Showing only the last 12 months of data. All periods are grouped by month and year (YYYY-MM).')

df = load_data()

def filter_last_12_months(df):
    # Standardize period to YYYY-MM
    df['Period'] = df['Payment Cycle Period (Calculated)'].apply(parse_period)
    # Remove rows with invalid period
    df = df[df['Period'].notna()]
    # Filter to last 12 months
    last_month = datetime.today().replace(day=1)
    first_month = (last_month - pd.DateOffset(months=12)).strftime('%Y-%m')
    df = df[df['Period'] >= first_month]
    return df

df_last12 = filter_last_12_months(df.copy())

# After loading and normalizing data, ensure company column is fully normalized and unique for dropdown
# Sidebar filters
# Build a mapping from normalized to prettified for display
company_norm_to_pretty = {}
for raw, norm in zip(df['Company'], df['Company_Normalized']):
    # Special case for Stone & Bridges
    if norm == 'stoneandbridges':
        pretty = 'Stone & Bridges'
    else:
        pretty = ' '.join([w.capitalize() for w in re.findall(r'[a-zA-Z0-9]+', raw)])
    company_norm_to_pretty[norm] = pretty
company_options_norm = sorted(set(df['Company_Normalized']))
company_options_display = ['All'] + [company_norm_to_pretty[norm] for norm in company_options_norm]
company_display = st.sidebar.selectbox('Company', company_options_display)
# Map display back to normalized value
if company_display == 'All':
    company_norm = 'All'
else:
    # Find the normalized value for the selected display
    company_norm = [k for k, v in company_norm_to_pretty.items() if v == company_display][0]

# Worker dropdown depends on company selection
worker_col = 'Normalized Name'
if company_norm != 'All':
    worker_options = ['All'] + sorted(df[df['Company_Normalized'] == company_norm][worker_col].unique())
else:
    worker_options = ['All'] + sorted(df[worker_col].unique())
worker = st.sidebar.selectbox('Worker', worker_options)

country_options = ['All'] + sorted(df['Country'].unique())
country = st.sidebar.selectbox('Country', country_options)

# New: Option to hide IC rate in costs over time
dice_hide_ic = st.sidebar.checkbox('Hide IC Rate in Costs Over Time', value=False)

# New: Option to show summary table by month
summary_mode = st.radio('Summary Table Mode', ['All Time', 'By Month'], horizontal=True)

filtered_df = filter_data(df, company_norm, worker, country)
filtered_df_last12 = filter_data(df_last12, company_norm, worker, country)

def summary_table(df):
    summary = df.groupby('Bucket')['Payment Amount (payment currency)'].sum().reset_index()
    summary = summary.rename(columns={'Payment Amount (payment currency)': 'Total Amount'})
    # Exclude 'Prepaid Deel credits' from summary
    summary = summary[summary['Bucket'].str.lower() != 'prepaid deel credits']
    # Special handling for profit: if any profit row in filtered df has n/a/NaN/empty, set to 'n/a'
    profit_rows = df[df['Bucket'].str.lower() == 'profit']
    if not profit_rows.empty and profit_rows['Payment Amount (payment currency)'].isnull().any():
        summary.loc[summary['Bucket'].str.lower() == 'profit', 'Total Amount'] = 'n/a'
        profit_val = 'n/a'
    else:
        profit_val = summary.loc[summary['Bucket'].str.lower() == 'profit', 'Total Amount'].values[0] if 'profit' in summary['Bucket'].str.lower().values else 0
    # Revenue: sum all revenue buckets
    revenue_val = summary[summary['Bucket'].str.lower().str.contains('revenue')]['Total Amount'].replace('n/a', 0).astype(float).sum() if not summary[summary['Bucket'].str.lower().str.contains('revenue')].empty else 0
    # Margin calculation
    if profit_val == 'n/a' or revenue_val == 0:
        margin_val = 'n/a'
    else:
        margin_val = f"{(float(profit_val) / revenue_val * 100):.1f}%"
    # Add margin row
    summary = pd.concat([summary, pd.DataFrame([{'Bucket': 'Margin', 'Total Amount': margin_val}])], ignore_index=True)
    # Add hours row
    if 'hours' in df.columns:
        hours_numeric = pd.to_numeric(df['hours'], errors='coerce')
        if hours_numeric.isnull().any():
            hours_sum = 'n/a'
        else:
            hours_sum = hours_numeric.sum()
        summary = pd.concat([summary, pd.DataFrame([{'Bucket': 'Hours', 'Total Amount': hours_sum}])], ignore_index=True)
    return summary  # Keep as number for correct sorting except for n/a

def highlight_hours(s):
    return [
        'font-family: monospace;' if str(v).lower() == 'hours' else ''
        for v in s
    ]

def summary_table_by_month(df):
    # Pivot table: rows=Period, columns=Bucket, values=sum of Payment Amount
    pivot = df.pivot_table(index='Period', columns='Bucket', values='Payment Amount (payment currency)', aggfunc='sum', fill_value=0)
    # Exclude 'Prepaid Deel credits' column if present
    if 'Prepaid Deel Credits' in pivot.columns:
        pivot = pivot.drop(columns=['Prepaid Deel Credits'])
    # Special handling for profit: if any profit row for a month is n/a/NaN/empty, set that cell to 'n/a'
    for period in pivot.index:
        profit_rows = df[(df['Period'] == period) & (df['Bucket'].str.lower() == 'profit')]
        if not profit_rows.empty and profit_rows['Payment Amount (payment currency)'].isnull().any():
            pivot.loc[period, 'Profit'] = 'n/a'
    # Format numbers with commas, leave 'n/a' as is
    def fmt(x):
        if isinstance(x, (int, float)) and pd.notna(x):
            return f"{x:,.0f}"
        return x
    return pivot.applymap(fmt)

# Dropdown to select month (with 'All' option)
available_months = sorted(filtered_df['Period'].dropna().unique())
month_options = ['All'] + available_months
month_choice = st.selectbox('Select Month', month_options, format_func=lambda x: x)
if month_choice == 'All':
    table_df = filtered_df
else:
    table_df = filtered_df[filtered_df['Period'] == month_choice]
st.header('Summary Table')
st.dataframe(summary_table(table_df).style.format({'Total Amount': format_total_amount}).apply(highlight_hours, subset=['Bucket']))

def plot_time_series_split(df, dice_hide_ic, company, worker, country):
    if 'Period' not in df.columns:
        st.write('No period column found.')
        return
    ts = df.groupby(['Period', 'Bucket'])['Payment Amount (payment currency)'].sum().reset_index()
    ts = ts.pivot(index='Period', columns='Bucket', values='Payment Amount (payment currency)').fillna(0)
    ts = ts.sort_index()
    import matplotlib.ticker as mticker
    from datetime import datetime
    import pandas as pd
    # Get all periods in the last 12 months
    all_periods = pd.date_range(
        start=(datetime.today().replace(day=1) - pd.DateOffset(months=12)).strftime('%Y-%m'),
        end=datetime.today().replace(day=1).strftime('%Y-%m'),
        freq='MS'
    ).strftime('%Y-%m')
    # Reindex to ensure all months are present
    ts = ts.reindex(all_periods, fill_value=0)
    # Costs only (exclude revenue and profit)
    cost_buckets_all = [b for b in ts.columns if b.lower() not in ['revenue', 'profit'] and 'revenue' not in b.lower()]
    # Exclude 'Prepaid Deel credits'
    cost_buckets_all = [b for b in cost_buckets_all if b.lower() != 'prepaid deel credits']
    # Optionally filter out IC rate (robust: exclude if both 'ic' and 'rate' in name)
    if dice_hide_ic:
        cost_buckets_all = [b for b in cost_buckets_all if not ('ic' in b.lower() and 'rate' in b.lower())]
    # Find top 5 cost buckets by total amount
    cost_totals = ts[cost_buckets_all].sum().sort_values(ascending=False)
    top5_cost_buckets = list(cost_totals.head(5).index)
    st.subheader('Costs Over Time (Top 5{})'.format(' (IC Rate Hidden)' if dice_hide_ic else ''))
    if top5_cost_buckets:
        fig1, ax1 = plt.subplots(figsize=(10, 4))
        ts[top5_cost_buckets].plot(ax=ax1, marker='o')
        ax1.set_ylabel('Amount')
        ax1.set_title('Costs Over Time (Top 5{})'.format(' (IC Rate Hidden)' if dice_hide_ic else ''))
        ax1.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: human_format(x)))
        st.pyplot(fig1)
    else:
        st.write('No cost buckets found.')
    # Revenue only (all revenue buckets)
    revenue_buckets = [b for b in ts.columns if 'revenue' in b.lower()]
    st.subheader('Revenue Over Time (All Types)')
    if revenue_buckets:
        fig2, ax2 = plt.subplots(figsize=(10, 4))
        ts[revenue_buckets].plot(ax=ax2, marker='o')
        ax2.set_ylabel('Amount')
        ax2.set_title('Revenue Over Time (All Types)')
        ax2.legend(title='Revenue Bucket')
        ax2.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: human_format(x)))
        st.pyplot(fig2)
    else:
        st.write('No revenue buckets found.')
    # Profit only: only show if any filter is set
    if not (company == 'All' and worker == 'All' and country == 'All'):
        profit_bucket = [b for b in ts.columns if b.lower() == 'profit']
        st.subheader('Profit Over Time')
        if profit_bucket:
            profit_series = ts[profit_bucket[0]] if profit_bucket[0] in ts else pd.Series(dtype=float)
            fig3, ax3 = plt.subplots(figsize=(10, 4))
            profit_series.plot(ax=ax3, marker='o', color='green')
            ax3.set_ylabel('Amount')
            ax3.set_title('Profit Over Time')
            ax3.legend(['Profit'])
            ax3.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: human_format(x)))
            st.pyplot(fig3)
        else:
            st.write('No profit bucket found.')

st.header('Time Series (by Payment Cycle Period)')
plot_time_series_split(filtered_df_last12, dice_hide_ic, company_display, worker, country)

def plot_bucket_bar_split(df):
    bucket_sum = df.groupby('Bucket')['Payment Amount (payment currency)'].sum()
    import matplotlib.ticker as mticker
    # Costs only
    cost_buckets_all = [b for b in bucket_sum.index if b.lower() not in ['revenue', 'profit'] and 'revenue' not in b.lower()]
    # Exclude 'Prepaid Deel credits'
    cost_buckets_all = [b for b in cost_buckets_all if b.lower() != 'prepaid deel credits']
    # Top 5 cost buckets
    top5_cost_buckets = list(bucket_sum[cost_buckets_all].sort_values(ascending=False).head(5).index)
    st.subheader('Costs by Bucket (Top 5, All Time)')
    if top5_cost_buckets:
        fig1, ax1 = plt.subplots(figsize=(8, 3))
        bucket_sum[top5_cost_buckets].sort_values().plot(kind='barh', ax=ax1, color='salmon')
        ax1.set_xlabel('Total Amount')
        ax1.set_title('Costs by Bucket (Top 5)')
        ax1.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: human_format(x)))
        st.pyplot(fig1)
    else:
        st.write('No cost buckets found.')

st.header('Bucket Breakdown (All Time)')
plot_bucket_bar_split(filtered_df)

def plot_margin_over_time(df):
    import matplotlib.ticker as mticker
    from datetime import datetime
    # Prepare data
    margin_df = df[df['Period'].notna()].copy()
    # Group by period
    profit_by_month = margin_df[margin_df['Bucket'].str.lower() == 'profit'].groupby('Period')['Payment Amount (payment currency)'].sum()
    revenue_by_month = margin_df[margin_df['Bucket'].str.lower().str.contains('revenue')].groupby('Period')['Payment Amount (payment currency)'].sum()
    # Get all periods in the last 12 months
    all_periods = pd.date_range(
        start=(datetime.today().replace(day=1) - pd.DateOffset(months=12)).strftime('%Y-%m'),
        end=datetime.today().replace(day=1).strftime('%Y-%m'),
        freq='MS'
    ).strftime('%Y-%m')
    margin_series = []
    for period in all_periods:
        profit = profit_by_month.get(period, None)
        revenue = revenue_by_month.get(period, None)
        if profit is None or pd.isna(profit):
            margin = None
        elif revenue is None or revenue == 0 or pd.isna(revenue):
            margin = None
        else:
            margin = profit / revenue * 100
        margin_series.append(margin)
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(all_periods, margin_series, marker='o', color='purple')
    ax.set_ylabel('Margin (%)')
    ax.set_title('Margin Over Time (Last 12 Months)')
    ax.set_xticklabels(all_periods, rotation=45)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'{x:.1f}%' if pd.notna(x) else ''))
    st.pyplot(fig)

st.header('Margin Over Time (Last 12 Months)')
plot_margin_over_time(filtered_df_last12) 
