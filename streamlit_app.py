# ================================================
#   Unemployment Analysis — Streamlit App
#   CodeAlpha Data Science Internship - Task 2
# ================================================

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# ------------------------------------------------
# Page Configuration
# ------------------------------------------------
st.set_page_config(
    page_title="Unemployment Analysis India",
    page_icon="📊",
    layout="wide"
)

# ------------------------------------------------
# Dark Theme CSS
# ------------------------------------------------
st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@600;700&family=DM+Sans:wght@300;400;500&display=swap');

        html, body, [class*="css"] {
            font-family: 'DM Sans', sans-serif;
            background-color: #080b12;
            color: #e2e0f0;
        }
        .stApp { background-color: #080b12; }

        .main-title {
            font-family: 'Playfair Display', serif;
            font-size: 2.8rem;
            font-weight: 700;
            background: linear-gradient(135deg, #5bbfde, #9b59f5, #e879b0);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            text-align: center;
            letter-spacing: 0.01em;
            margin-bottom: 0.2rem;
        }
        .subtitle {
            text-align: center;
            color: #5a5678;
            font-size: 0.88rem;
            letter-spacing: 0.08em;
            text-transform: uppercase;
            margin-bottom: 1.5rem;
        }
        .section-header {
            font-family: 'Playfair Display', serif;
            font-size: 1.4rem;
            font-weight: 600;
            color: #9b59f5;
            border-left: 3px solid #e879b0;
            padding-left: 0.8rem;
            margin-top: 1.8rem;
            margin-bottom: 1rem;
        }
        .metric-card {
            background: linear-gradient(135deg, #0f1220, #161929);
            border-radius: 14px;
            padding: 1.3rem 1rem;
            text-align: center;
            border: 1px solid #1e2238;
            box-shadow: 0 4px 24px rgba(91, 191, 222, 0.06);
        }
        .metric-value {
            font-family: 'Playfair Display', serif;
            font-size: 2rem;
            font-weight: 700;
            background: linear-gradient(135deg, #5bbfde, #9b59f5);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }
        .metric-label {
            font-size: 0.75rem;
            color: #5a5678;
            margin-top: 0.3rem;
            text-transform: uppercase;
            letter-spacing: 0.09em;
        }
        .insight-card {
            background: linear-gradient(135deg, #0f1220, #12162a);
            border-radius: 12px;
            padding: 1.2rem 1.4rem;
            border: 1px solid #1e2238;
            margin-bottom: 0.8rem;
        }
        .insight-number {
            font-family: 'Playfair Display', serif;
            font-size: 1.8rem;
            color: #e879b0;
            line-height: 1;
        }
        .insight-title {
            font-size: 0.9rem;
            font-weight: 600;
            color: #9b59f5;
            text-transform: uppercase;
            letter-spacing: 0.06em;
            margin-bottom: 0.4rem;
        }
        .insight-text {
            font-size: 0.88rem;
            color: #9e9bbf;
            line-height: 1.6;
        }
        div[data-testid="stSidebar"] {
            background: linear-gradient(180deg, #060912, #0c0f1c) !important;
            border-right: 1px solid #1a1d2e !important;
        }
        section[data-testid="stSidebar"] {
            background: linear-gradient(180deg, #060912, #0c0f1c) !important;
        }
        div[data-testid="stSidebar"] * { color: #c8c4e0 !important; }
        div[data-testid="stMetric"] {
            background: #0f1220;
            border-radius: 10px;
            padding: 0.8rem 1rem;
            border: 1px solid #1e2238;
        }
        hr { border-color: #1a1d2e; }
    </style>
""", unsafe_allow_html=True)

# ------------------------------------------------
# Matplotlib Dark Theme
# ------------------------------------------------
plt.rcParams.update({
    'figure.facecolor' : '#080b12',
    'axes.facecolor'   : '#0f1220',
    'axes.edgecolor'   : '#1e2238',
    'axes.labelcolor'  : '#c8c4e0',
    'xtick.color'      : '#c8c4e0',
    'ytick.color'      : '#c8c4e0',
    'text.color'       : '#e2e0f0',
    'grid.color'       : '#1a1d2e',
    'grid.alpha'       : 0.6,
    'axes.titlecolor'  : '#e2e0f0',
    'legend.facecolor' : '#0f1220',
    'legend.edgecolor' : '#1e2238',
    'figure.edgecolor' : '#080b12',
})

COLOR_PRE   = '#5bbfde'
COLOR_COVID = '#e879b0'

# ------------------------------------------------
# Load and Prepare Data
# ------------------------------------------------
@st.cache_data
def load_data():
    unemp = pd.read_csv('Unemployment in India.csv')
    unemp.columns = unemp.columns.str.strip()

    unemp_original = unemp.copy()
    unemp = unemp.dropna()

    unemp['Date'] = pd.to_datetime(unemp['Date'].str.strip(), dayfirst=True)

    unemp.rename(columns={
        'Estimated Unemployment Rate (%)'         : 'Unemployment_Rate',
        'Estimated Employed'                      : 'Employed',
        'Estimated Labour Participation Rate (%)' : 'Labour_Participation_Rate'
    }, inplace=True)

    unemp['Year']       = unemp['Date'].dt.year
    unemp['Month']      = unemp['Date'].dt.month
    unemp['Month_Name'] = unemp['Date'].dt.strftime('%b')
    unemp['Period']     = unemp['Date'].apply(
        lambda x: 'During Covid (2020)'
        if x >= pd.Timestamp('2020-03-01') else 'Pre Covid (2019-2020)'
    )

    # Handle outliers using capping — no rows deleted
    for col in ['Unemployment_Rate', 'Employed', 'Labour_Participation_Rate']:
        Q1  = unemp[col].quantile(0.25)
        Q3  = unemp[col].quantile(0.75)
        IQR = Q3 - Q1
        unemp[col] = unemp[col].clip(lower=Q1 - 1.5*IQR, upper=Q3 + 1.5*IQR)

    return unemp, unemp_original

try:
    unemp, unemp_original = load_data()
except FileNotFoundError:
    st.error("Unemployment_in_India.csv not found. Place it in the same folder as app.py.")
    st.stop()

# ------------------------------------------------
# Compute Key Statistics
# ------------------------------------------------
pre_mean   = unemp[unemp['Period'] == 'Pre Covid (2019-2020)']['Unemployment_Rate'].mean()
covid_mean = unemp[unemp['Period'] == 'During Covid (2020)']['Unemployment_Rate'].mean()
increase   = covid_mean - pre_mean
region_avg = unemp.groupby('Region')['Unemployment_Rate'].mean()
area_avg   = unemp.groupby('Area')['Unemployment_Rate'].mean()

pre_period   = unemp[unemp['Period'] == 'Pre Covid (2019-2020)'].groupby('Region')['Unemployment_Rate'].mean()
covid_period = unemp[unemp['Period'] == 'During Covid (2020)'].groupby('Region')['Unemployment_Rate'].mean()
compare      = pd.DataFrame({'Pre Covid': pre_period, 'During Covid': covid_period}).dropna()
compare['Increase'] = compare['During Covid'] - compare['Pre Covid']

most_affected  = compare['Increase'].idxmax()
least_affected = compare['Increase'].idxmin()

# ------------------------------------------------
# Header
# ------------------------------------------------
st.markdown('<div class="main-title">📊 Unemployment Analysis — India</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">CodeAlpha Data Science Internship &nbsp;·&nbsp; Task 2 &nbsp;·&nbsp; Aiman &nbsp;·&nbsp; CA/DF1/54987</div>', unsafe_allow_html=True)
st.markdown("---")

# ------------------------------------------------
# Sidebar
# ------------------------------------------------
st.sidebar.markdown("## 📊 Navigation")
page = st.sidebar.radio(
    "Go to",
    ["Overview", "Covid-19 Impact", "Regional Analysis",
     "Rural vs Urban", "Visualizations", "Key Insights"]
)
st.sidebar.markdown("---")
st.sidebar.markdown("### Dataset Info")
st.sidebar.markdown(f"**Total Records :** {len(unemp)}")
st.sidebar.markdown(f"**Regions       :** {unemp['Region'].nunique()}")
st.sidebar.markdown(f"**Date Range    :** May 2019 — Jun 2020")
st.sidebar.markdown(f"**Outlier Method:** Capping (IQR)")
st.sidebar.markdown(f"**Missing Rows  :** 28 dropped")


# ================================================
# Page 1 — Overview
# ================================================
if page == "Overview":

    st.markdown('<div class="section-header">Dataset Overview</div>', unsafe_allow_html=True)

    c1, c2, c3, c4, c5 = st.columns(5)
    for col, val, label in zip(
        [c1, c2, c3, c4, c5],
        ["768", "740", "28", f"{unemp['Unemployment_Rate'].mean():.1f}%", "2"],
        ["Raw Rows", "After Cleaning", "Regions", "Overall Avg Rate", "Area Types"]
    ):
        with col:
            st.markdown(f'<div class="metric-card"><div class="metric-value">{val}</div><div class="metric-label">{label}</div></div>', unsafe_allow_html=True)

    st.markdown('<div class="section-header">First 5 Rows</div>', unsafe_allow_html=True)
    st.dataframe(unemp.head(), use_container_width=True)

    st.markdown('<div class="section-header">Statistical Summary</div>', unsafe_allow_html=True)
    st.dataframe(
        unemp[['Unemployment_Rate','Employed','Labour_Participation_Rate']].describe().round(2),
        use_container_width=True
    )

    col1, col2 = st.columns(2)
    with col1:
        st.markdown('<div class="section-header">Numerical Columns</div>', unsafe_allow_html=True)
        for c in unemp.select_dtypes(include='number').columns.tolist():
            st.markdown(f"&nbsp;&nbsp;&nbsp;• **{c}**")
    with col2:
        st.markdown('<div class="section-header">Categorical Columns</div>', unsafe_allow_html=True)
        for c in unemp.select_dtypes(include='object').columns.tolist():
            st.markdown(f"&nbsp;&nbsp;&nbsp;• **{c}**")

    st.markdown('<div class="section-header">Missing Values & Outlier Handling</div>', unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Rows Before Cleaning", 768)
        st.metric("Rows After Cleaning", 740, delta="-28 empty records removed")
        st.info("The 28 removed rows had no data at all. The original 768-row data is preserved in memory.")
    with col2:
        st.metric("Outlier Method", "Capping (IQR)")
        st.metric("Rows Deleted by Capping", 0, delta="All 740 rows safe")
        st.success("Capping limits extreme values to bounds — no rows are deleted, all data stays intact.")


# ================================================
# Page 2 — Covid-19 Impact
# ================================================
elif page == "Covid-19 Impact":

    st.markdown('<div class="section-header">Covid-19 Impact Summary</div>', unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown(f'<div class="metric-card"><div class="metric-value">{pre_mean:.1f}%</div><div class="metric-label">Pre Covid Avg Rate</div></div>', unsafe_allow_html=True)
    with c2:
        st.markdown(f'<div class="metric-card"><div class="metric-value">{covid_mean:.1f}%</div><div class="metric-label">During Covid Avg Rate</div></div>', unsafe_allow_html=True)
    with c3:
        st.markdown(f'<div class="metric-card"><div class="metric-value">+{increase:.1f}%</div><div class="metric-label">Increase Due to Covid</div></div>', unsafe_allow_html=True)

    st.markdown('<div class="section-header">Pre Covid vs During Covid — Comparison</div>', unsafe_allow_html=True)

    fig, ax = plt.subplots(figsize=(7, 4))
    periods = ['Pre Covid\n(May 2019 - Feb 2020)', 'During Covid\n(Mar 2020 - Jun 2020)']
    means   = [pre_mean, covid_mean]
    bars    = ax.bar(periods, means, color=[COLOR_PRE, COLOR_COVID],
                     edgecolor='#1e2238', width=0.45, alpha=0.9)
    for bar, val in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                f'{val:.1f}%', ha='center', fontsize=12,
                color='#e2e0f0', fontweight='bold')
    ax.set_title('Average Unemployment Rate: Pre Covid vs During Covid',
                 fontsize=12, fontweight='bold', pad=12)
    ax.set_ylabel('Unemployment Rate (%)', fontsize=10)
    ax.spines[['top','right']].set_visible(False)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    st.markdown('<div class="section-header">Unemployment Trend with Covid Marker</div>', unsafe_allow_html=True)

    monthly_avg = unemp.groupby('Date')['Unemployment_Rate'].mean().reset_index()
    fig2, ax2   = plt.subplots(figsize=(11, 4))
    ax2.plot(monthly_avg['Date'], monthly_avg['Unemployment_Rate'],
             color='#9b59f5', linewidth=2.5, marker='o', markersize=5,
             label='Unemployment Rate')
    ax2.axvline(pd.Timestamp('2020-03-01'), color=COLOR_COVID, linestyle='--',
                linewidth=2, label='Covid Lockdown (Mar 2020)')
    ax2.fill_between(monthly_avg['Date'], monthly_avg['Unemployment_Rate'],
                     alpha=0.12, color='#9b59f5')
    ax2.set_title('Unemployment Rate Trend Over Time (India)',
                  fontsize=12, fontweight='bold')
    ax2.set_xlabel('Date', fontsize=10)
    ax2.set_ylabel('Unemployment Rate (%)', fontsize=10)
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
    ax2.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    plt.xticks(rotation=45)
    ax2.legend(fontsize=10)
    ax2.spines[['top','right']].set_visible(False)
    plt.tight_layout()
    st.pyplot(fig2)
    plt.close()

    st.markdown('<div class="section-header">Most & Least Affected Regions</div>', unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Most Covid-Affected Region", most_affected,
                  delta=f"+{compare.loc[most_affected,'Increase']:.2f}% increase")
    with col2:
        st.metric("Least Covid-Affected Region", least_affected,
                  delta=f"{compare.loc[least_affected,'Increase']:.2f}% change")


# ================================================
# Page 3 — Regional Analysis
# ================================================
elif page == "Regional Analysis":

    st.markdown('<div class="section-header">Average Unemployment Rate by Region</div>', unsafe_allow_html=True)

    sorted_region = region_avg.sort_values()
    bar_colors = ['#e879b0' if v >= region_avg.mean() else '#5bbfde'
                  for v in sorted_region.values]
    fig, ax = plt.subplots(figsize=(10, 9))
    bars = ax.barh(sorted_region.index, sorted_region.values,
                   color=bar_colors, edgecolor='#1e2238')
    ax.axvline(region_avg.mean(), color='#f5a623', linestyle='--',
               linewidth=1.8, label=f'National Avg: {region_avg.mean():.1f}%')
    for bar, val in zip(bars, sorted_region.values):
        ax.text(bar.get_width() + 0.2, bar.get_y() + bar.get_height()/2,
                f'{val:.1f}%', va='center', fontsize=8, color='#e2e0f0')
    ax.set_title('Average Unemployment Rate by Region',
                 fontsize=13, fontweight='bold', pad=12)
    ax.set_xlabel('Unemployment Rate (%)', fontsize=10)
    ax.legend(fontsize=10)
    ax.spines[['top','right']].set_visible(False)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    st.markdown('<div class="section-header">Pre Covid vs During Covid by Region</div>', unsafe_allow_html=True)

    compare_sorted = compare.sort_values('Increase')
    fig2, ax2 = plt.subplots(figsize=(10, 9))
    x = np.arange(len(compare_sorted))
    w = 0.38
    ax2.barh(x - w/2, compare_sorted['Pre Covid'],   w, label='Pre Covid',
             color=COLOR_PRE,   edgecolor='#1e2238', alpha=0.85)
    ax2.barh(x + w/2, compare_sorted['During Covid'], w, label='During Covid',
             color=COLOR_COVID, edgecolor='#1e2238', alpha=0.85)
    ax2.set_yticks(x)
    ax2.set_yticklabels(compare_sorted.index, fontsize=8)
    ax2.set_title('Pre Covid vs During Covid Unemployment by Region',
                  fontsize=12, fontweight='bold', pad=12)
    ax2.set_xlabel('Unemployment Rate (%)', fontsize=10)
    ax2.legend(fontsize=10)
    ax2.spines[['top','right']].set_visible(False)
    plt.tight_layout()
    st.pyplot(fig2)
    plt.close()

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Highest Unemployment", region_avg.idxmax(),
                  delta=f"{region_avg.max():.1f}%")
    with col2:
        st.metric("Lowest Unemployment", region_avg.idxmin(),
                  delta=f"{region_avg.min():.1f}%")
    with col3:
        st.metric("National Average", f"{region_avg.mean():.1f}%")


# ================================================
# Page 4 — Rural vs Urban
# ================================================
elif page == "Rural vs Urban":

    st.markdown('<div class="section-header">Rural vs Urban Unemployment</div>', unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    with c1:
        st.markdown(f'<div class="metric-card"><div class="metric-value">{area_avg["Rural"]:.1f}%</div><div class="metric-label">Rural Average</div></div>', unsafe_allow_html=True)
    with c2:
        st.markdown(f'<div class="metric-card"><div class="metric-value">{area_avg["Urban"]:.1f}%</div><div class="metric-label">Urban Average</div></div>', unsafe_allow_html=True)

    st.markdown('<div class="section-header">Trend Over Time — Rural vs Urban</div>', unsafe_allow_html=True)

    area_monthly = unemp.groupby(['Date','Area'])['Unemployment_Rate'].mean().reset_index()
    fig, ax = plt.subplots(figsize=(11, 4))
    for area, color in zip(['Rural','Urban'], [COLOR_PRE, COLOR_COVID]):
        subset = area_monthly[area_monthly['Area'] == area]
        ax.plot(subset['Date'], subset['Unemployment_Rate'],
                color=color, linewidth=2.5, marker='o', markersize=4, label=area)
        ax.fill_between(subset['Date'], subset['Unemployment_Rate'],
                        alpha=0.1, color=color)
    ax.axvline(pd.Timestamp('2020-03-01'), color='#f5a623', linestyle='--',
               linewidth=1.8, label='Covid Lockdown')
    ax.set_title('Rural vs Urban Unemployment Rate Over Time',
                 fontsize=12, fontweight='bold', pad=12)
    ax.set_xlabel('Date', fontsize=10)
    ax.set_ylabel('Unemployment Rate (%)', fontsize=10)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    plt.xticks(rotation=45)
    ax.legend(fontsize=10)
    ax.spines[['top','right']].set_visible(False)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    st.markdown('<div class="section-header">Area Value Counts</div>', unsafe_allow_html=True)
    st.dataframe(
        unemp['Area'].value_counts().reset_index().rename(
            columns={'index':'Area','Area':'Count'}),
        use_container_width=True
    )


# ================================================
# Page 5 — Visualizations
# ================================================
elif page == "Visualizations":

    st.markdown('<div class="section-header">Region vs Month Heatmap</div>', unsafe_allow_html=True)

    pivot = unemp.pivot_table(
        values='Unemployment_Rate', index='Region',
        columns='Month_Name', aggfunc='mean'
    )
    month_order = [m for m in ['May','Jun','Jul','Aug','Sep','Oct',
                               'Nov','Dec','Jan','Feb','Mar','Apr']
                   if m in pivot.columns]
    pivot = pivot[month_order]

    fig, ax = plt.subplots(figsize=(12, 9))
    sns.heatmap(pivot, cmap='RdPu', annot=True, fmt='.1f',
                linewidths=0.4, linecolor='#080b12',
                annot_kws={'size': 7, 'color': '#e2e0f0'},
                cbar_kws={'label': 'Unemployment Rate (%)'},
                ax=ax)
    ax.set_title('Unemployment Rate Heatmap — Region vs Month',
                 fontsize=13, fontweight='bold', pad=12)
    ax.set_xlabel('Month', fontsize=10)
    ax.set_ylabel('Region', fontsize=10)
    plt.xticks(rotation=0)
    plt.yticks(rotation=0, fontsize=8)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    st.markdown('<div class="section-header">Correlation Heatmap</div>', unsafe_allow_html=True)

    fig2, ax2 = plt.subplots(figsize=(6, 4))
    corr = unemp[['Unemployment_Rate','Employed','Labour_Participation_Rate']].corr()
    sns.heatmap(corr, annot=True, fmt='.2f', cmap='RdPu',
                square=True, linewidths=0.5, linecolor='#1e2238',
                annot_kws={'size': 11, 'color': '#e2e0f0'}, ax=ax2)
    ax2.set_title('Feature Correlation Heatmap', fontsize=12, fontweight='bold')
    plt.tight_layout()
    st.pyplot(fig2)
    plt.close()

    st.markdown('<div class="section-header">Yearly Average Unemployment</div>', unsafe_allow_html=True)
    yearly = unemp.groupby('Year')['Unemployment_Rate'].mean().reset_index()
    yearly.columns = ['Year', 'Avg Unemployment Rate (%)']
    yearly['Avg Unemployment Rate (%)'] = yearly['Avg Unemployment Rate (%)'].round(2)
    st.dataframe(yearly, use_container_width=True)


# ================================================
# Page 6 — Key Insights
# ================================================
elif page == "Key Insights":

    st.markdown('<div class="section-header">Key Insights & Policy Implications</div>', unsafe_allow_html=True)

    insights = [
        {
            "num": "01", "title": "Overall Trend",
            "text": f"India's average unemployment was {pre_mean:.1f}% before Covid-19. After the national lockdown in March 2020, it jumped to {covid_mean:.1f}% — a rise of {increase:.1f} percentage points in just a few months."
        },
        {
            "num": "02", "title": "Covid-19 Impact",
            "text": f"The lockdown caused a nationwide spike in unemployment. The most severely affected region was {most_affected}, while {least_affected} showed the least change. The peak was recorded in April–May 2020."
        },
        {
            "num": "03", "title": "Regional Patterns",
            "text": "Tripura recorded the highest average unemployment at 28.4%, while Meghalaya had the lowest at 4.8%. Regions with pre-existing economic fragility suffered the most during the pandemic."
        },
        {
            "num": "04", "title": "Rural vs Urban",
            "text": f"Urban unemployment ({area_avg['Urban']:.1f}%) was higher than rural ({area_avg['Rural']:.1f}%). Urban areas saw sharper spikes during lockdown because factory shutdowns and service sector closures hit city workers hardest."
        },
        {
            "num": "05", "title": "Rural Resilience",
            "text": "Rural areas showed greater stability during the pandemic. The agricultural sector, being an essential service, continued operating — acting as a natural economic buffer for rural workers."
        },
        {
            "num": "06", "title": "Policy Implications",
            "text": "Urban employment safety nets must be activated quickly during crises. States with consistently high pre-Covid unemployment need structural reforms. Emergency financial support should reach workers within days, not weeks."
        },
    ]

    col1, col2 = st.columns(2)
    for i, ins in enumerate(insights):
        with (col1 if i % 2 == 0 else col2):
            st.markdown(f"""
                <div class="insight-card">
                    <div class="insight-number">{ins["num"]}</div>
                    <div class="insight-title">{ins["title"]}</div>
                    <div class="insight-text">{ins["text"]}</div>
                </div>
            """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown('<div class="section-header">Summary Statistics</div>', unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)
    for col, val, label in zip(
        [c1, c2, c3, c4],
        [f"{pre_mean:.1f}%", f"{covid_mean:.1f}%", f"+{increase:.1f}%", "28"],
        ["Pre Covid Avg", "Covid Period Avg", "Covid Increase", "Regions Covered"]
    ):
        with col:
            st.markdown(f'<div class="metric-card"><div class="metric-value">{val}</div><div class="metric-label">{label}</div></div>', unsafe_allow_html=True)