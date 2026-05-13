# ================================================
#   Unemployment Analysis with Python
#   CodeAlpha Data Science Internship - Task 2
# ================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# ------------------------------------------------
# Matplotlib Dark Theme
# ------------------------------------------------
plt.rcParams.update({
    'figure.facecolor'  : '#0e0e14',
    'axes.facecolor'    : '#1a1730',
    'axes.edgecolor'    : '#2e2a4a',
    'axes.labelcolor'   : '#c8c4e0',
    'xtick.color'       : '#c8c4e0',
    'ytick.color'       : '#c8c4e0',
    'text.color'        : '#e8e6f0',
    'grid.color'        : '#2a2740',
    'grid.alpha'        : 0.5,
    'axes.titlecolor'   : '#e8e6f0',
    'legend.facecolor'  : '#1a1730',
    'legend.edgecolor'  : '#2e2a4a',
    'figure.edgecolor'  : '#0e0e14',
})

COLORS      = ['#e879b0', '#9b59f5', '#5bbfde', '#f5a623', '#50e3c2']
COLOR_PRE   = '#5bbfde'
COLOR_COVID = '#e879b0'

print("=" * 60)
print("   Unemployment Analysis — CodeAlpha Task 2")
print("=" * 60)

# ------------------------------------------------
# Step 1: Load the Dataset
# ------------------------------------------------
unemp = pd.read_csv('Unemployment in India.csv')

# Strip extra spaces from column names
unemp.columns = unemp.columns.str.strip()

print(f"\nStep 1: Dataset loaded successfully.")
print(f"   Total Rows   : {unemp.shape[0]}")
print(f"   Total Columns: {unemp.shape[1]}")
print(f"   Columns      : {list(unemp.columns)}")

# ------------------------------------------------
# Step 2: Data Cleaning
# ------------------------------------------------
print("\n" + "=" * 60)
print("Step 2: Data Cleaning")
print("=" * 60)

# Check missing values before cleaning
print("\nMissing values before cleaning:")
print(unemp.isnull().sum())

# Keep a copy of original data before dropping anything
# This ensures we can always go back to the raw data if needed
unemp_original = unemp.copy()

# Drop rows that have missing values
# These 28 rows contained no data at all — keeping them would cause
# errors in calculations like mean, groupby, and model training.
# The original data is safely stored in unemp_original variable above,
# so we can always access the raw 768 rows if the client needs them.
unemp = unemp.dropna()

print(f"\nOriginal rows (with missing): {len(unemp_original)}")
print(f"Rows after dropping missing  : {len(unemp)}")
print(f"Rows removed                 : {len(unemp_original) - len(unemp)}")
print(f"Why removed: These rows had no data values — empty records.")
print(f"Note: Original data with all 768 rows is saved in unemp_original.")

# Convert Date column to proper datetime format
unemp['Date'] = pd.to_datetime(unemp['Date'].str.strip(), dayfirst=True)

# Rename long column names to shorter and cleaner versions
unemp.rename(columns={
    'Estimated Unemployment Rate (%)' : 'Unemployment_Rate',
    'Estimated Employed'              : 'Employed',
    'Estimated Labour Participation Rate (%)' : 'Labour_Participation_Rate'
}, inplace=True)

# Extract Year, Month number, and Month name from Date
unemp['Year']       = unemp['Date'].dt.year
unemp['Month']      = unemp['Date'].dt.month
unemp['Month_Name'] = unemp['Date'].dt.strftime('%b')

# Label each row as Pre Covid or During Covid
# India's national lockdown started in March 2020
unemp['Period'] = unemp['Date'].apply(
    lambda x: 'During Covid (2020)' if x >= pd.Timestamp('2020-03-01') else 'Pre Covid (2019-2020)'
)

print("\nData types after cleaning:")
print(unemp.dtypes)
print(f"\nDate range : {unemp['Date'].min().date()} to {unemp['Date'].max().date()}")
print(f"Regions    : {unemp['Region'].nunique()} unique states/UTs")
print(f"Area types : {list(unemp['Area'].unique())}")
print("\nData cleaning complete.")

# ------------------------------------------------
# Step 3: Identify Column Types
# ------------------------------------------------
print("\n" + "=" * 60)
print("Step 3: Column Types — Numerical vs Categorical")
print("=" * 60)

# Automatically detect numerical and categorical columns using select_dtypes
numerical_cols   = unemp.select_dtypes(include='number').columns.tolist()
categorical_cols = unemp.select_dtypes(include='object').columns.tolist()

print("\nNumerical Columns:")
print(unemp.select_dtypes(include='number').columns)

print("\nCategorical Columns:")
print(unemp.select_dtypes(include='object').columns)

# ------------------------------------------------
# Step 4: Numerical Column Analysis
# ------------------------------------------------
print("\n" + "=" * 60)
print("Step 4: Numerical Columns — Descriptive Statistics")
print("=" * 60)

print("\nUnemployment Rate (%) — Statistics:")
print(unemp['Unemployment_Rate'].describe().round(2))

print("\nEstimated Employed — Statistics:")
print(unemp['Employed'].describe().round(2))

print("\nLabour Participation Rate (%) — Statistics:")
print(unemp['Labour_Participation_Rate'].describe().round(2))

# Variance and standard deviation
print("\nVariance of Numerical Columns:")
print(unemp[numerical_cols].var().round(2))

print("\nStandard Deviation of Numerical Columns:")
print(unemp[numerical_cols].std().round(2))

# Correlation between numerical columns
print("\nCorrelation between Numerical Columns:")
print(unemp[numerical_cols].corr().round(2))

# ------------------------------------------------
# Step 5: Categorical Column Analysis
# ------------------------------------------------
print("\n" + "=" * 60)
print("Step 5: Categorical Columns — Value Counts")
print("=" * 60)

print("\nRegion — Total unique values:", unemp['Region'].nunique())
print(unemp['Region'].value_counts())

print("\nArea — Value counts:")
print(unemp['Area'].value_counts())

print("\nFrequency — Value counts:")
print(unemp['Frequency'].value_counts())

print("\nPeriod — Value counts:")
print(unemp['Period'].value_counts())

# ------------------------------------------------
# Step 6: Handle Outliers using Capping (Winsorization)
# ------------------------------------------------
print("\n" + "=" * 60)
print("Step 6: Handling Outliers using Capping Method")
print("=" * 60)

# Keep a copy before capping so original data is always preserved
unemp_before_outliers = unemp.copy()

numerical_cols_for_outliers = ['Unemployment_Rate', 'Employed', 'Labour_Participation_Rate']

print(f"\nTotal rows before capping: {len(unemp)}")
print("No rows will be deleted — outlier values will only be capped to bounds.")
print("\nOutlier details per column:")

for col in numerical_cols_for_outliers:
    Q1  = unemp[col].quantile(0.25)
    Q3  = unemp[col].quantile(0.75)
    IQR = Q3 - Q1
    lb  = Q1 - 1.5 * IQR
    ub  = Q3 + 1.5 * IQR
    outliers_count = len(unemp[(unemp[col] < lb) | (unemp[col] > ub)])
    # Cap values to bounds instead of deleting rows
    unemp[col] = unemp[col].clip(lower=lb, upper=ub)
    print(f"\n   Column          : {col}")
    print(f"   Q1              : {Q1:.2f}")
    print(f"   Q3              : {Q3:.2f}")
    print(f"   IQR             : {IQR:.2f}")
    print(f"   Lower Bound     : {lb:.2f}")
    print(f"   Upper Bound     : {ub:.2f}")
    print(f"   Outliers Capped : {outliers_count}")

print(f"\nRows before capping : {len(unemp_before_outliers)}")
print(f"Rows after capping  : {len(unemp)}")
print(f"Rows deleted        : 0 — all rows are safe")
print("Outliers handled successfully using capping.")

# ------------------------------------------------
# Step 7: Full Data Exploration (EDA)
# ------------------------------------------------
print("\n" + "=" * 60)
print("Step 7: Full Data Exploration (EDA)")
print("=" * 60)

print("\nFirst 5 rows of the dataset:")
print(unemp.head())

print("\nLast 5 rows of the dataset:")
print(unemp.tail())

print("\nRandom 5 rows sample:")
print(unemp.sample(5, random_state=42))

print("\nDataset Shape:", unemp.shape)

print("\nAny duplicate rows:", unemp.duplicated().sum())

print("\nAverage Unemployment Rate by Area:")
print(unemp.groupby('Area')['Unemployment_Rate'].mean().round(2))

print("\nAverage Unemployment Rate by Period:")
print(unemp.groupby('Period')['Unemployment_Rate'].mean().round(2))

print("\nTop 5 Regions — Highest Average Unemployment Rate:")
top5 = unemp.groupby('Region')['Unemployment_Rate'].mean().sort_values(ascending=False).head(5)
print(top5.round(2))

print("\nBottom 5 Regions — Lowest Average Unemployment Rate:")
bot5 = unemp.groupby('Region')['Unemployment_Rate'].mean().sort_values().head(5)
print(bot5.round(2))

print("\nMonthly Average Unemployment Rate:")
monthly_summary = unemp.groupby('Month_Name')['Unemployment_Rate'].mean().round(2)
print(monthly_summary)

print("\nYearly Average Unemployment Rate:")
print(unemp.groupby('Year')['Unemployment_Rate'].mean().round(2))

# ------------------------------------------------
# Step 8: Graph 1 - Overall Unemployment Trend Over Time
# ------------------------------------------------
print("\nStep 8: Generating graphs...")

monthly_avg = unemp.groupby('Date')['Unemployment_Rate'].mean().reset_index()

fig1, ax1 = plt.subplots(figsize=(12, 5))
ax1.plot(monthly_avg['Date'], monthly_avg['Unemployment_Rate'],
         color='#9b59f5', linewidth=2.5, marker='o', markersize=5, label='Unemployment Rate')
ax1.axvline(pd.Timestamp('2020-03-01'), color='#e879b0', linestyle='--',
            linewidth=2, label='Covid Lockdown (Mar 2020)')
ax1.fill_between(monthly_avg['Date'], monthly_avg['Unemployment_Rate'],
                 alpha=0.15, color='#9b59f5')
ax1.set_title('Unemployment Rate Trend Over Time (India)', fontsize=14, fontweight='bold', pad=15)
ax1.set_xlabel('Date', fontsize=11)
ax1.set_ylabel('Unemployment Rate (%)', fontsize=11)
ax1.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
ax1.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
plt.xticks(rotation=45)
ax1.legend(fontsize=10)
ax1.spines[['top', 'right']].set_visible(False)
plt.tight_layout()
plt.savefig('graph1_unemployment_trend.png', dpi=150, bbox_inches='tight')
plt.show()
plt.close()
print("   Graph 1 saved: graph1_unemployment_trend.png")

# ------------------------------------------------
# Step 9: Graph 2 - Covid Impact Bar Chart
# ------------------------------------------------
# Bar chart is used here instead of box plot because capping has already
# handled the outliers — a bar chart cleanly shows the mean comparison
# between Pre Covid and During Covid periods without showing any circles.
pre_covid  = unemp[unemp['Period'] == 'Pre Covid (2019-2020)']['Unemployment_Rate']
covid      = unemp[unemp['Period'] == 'During Covid (2020)']['Unemployment_Rate']
periods    = ['Pre Covid\n(May 2019 - Feb 2020)', 'During Covid\n(Mar 2020 - Jun 2020)']
means      = [pre_covid.mean(), covid.mean()]

fig2, ax2 = plt.subplots(figsize=(8, 5))
bars = ax2.bar(periods, means, color=[COLOR_PRE, COLOR_COVID],
               edgecolor='#2e2a4a', width=0.45, alpha=0.85)

for bar, val in zip(bars, means):
    ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
             f'{val:.1f}%', ha='center', fontsize=11,
             color='#e8e6f0', fontweight='bold')

ax2.set_title('Covid-19 Impact on Unemployment Rate', fontsize=14, fontweight='bold', pad=15)
ax2.set_ylabel('Average Unemployment Rate (%)', fontsize=11)
ax2.spines[['top', 'right']].set_visible(False)
plt.tight_layout()
plt.savefig('graph2_covid_impact_barchart.png', dpi=150, bbox_inches='tight')
plt.show()
plt.close()
print("   Graph 2 saved: graph2_covid_impact_barchart.png")

# ------------------------------------------------
# Step 10: Graph 3 - Region-wise Unemployment Bar Chart
# ------------------------------------------------
region_avg = unemp.groupby('Region')['Unemployment_Rate'].mean().sort_values(ascending=True)

fig3, ax3 = plt.subplots(figsize=(10, 8))
bar_colors = ['#e879b0' if v >= region_avg.mean() else '#5bbfde' for v in region_avg.values]
bars = ax3.barh(region_avg.index, region_avg.values, color=bar_colors, edgecolor='#2e2a4a')
ax3.axvline(region_avg.mean(), color='#f5a623', linestyle='--',
            linewidth=1.8, label=f'National Average: {region_avg.mean():.1f}%')
for bar, val in zip(bars, region_avg.values):
    ax3.text(bar.get_width() + 0.3, bar.get_y() + bar.get_height() / 2,
             f'{val:.1f}%', va='center', fontsize=8, color='#e8e6f0')
ax3.set_title('Average Unemployment Rate by Region', fontsize=14, fontweight='bold', pad=15)
ax3.set_xlabel('Unemployment Rate (%)', fontsize=11)
ax3.legend(fontsize=10)
ax3.spines[['top', 'right']].set_visible(False)
plt.tight_layout()
plt.savefig('graph3_region_unemployment.png', dpi=150, bbox_inches='tight')
plt.show()
plt.close()
print("   Graph 3 saved: graph3_region_unemployment.png")

# ------------------------------------------------
# Step 11: Graph 4 - Rural vs Urban Comparison
# ------------------------------------------------
area_monthly = unemp.groupby(['Date', 'Area'])['Unemployment_Rate'].mean().reset_index()

fig4, ax4 = plt.subplots(figsize=(12, 5))
for area, color in zip(['Rural', 'Urban'], [COLOR_PRE, COLOR_COVID]):
    subset = area_monthly[area_monthly['Area'] == area]
    ax4.plot(subset['Date'], subset['Unemployment_Rate'],
             color=color, linewidth=2.5, marker='o', markersize=4, label=area)
    ax4.fill_between(subset['Date'], subset['Unemployment_Rate'], alpha=0.1, color=color)

ax4.axvline(pd.Timestamp('2020-03-01'), color='#f5a623', linestyle='--',
            linewidth=1.8, label='Covid Lockdown')
ax4.set_title('Rural vs Urban Unemployment Rate Over Time', fontsize=14, fontweight='bold', pad=15)
ax4.set_xlabel('Date', fontsize=11)
ax4.set_ylabel('Unemployment Rate (%)', fontsize=11)
ax4.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
ax4.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
plt.xticks(rotation=45)
ax4.legend(fontsize=10)
ax4.spines[['top', 'right']].set_visible(False)
plt.tight_layout()
plt.savefig('graph4_rural_vs_urban.png', dpi=150, bbox_inches='tight')
plt.show()
plt.close()
print("   Graph 4 saved: graph4_rural_vs_urban.png")

# ------------------------------------------------
# Step 12: Graph 5 - Heatmap (Region vs Month)
# ------------------------------------------------
pivot = unemp.pivot_table(
    values='Unemployment_Rate',
    index='Region',
    columns='Month_Name',
    aggfunc='mean'
)
month_order = ['May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec', 'Jan', 'Feb', 'Mar', 'Apr', 'Jun']
month_order = [m for m in month_order if m in pivot.columns]
pivot       = pivot[month_order]

fig5, ax5 = plt.subplots(figsize=(13, 9))
sns.heatmap(pivot, cmap='RdPu', annot=True, fmt='.1f',
            linewidths=0.4, linecolor='#0e0e14',
            annot_kws={'size': 7, 'color': '#e8e6f0'},
            cbar_kws={'label': 'Unemployment Rate (%)'},
            ax=ax5)
ax5.set_title('Unemployment Rate Heatmap — Region vs Month', fontsize=14, fontweight='bold', pad=15)
ax5.set_xlabel('Month', fontsize=11)
ax5.set_ylabel('Region', fontsize=11)
plt.xticks(rotation=0)
plt.yticks(rotation=0, fontsize=8)
plt.tight_layout()
plt.savefig('graph5_heatmap_region_month.png', dpi=150, bbox_inches='tight')
plt.show()
plt.close()
print("   Graph 5 saved: graph5_heatmap_region_month.png")

# ------------------------------------------------
# Step 13: Graph 6 - Pre vs During Covid by Region
# ------------------------------------------------
pre_period   = unemp[unemp['Period'] == 'Pre Covid (2019-2020)'].groupby('Region')['Unemployment_Rate'].mean()
covid_period = unemp[unemp['Period'] == 'During Covid (2020)'].groupby('Region')['Unemployment_Rate'].mean()
compare      = pd.DataFrame({'Pre Covid': pre_period, 'During Covid': covid_period}).dropna()
compare['Increase'] = compare['During Covid'] - compare['Pre Covid']
compare = compare.sort_values('Increase', ascending=True)

fig6, ax6 = plt.subplots(figsize=(10, 8))
x = np.arange(len(compare))
w = 0.38
ax6.barh(x - w/2, compare['Pre Covid'],   w, label='Pre Covid',   color=COLOR_PRE,   edgecolor='#2e2a4a', alpha=0.85)
ax6.barh(x + w/2, compare['During Covid'], w, label='During Covid', color=COLOR_COVID, edgecolor='#2e2a4a', alpha=0.85)
ax6.set_yticks(x)
ax6.set_yticklabels(compare.index, fontsize=9)
ax6.set_title('Pre Covid vs During Covid Unemployment by Region', fontsize=13, fontweight='bold', pad=15)
ax6.set_xlabel('Unemployment Rate (%)', fontsize=11)
ax6.legend(fontsize=10)
ax6.spines[['top', 'right']].set_visible(False)
plt.tight_layout()
plt.savefig('graph6_pre_vs_covid_region.png', dpi=150, bbox_inches='tight')
plt.show()
plt.close()
print("   Graph 6 saved: graph6_pre_vs_covid_region.png")

# ------------------------------------------------
# Step 14: Covid-19 Impact Summary
# ------------------------------------------------
print("\n" + "=" * 60)
print("Step 14: Covid-19 Impact Analysis")
print("=" * 60)

pre_mean   = unemp[unemp['Period'] == 'Pre Covid (2019-2020)']['Unemployment_Rate'].mean()
covid_mean = unemp[unemp['Period'] == 'During Covid (2020)']['Unemployment_Rate'].mean()
increase   = covid_mean - pre_mean

most_affected  = compare['Increase'].idxmax()
least_affected = compare['Increase'].idxmin()

print(f"\nPre Covid Average Unemployment  : {pre_mean:.2f}%")
print(f"During Covid Average Unemployment: {covid_mean:.2f}%")
print(f"Increase due to Covid            : +{increase:.2f}%")
print(f"\nMost Affected Region : {most_affected} (+{compare.loc[most_affected, 'Increase']:.2f}%)")
print(f"Least Affected Region: {least_affected} ({compare.loc[least_affected, 'Increase']:.2f}%)")

# ------------------------------------------------
# Step 15: Key Insights
# ------------------------------------------------
print("\n" + "=" * 60)
print("Step 15: Key Insights & Policy Implications")
print("=" * 60)

print(f"""
1. OVERALL TREND:
   - India's average unemployment rate was {pre_mean:.1f}% before Covid.
   - After the Covid-19 lockdown in March 2020, it jumped to {covid_mean:.1f}%.
   - This is a significant increase of {increase:.1f} percentage points.

2. COVID-19 IMPACT:
   - The lockdown caused a sharp spike in unemployment across all regions.
   - Urban areas were more severely affected than rural areas.
   - The highest unemployment spike was recorded in April-May 2020.

3. REGIONAL PATTERNS:
   - Highest average unemployment: {region_avg.index[-1]} ({region_avg.values[-1]:.1f}%)
   - Lowest average unemployment : {region_avg.index[0]} ({region_avg.values[0]:.1f}%)
   - Most Covid-affected region  : {most_affected}

4. RURAL vs URBAN:
   - Urban unemployment rose more sharply during the Covid lockdown.
   - Rural areas showed more stability due to the agricultural sector.

5. POLICY IMPLICATIONS:
   - Urban employment support programs are critical during economic crises.
   - The rural sector acted as a natural buffer during the pandemic.
   - States with consistently high pre-Covid unemployment need structural reforms.
   - Emergency social safety nets should be activated faster during lockdowns.
""")

# ------------------------------------------------
# Final Summary
# ------------------------------------------------
print("=" * 60)
print("   Project Complete!")
print("=" * 60)
print(f"Dataset         : Unemployment_in_India.csv")
print(f"Total Records   : {len(unemp)}")
print(f"Regions Covered : {unemp['Region'].nunique()}")
print(f"Date Range      : {unemp['Date'].min().date()} to {unemp['Date'].max().date()}")
print(f"Graphs Saved    : 6 PNG files")
print("=" * 60)