
import os, pickle, warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap
import streamlit as st
from datetime import date

warnings.filterwarnings('ignore')

BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
MODEL_PKL   = os.path.join(BASE_DIR, 'models', 'xgb_model.pkl')
CLEANED_CSV = os.path.join(BASE_DIR, 'data',   'Data_Cleaned.csv')

# ─────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="🦟 Dengue Predictor - Sri Lanka",
    layout="wide"
)

# ─────────────────────────────────────────────────────────────
# LOAD MODEL & DATA
# ─────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    with open(MODEL_PKL, 'rb') as f:
        saved = pickle.load(f)
    return saved['model'], saved['features']

@st.cache_data
def load_data():
    return pd.read_csv(CLEANED_CSV)

model, FEATURES = load_model()
df              = load_data()

# ─────────────────────────────────────────────────────────────
# LOOKUP TABLES
# ─────────────────────────────────────────────────────────────
DISTRICT_INFO = {
    "Ampara"       : {"lat": 7.2953,  "lon": 81.6747, "elev": 62,   "prov_enc": 0, "dist_enc": 0},
    "Anuradhapura" : {"lat": 8.3114,  "lon": 80.4037, "elev": 90,   "prov_enc": 4, "dist_enc": 1},
    "Badulla"      : {"lat": 6.9934,  "lon": 81.0550, "elev": 669,  "prov_enc": 7, "dist_enc": 2},
    "Batticaloa"   : {"lat": 7.7170,  "lon": 81.7000, "elev": 2,    "prov_enc": 0, "dist_enc": 3},
    "Colombo"      : {"lat": 6.9271,  "lon": 79.9073, "elev": 4,    "prov_enc": 8, "dist_enc": 4},
    "Galle"        : {"lat": 6.0535,  "lon": 80.2210, "elev": 9,    "prov_enc": 6, "dist_enc": 5},
    "Gampaha"      : {"lat": 7.0873,  "lon": 80.0144, "elev": 19,   "prov_enc": 8, "dist_enc": 6},
    "Hambantota"   : {"lat": 6.1241,  "lon": 81.1185, "elev": 22,   "prov_enc": 6, "dist_enc": 7},
    "Jaffna"       : {"lat": 9.6615,  "lon": 80.0255, "elev": 9,    "prov_enc": 3, "dist_enc": 8},
    "Kalutara"     : {"lat": 6.5854,  "lon": 79.9607, "elev": 5,    "prov_enc": 8, "dist_enc": 9},
    "Kandy"        : {"lat": 7.2906,  "lon": 80.6337, "elev": 499,  "prov_enc": 1, "dist_enc": 10},
    "Kegalle"      : {"lat": 7.2513,  "lon": 80.3464, "elev": 115,  "prov_enc": 5, "dist_enc": 11},
    "Kilinochchi"  : {"lat": 9.3803,  "lon": 80.3770, "elev": 30,   "prov_enc": 3, "dist_enc": 12},
    "Kurunegala"   : {"lat": 7.4867,  "lon": 80.3647, "elev": 116,  "prov_enc": 2, "dist_enc": 13},
    "Mannar"       : {"lat": 8.9810,  "lon": 79.9044, "elev": 2,    "prov_enc": 3, "dist_enc": 14},
    "Matale"       : {"lat": 7.4676,  "lon": 80.6234, "elev": 200,  "prov_enc": 1, "dist_enc": 15},
    "Matara"       : {"lat": 5.9549,  "lon": 80.5550, "elev": 9,    "prov_enc": 6, "dist_enc": 16},
    "Monaragala"   : {"lat": 6.8728,  "lon": 81.3506, "elev": 200,  "prov_enc": 7, "dist_enc": 17},
    "Mullativu"    : {"lat": 9.2671,  "lon": 80.8128, "elev": 8,    "prov_enc": 3, "dist_enc": 18},
    "Nuwara Eliya" : {"lat": 6.9497,  "lon": 80.7891, "elev": 1868, "prov_enc": 1, "dist_enc": 19},
    "Polonnaruwa"  : {"lat": 7.9403,  "lon": 81.0188, "elev": 59,   "prov_enc": 4, "dist_enc": 20},
    "Puttalam"     : {"lat": 8.0408,  "lon": 79.8394, "elev": 2,    "prov_enc": 2, "dist_enc": 21},
    "Ratnapura"    : {"lat": 6.6828,  "lon": 80.3992, "elev": 70,   "prov_enc": 5, "dist_enc": 22},
    "Trincomalee"  : {"lat": 8.5874,  "lon": 81.2152, "elev": 6,    "prov_enc": 0, "dist_enc": 23},
    "Vavuniya"     : {"lat": 8.7514,  "lon": 80.4971, "elev": 93,   "prov_enc": 3, "dist_enc": 24},
}

# Typical weather defaults per season — used when user has no real weather data
SEASON_WEATHER = {
    'SouthwestMonsoon': {'temp': 27.5, 'precip': 6.5, 'humidity': 82},
    'NortheastMonsoon': {'temp': 26.8, 'precip': 4.2, 'humidity': 78},
    'InterMonsoon'    : {'temp': 28.2, 'precip': 2.8, 'humidity': 74},
}

SEASON_ENC  = {'SouthwestMonsoon': 2, 'NortheastMonsoon': 1, 'InterMonsoon': 0}
MONTH_NAMES = {1:"January",2:"February",3:"March",4:"April",5:"May",6:"June",
               7:"July",8:"August",9:"September",10:"October",11:"November",12:"December"}

def get_season(month):
    if month in [5,6,7,8,9]:    return 'SouthwestMonsoon'
    elif month in [10,11,12,1]: return 'NortheastMonsoon'
    else:                        return 'InterMonsoon'

def risk_label(n):
    if   n < 50:  return "🟢 Low",       "green"
    elif n < 200: return "🟡 Moderate",  "orange"
    elif n < 500: return "🟠 High",      "orangered"
    else:         return "🔴 Very High", "red"

def build_feature_row(district, year, month, temp, precip, humidity, lag1, lag2, rolling3):
    d      = DISTRICT_INFO[district]
    season = get_season(month)
    return pd.DataFrame([{
        'Year'              : year,
        'Month_sin'         : np.sin(2 * np.pi * month / 12),
        'Month_cos'         : np.cos(2 * np.pi * month / 12),
        'Latitude'          : d['lat'],
        'Longitude'         : d['lon'],
        'Elevation'         : d['elev'],
        'Temp_avg'          : temp,
        'Precipitation_avg' : precip,
        'Humidity_avg'      : humidity,
        'Province_enc'      : d['prov_enc'],
        'District_enc'      : d['dist_enc'],
        'Season_enc'        : SEASON_ENC[season],
        'Cases_lag1'        : lag1,
        'Cases_lag2'        : lag2,
        'Cases_rolling3'    : rolling3,
    }])[FEATURES]

def district_hist_avg(district, month):
    """Historical average cases for this district+month from training data."""
    d      = DISTRICT_INFO[district]
    subset = df[(df['District_enc'] == d['dist_enc']) & (df['Month'] == month)]
    return float(subset['Cases'].mean()) if len(subset) > 0 else 80.0

def add_months(year, month, n):
    """Advance (year, month) by n months."""
    total = (year - 1) * 12 + (month - 1) + n
    return (total // 12) + 1, (total % 12) + 1

st.title("🦟 Dengue Case Predictor — Sri Lanka")
st.markdown(
    "XGBoost model trained on 2019–2021 Sri Lanka district-level data. "
    "Predictions explained with **SHAP**."
)
st.divider()

# ─────────────────────────────────────────────────────────────
# MODE SELECTOR
# ─────────────────────────────────────────────────────────────
mode = st.radio(
    "**Select Prediction Mode:**",
    ["📅 Single Month Prediction", "📈 Multi-Month Forecast"],
    horizontal=True
)
st.divider()

# ═════════════════════════════════════════════════════════════
# MODE 1 — SINGLE MONTH
# ═════════════════════════════════════════════════════════════
if mode == "📅 Single Month Prediction":

    st.subheader("📅 Single Month Prediction")
    st.caption(
        "Predict dengue cases for one specific future month. "
        "Lag data (last month's cases) is **optional** — if you don't have it, "
        "the app estimates it automatically from historical district averages."
    )

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("**📍 Location & Time**")
        district = st.selectbox("District", sorted(DISTRICT_INFO.keys()), index=4)
        month    = st.slider("Month", 1, 12, date.today().month)
        year     = st.number_input("Year", min_value=2022, max_value=2035,
                                   value=date.today().year, step=1)
        season   = get_season(month)
        st.caption(f"Season: **{season}**")

    with col2:
        st.markdown("**🌡️ Weather Inputs**")
        sw       = SEASON_WEATHER[season]
        temp     = st.number_input("Avg Temperature (°C)",       15.0, 35.0, float(sw['temp']),    step=0.5)
        precip   = st.number_input("Avg Precipitation (mm/day)",  0.0, 30.0, float(sw['precip']),  step=0.5)
        humidity = st.number_input("Avg Humidity (%)",           40.0,100.0, float(sw['humidity']), step=1.0)
        st.caption("Defaults are seasonal averages — adjust if you have real forecasts.")

    with col3:
        st.markdown("**📊 Previous Cases (optional)**")
        have_lag = st.toggle("I have last month's case data", value=False)

        if have_lag:
            lag1     = float(st.number_input("Cases last month",   0, 5000, 100, step=10))
            lag2     = float(st.number_input("Cases 2 months ago", 0, 5000,  80, step=10))
            rolling3 = (lag1 + lag2) / 2.0
            st.caption(f"Rolling avg: **{rolling3:.0f}**")
        else:
            # Fall back to historical district+month averages
            prev_mo  = (month - 2) % 12 + 1   # month - 1
            prev2_mo = (month - 3) % 12 + 1   # month - 2
            lag1     = district_hist_avg(district, prev_mo)
            lag2     = district_hist_avg(district, prev2_mo)
            rolling3 = (lag1 + lag2) / 2.0
            st.info(
                f"No lag data entered — using district historical averages:\n\n"
                f"• Last month avg: **{lag1:.0f}** cases\n"
                f"• 2 months ago avg: **{lag2:.0f}** cases"
            )

    if st.button("🔍 Predict", type="primary", use_container_width=True, key="single_predict"):
        X_input     = build_feature_row(district, year, month, temp, precip, humidity,
                                        lag1, lag2, rolling3)
        prediction  = int(max(0, model.predict(X_input)[0]))
        risk, color = risk_label(prediction)

        st.divider()
        st.subheader("📈 Result")
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Predicted Cases", f"{prediction:,}")
        m2.metric("District",        district)
        m3.metric("Month",           MONTH_NAMES[month])
        m4.metric("Year",            str(int(year)))
        st.markdown(
            f"**Risk Level:** <span style='color:{color}; font-size:1.4em'>{risk}</span>",
            unsafe_allow_html=True
        )

        if not have_lag:
            st.warning(
                "⚠️ Lag features were estimated from historical averages, not real data. "
                "Treat this as an approximate forecast."
            )

        # SHAP waterfall
        st.subheader("🔬 SHAP Explanation")
        st.caption("Red = pushed prediction **up ↑**  |  Blue = pushed prediction **down ↓**")
        explainer   = shap.TreeExplainer(model)
        shap_vals   = explainer.shap_values(X_input)
        explanation = shap.Explanation(
            values=shap_vals[0], base_values=explainer.expected_value,
            data=X_input.values[0], feature_names=FEATURES
        )
        fig, _ = plt.subplots(figsize=(10, 5))
        shap.waterfall_plot(explanation, show=False)
        plt.tight_layout()
        st.pyplot(fig); plt.close()

        # SHAP table
        shap_tbl = pd.DataFrame({
            'Feature'    : FEATURES,
            'Value'      : X_input.values[0].round(2),
            'SHAP Impact': shap_vals[0].round(2),
            'Direction'  : ['↑ Increases' if v > 0 else '↓ Reduces' for v in shap_vals[0]]
        }).sort_values('SHAP Impact', key=abs, ascending=False).reset_index(drop=True)
        st.dataframe(shap_tbl, use_container_width=True)

        # Historical comparison
        d    = DISTRICT_INFO[district]
        hist = df[(df['District_enc'] == d['dist_enc']) & (df['Month'] == month)]
        if len(hist) > 0:
            st.subheader("📊 vs Historical Same Month & District")
            h1, h2, h3 = st.columns(3)
            h1.metric("Historical Average", f"{hist['Cases'].mean():.0f}")
            h2.metric("Historical Max",     f"{hist['Cases'].max():.0f}")
            h3.metric("Prediction vs Avg",  f"{prediction:,}",
                      delta=f"{prediction - hist['Cases'].mean():+.0f}")


# ═════════════════════════════════════════════════════════════
# MODE 2 — MULTI-MONTH FORECAST
# ═════════════════════════════════════════════════════════════
else:
    st.subheader("📈 Multi-Month Forecast")
    st.caption(
        "Predict dengue cases across many months. No lag data needed — "
        "each month's prediction is automatically fed into the next month as the lag value "
        "(rolling forecast). Works for any number of months into the future."
    )

    col1, col2 = st.columns([1, 2])

    with col1:
        st.markdown("**📍 Forecast Settings**")
        district        = st.selectbox("District", sorted(DISTRICT_INFO.keys()),
                                       index=4, key="fc_dist")
        start_year      = int(st.number_input("Start Year", min_value=2022, max_value=2035,
                                              value=date.today().year, step=1, key="fc_yr"))
        start_month_name = st.selectbox("Start Month", list(MONTH_NAMES.values()),
                                        index=date.today().month - 1, key="fc_mo")
        start_month     = list(MONTH_NAMES.values()).index(start_month_name) + 1
        forecast_months = st.slider("Months to forecast", 3, 36, 12)

        st.markdown("---")
        st.markdown("**🌡️ Weather Assumption**")
        weather_mode = st.radio(
            "Weather data:",
            ["Auto (use seasonal averages)", "Manual (I'll enter month by month)"],
            key="wmode"
        )

        st.markdown("---")
        st.markdown("**📌 Starting Case Count (optional)**")
        know_start = st.toggle("I know this month's current cases", value=False, key="know_start")
        seed_cases = 0.0
        if know_start:
            seed_cases = float(st.number_input("Current month cases", 0, 5000,
                                               100, step=10, key="seed"))
            st.caption("Used as the lag seed for the first forecast month.")

    with col2:
        if weather_mode == "Manual (I'll enter month by month)":
            st.markdown("**Edit expected weather for each month:**")
            rows = []
            yr, mo = start_year, start_month
            for i in range(forecast_months):
                s  = get_season(mo)
                sw = SEASON_WEATHER[s]
                rows.append({
                    'Month'        : f"{MONTH_NAMES[mo][:3]} {yr}",
                    'Temp_avg'     : sw['temp'],
                    'Precip_avg'   : sw['precip'],
                    'Humidity_avg' : sw['humidity']
                })
                yr, mo = add_months(yr, mo, 1)

            edited_weather = st.data_editor(
                pd.DataFrame(rows),
                use_container_width=True,
                num_rows="fixed",
                column_config={
                    'Month'        : st.column_config.TextColumn("Month", disabled=True),
                    'Temp_avg'     : st.column_config.NumberColumn("Temp °C",  min_value=15.0, max_value=35.0, step=0.5),
                    'Precip_avg'   : st.column_config.NumberColumn("Rain mm",  min_value=0.0,  max_value=30.0, step=0.5),
                    'Humidity_avg' : st.column_config.NumberColumn("Humidity", min_value=40.0, max_value=100.0,step=1.0),
                }
            )
        else:
            st.info(
                "**Auto mode:** weather defaults to typical seasonal averages for each month.\n\n"
                "Switch to **Manual** mode if you have weather forecasts you want to use."
            )

    # ── Run Forecast ──────────────────────────────────────────
    if st.button("📈 Run Forecast", type="primary", use_container_width=True, key="fc_btn"):

        # Seed the lag chain
        prev_mo  = (start_month - 2) % 12 + 1
        prev2_mo = (start_month - 3) % 12 + 1
        lag1_val = seed_cases if know_start else district_hist_avg(district, prev_mo)
        lag2_val = district_hist_avg(district, prev2_mo)
        roll_val = (lag1_val + lag2_val) / 2.0

        results = []
        yr, mo  = start_year, start_month

        for i in range(forecast_months):
            s  = get_season(mo)

            if weather_mode == "Auto (use seasonal averages)":
                sw     = SEASON_WEATHER[s]
                temp_v = sw['temp']; prec_v = sw['precip']; hum_v = sw['humidity']
            else:
                row    = edited_weather.iloc[i]
                temp_v = float(row['Temp_avg'])
                prec_v = float(row['Precip_avg'])
                hum_v  = float(row['Humidity_avg'])

            X_row = build_feature_row(district, yr, mo, temp_v, prec_v, hum_v,
                                      lag1_val, lag2_val, roll_val)
            pred  = float(max(0, model.predict(X_row)[0]))

            results.append({
                'Label'          : f"{MONTH_NAMES[mo][:3]} {yr}",
                'Month_Name'     : MONTH_NAMES[mo],
                'Year'           : yr,
                'Month'          : mo,
                'Season'         : s,
                'Temp_avg'       : round(temp_v, 1),
                'Precipitation'  : round(prec_v, 1),
                'Humidity'       : round(hum_v,  1),
                'Predicted_Cases': int(round(pred)),
                'Risk'           : risk_label(int(pred))[0],
            })

            # Roll forward: prediction becomes next month's lag1
            lag2_val = lag1_val
            lag1_val = pred
            roll_val = (lag1_val + lag2_val) / 2.0
            yr, mo   = add_months(yr, mo, 1)

        fc_df = pd.DataFrame(results)

        # ── Chart ─────────────────────────────────────────────
        st.divider()
        st.subheader(f"📊 {district} — {forecast_months}-Month Forecast")

        bar_colors = []
        for v in fc_df['Predicted_Cases']:
            if   v < 50:  bar_colors.append('#27ae60')
            elif v < 200: bar_colors.append('#f39c12')
            elif v < 500: bar_colors.append('#e67e22')
            else:         bar_colors.append('#e74c3c')

        fig, ax = plt.subplots(figsize=(max(10, forecast_months * 0.75), 5))
        bars = ax.bar(range(len(fc_df)), fc_df['Predicted_Cases'],
                      color=bar_colors, edgecolor='white', linewidth=0.5)

        ax.axhline(50,  color='#27ae60', linestyle='--', lw=0.9, alpha=0.5, label='Low / Moderate boundary (50)')
        ax.axhline(200, color='#f39c12', linestyle='--', lw=0.9, alpha=0.5, label='Moderate / High boundary (200)')
        ax.axhline(500, color='#e74c3c', linestyle='--', lw=0.9, alpha=0.5, label='High / Very High boundary (500)')

        ax.set_xticks(range(len(fc_df)))
        ax.set_xticklabels(fc_df['Label'], fontsize=max(6, 9 - forecast_months // 8), rotation=45, ha='right')
        ax.set_ylabel('Predicted Cases', fontsize=11)
        ax.set_title(f'Dengue Forecast — {district}  (Starting {MONTH_NAMES[start_month]} {start_year})', fontsize=13)
        ax.legend(fontsize=8, loc='upper right')

        for bar, val in zip(bars, fc_df['Predicted_Cases']):
            if val > 0:
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                        str(val), ha='center', va='bottom',
                        fontsize=max(5, 8 - forecast_months // 10))

        plt.tight_layout()
        st.pyplot(fig); plt.close()

        # ── Summary metrics ───────────────────────────────────
        peak      = fc_df.loc[fc_df['Predicted_Cases'].idxmax()]
        high_risk = (fc_df['Predicted_Cases'] >= 200).sum()

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Total Predicted Cases",  f"{fc_df['Predicted_Cases'].sum():,}")
        m2.metric("Monthly Average",        f"{fc_df['Predicted_Cases'].mean():.0f}")
        m3.metric("Peak Month",             f"{peak['Month_Name']} {peak['Year']}")
        m4.metric("Peak Predicted Cases",   f"{peak['Predicted_Cases']:,}")

        if high_risk > 0:
            st.warning(f"⚠️ **{high_risk} month(s)** reach **Moderate risk or above** (≥ 200 cases).")
        else:
            st.success("✅ All forecast months are projected in the Low risk range (< 50 cases).")

        # ── Full table ────────────────────────────────────────
        st.subheader("📋 Full Forecast Table")
        display_df = fc_df[['Month_Name','Year','Season','Temp_avg',
                             'Precipitation','Humidity','Predicted_Cases','Risk']].copy()
        display_df.columns = ['Month','Year','Season','Temp °C',
                               'Rain mm','Humidity %','Predicted Cases','Risk']
        st.dataframe(display_df, use_container_width=True, hide_index=True)

        # Download CSV
        csv = display_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            "⬇️ Download Forecast as CSV", csv,
            file_name=f"dengue_forecast_{district.replace(' ','_')}_{start_year}.csv",
            mime='text/csv'
        )

        # ── Long-range disclaimer ─────────────────────────────
        st.divider()
        if forecast_months > 12:
            st.warning(
                "⚠️ **Long-range forecast note:** Predictions beyond 12 months accumulate "
                "uncertainty because each prediction feeds as lag into the next. "
                "The further ahead you forecast, the less precise individual monthly "
                "counts become. Use for **trend direction** rather than exact numbers."
            )
        if not know_start:
            st.info(
                "ℹ️ The forecast was seeded with historical district averages as starting lag values. "
                "For a more accurate first few months, enable **'I know this month's current cases'**."
            )
        st.caption("Model: XGBoost | Trained on 2019–2021 Sri Lanka data | Educational use only.")

with st.sidebar:
    st.header("ℹ️ About")
    st.markdown("""
**Algorithm:** XGBoost Regressor

**Prediction Modes:**

📅 **Single Month**
Predict one specific future month.
Toggle lag data on/off — if off, historical district averages are used automatically.

📈 **Multi-Month Forecast**
Predict 3–36 months ahead.
No real lag data needed — each month's prediction feeds into the next automatically (rolling forecast).

---
**Features used:**
- Geographic: lat/lon, elevation, district, province
- Temporal: year, month (sin/cos), season
- Weather: temperature, precipitation, humidity
- Lag: last 2 months + rolling avg

**XAI:** SHAP (SHapley Additive exPlanations)

**Training:** 2019–2020 | **Test:** 2021
    """)

    st.header("🗓️ Season Reference")
    st.markdown("""
| Season | Months |
|--------|--------|
| SW Monsoon | May–Sep |
| NE Monsoon | Oct–Jan |
| Inter-Monsoon | Feb–Apr |
    """)

    st.header("🎨 Risk Levels")
    st.markdown("""
| | Level | Cases |
|-|-------|-------|
| 🟢 | Low | < 50 |
| 🟡 | Moderate | 50–199 |
| 🟠 | High | 200–499 |
| 🔴 | Very High | ≥ 500 |
    """)