from __future__ import annotations

import io
import re
import zipfile
import base64
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import altair as alt
import numpy as np
import pandas as pd
import requests
import statsmodels.api as sm
import streamlit as st
import yfinance as yf


st.set_page_config(
    page_title="Factor Lens | Investing in Evidence",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

cover_b64 = base64.b64encode(Path("assets/cover.png").read_bytes()).decode()
st.markdown(
    f"""
    <style>
    :root {{ --charcoal:#202020; --ink:#2a2a2a; --red:#c83a30; --paper:#f5f4f1; --silver:#d6d6d6; }}
    .stApp {{ background:
      linear-gradient(rgba(16,16,16,.28), rgba(16,16,16,.38)),
      url("data:image/png;base64,{cover_b64}") center center / cover fixed;
      color:var(--ink); }}
    [data-testid="stMainBlockContainer"] {{
      background:rgba(245,244,241,.94); border:1px solid rgba(255,255,255,.35);
      border-radius:18px; padding:2.1rem 2.5rem 2.5rem;
      margin-top:2rem; margin-bottom:2rem;
      box-shadow:0 18px 55px rgba(0,0,0,.32);
      backdrop-filter:blur(4px); -webkit-backdrop-filter:blur(4px);
    }}
    [data-testid="stSidebar"] {{ background:#202020; border-right:3px solid #c83a30; }}
    [data-testid="stSidebar"] * {{ color:#f5f4f1; }}
    [data-testid="stSidebar"] input, [data-testid="stSidebar"] textarea {{ color:#202020; }}
    [data-testid="stSidebar"] img {{ border-radius:12px; border:1px solid #555; }}
    .hero {{ position:relative; overflow:hidden; padding:2.05rem 2.3rem;
      border-radius:16px; color:white;
      background:linear-gradient(110deg,rgba(24,24,24,.82),rgba(45,45,45,.58)),
      url("data:image/png;base64,{cover_b64}") center 45% / cover;
      margin-bottom:1.5rem; border:1px solid rgba(255,255,255,.2);
      box-shadow:0 8px 24px rgba(0,0,0,.16); }}
    .hero::after {{ content:""; position:absolute; left:0; bottom:0; width:100%;
      height:3px; background:linear-gradient(90deg,#c83a30 0%,#c83a30 34%,rgba(200,58,48,0) 82%); }}
    .hero h1 {{ position:relative; z-index:1; margin:0 0 .4rem;
      font-size:2.25rem; letter-spacing:.015em; font-weight:700;
      text-shadow:0 2px 12px rgba(0,0,0,.35); }}
    .hero p {{ position:relative; z-index:1; margin:0; opacity:.9;
      font-size:1.03rem; line-height:1.5; max-width:720px; }}
    div[data-testid="stMetric"] {{ background:rgba(255,255,255,.94); border:1px solid #d6d6d6;
      padding:1rem; border-radius:10px; box-shadow:0 3px 12px rgba(0,0,0,.06); }}
    .note {{ background:#f0eeee; border-left:4px solid #c83a30; padding:.8rem 1rem;
      border-radius:0 8px 8px 0; color:#333; }}
    .stButton > button, [data-testid="stFormSubmitButton"] button {{
      width:100%; background:#e7e3de !important; color:#171717 !important;
      border:1px solid #aaa49d !important; font-weight:800; border-radius:10px;
      padding:.72rem .85rem; box-shadow:0 4px 14px rgba(0,0,0,.16);
      transition:transform .18s ease, box-shadow .18s ease, background .18s ease; }}
    .stButton > button p, [data-testid="stFormSubmitButton"] button p {{
      color:#171717 !important; font-weight:800 !important; }}
    .stButton > button:hover, [data-testid="stFormSubmitButton"] button:hover {{
      transform:translateY(-1px); background:#f7f5f2 !important;
      border-color:#c83a30 !important; color:#171717 !important;
      box-shadow:0 7px 18px rgba(0,0,0,.2); }}
    .stButton > button:focus, [data-testid="stFormSubmitButton"] button:focus {{
      color:#171717 !important; border-color:#c83a30 !important;
      box-shadow:0 0 0 2px rgba(200,58,48,.25); }}
    [data-testid="stTabs"] button[aria-selected="true"] {{ color:#c83a30; }}
        [data-testid="stDataFrame"], [data-testid="stTable"] {{ width:100%; }}
        [data-testid="stDataFrame"] thead th,
        [data-testid="stDataFrame"] thead tr th,
        [data-testid="stTable"] thead th,
        [data-testid="stTable"] thead tr th,
        [data-testid="stDataFrame"] [role="columnheader"],
        [data-testid="stDataFrame"] [role="rowheader"],
        [data-testid="stDataFrame"] [role="gridcell"],
        [data-testid="stTable"] [role="columnheader"],
        [data-testid="stTable"] [role="rowheader"],
        [data-testid="stTable"] [role="gridcell"] {{
            background-color:#3a3a3a !important;
            color:#000000 !important;
            font-weight:700;
            text-align:center !important;
            vertical-align:middle !important;
        }}
        [data-testid="stDataFrame"] tbody td,
        [data-testid="stTable"] tbody td {{
            color:#111111;
            text-align:center !important;
            vertical-align:middle !important;
        }}
    h1, h2, h3 {{ color:#202020; letter-spacing:.01em; }}
    .hero h1 {{ color:white; }}
    @media (max-width: 768px) {{
      [data-testid="stMainBlockContainer"] {{ margin-top:.5rem; padding:1.25rem; border-radius:12px; }}
      .hero {{ padding:1.5rem; }}
    }}
    </style>
    """,
    unsafe_allow_html=True,
)


MODEL_FACTORS = {
    "CAPM": ["MKT"],
    "Fama-French 3-factor": ["MKT", "SMB", "HML"],
    "Carhart 4-factor": ["MKT", "SMB", "HML", "MOM"],
    "Fama-French 5-factor": ["MKT", "SMB", "HML", "RMW", "CMA"],
    "Fama-French 5-factor + momentum": ["MKT", "SMB", "HML", "RMW", "CMA", "MOM"],
}

EUROPE_SUFFIXES = (".L", ".AS", ".BR", ".CO", ".DE", ".F", ".HE", ".IR", ".LS", ".MC", ".MI", ".OL", ".PA", ".ST", ".SW", ".VI")


@dataclass
class Analysis:
    ticker: str
    fund_name: str
    region: str
    model_name: str
    data: pd.DataFrame
    result: object
    prices: pd.Series
    factors: list[str]

    @property
    def display_name(self) -> str:
        return f"{self.fund_name} ({self.ticker})" if self.fund_name else self.ticker


def infer_region(ticker: str) -> str:
    return "Europe" if ticker.upper().endswith(EUROPE_SUFFIXES) else "United States"


@st.cache_data(ttl=24 * 60 * 60, show_spinner=False)
def get_fund_name(ticker: str) -> str:
    """Return Yahoo's preferred security name without making analysis depend on it."""
    try:
        info = yf.Ticker(ticker).get_info()
        return str(info.get("longName") or info.get("shortName") or "").strip()
    except Exception:
        return ""


def _dataset_filename(region: str, kind: str) -> str:
    if region == "United States":
        return {
            "3": "F-F_Research_Data_Factors_CSV.zip",
            "5": "F-F_Research_Data_5_Factors_2x3_CSV.zip",
            "mom": "F-F_Momentum_Factor_CSV.zip",
        }[kind]
    return {
        "3": "Europe_3_Factors_CSV.zip",
        "5": "Europe_5_Factors_CSV.zip",
        "mom": "Europe_Mom_Factor_CSV.zip",
    }[kind]


def _parse_french_monthly(raw: str) -> pd.DataFrame:
    lines = raw.splitlines()
    header_idx = next(
        (
            i
            for i, line in enumerate(lines)
            if line.strip().startswith(",")
            and i + 1 < len(lines)
            and re.match(r"^\s*\d{6}\s*,", lines[i + 1])
        ),
        None,
    )
    if header_idx is None:
        raise ValueError("The factor file did not contain a recognised monthly header.")
    header = "Date" + lines[header_idx].strip()
    rows: list[str] = []
    for line in lines[header_idx + 1 :]:
        if re.match(r"^\s*\d{6}\s*,", line):
            rows.append(line.strip())
        elif rows:
            break
    if not rows:
        raise ValueError("No monthly observations were found in the factor file.")
    frame = pd.read_csv(io.StringIO(header + "\n" + "\n".join(rows)))
    frame.columns = [str(c).strip() for c in frame.columns]
    frame["Date"] = pd.to_datetime(frame["Date"].astype(str), format="%Y%m").dt.to_period("M")
    frame = frame.set_index("Date")
    rename = {"Mkt-RF": "MKT", "Mom": "MOM", "MOM": "MOM", "WML": "MOM"}
    frame = frame.rename(columns=rename)
    for col in frame.columns:
        frame[col] = pd.to_numeric(frame[col], errors="coerce") / 100.0
    return frame


@st.cache_data(ttl=24 * 60 * 60, show_spinner=False)
def download_french_dataset(region: str, kind: str) -> pd.DataFrame:
    filename = _dataset_filename(region, kind)
    url = f"https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/{filename}"
    response = requests.get(url, timeout=30, headers={"User-Agent": "InvestingInEvidence/1.0"})
    response.raise_for_status()
    with zipfile.ZipFile(io.BytesIO(response.content)) as archive:
        csv_name = next(name for name in archive.namelist() if name.lower().endswith((".csv", ".txt")))
        raw = archive.read(csv_name).decode("latin-1")
    return _parse_french_monthly(raw)


def get_factors(region: str, required: list[str]) -> pd.DataFrame:
    kind = "5" if any(f in required for f in ("RMW", "CMA")) else "3"
    factors = download_french_dataset(region, kind)
    if "MOM" in required:
        momentum = download_french_dataset(region, "mom")
        factors = factors.join(momentum[["MOM"]], how="inner")
    needed = list(dict.fromkeys(required + ["RF"]))
    missing = [column for column in needed if column not in factors]
    if missing:
        raise ValueError(f"Factor dataset is missing: {', '.join(missing)}")
    return factors[needed]


@st.cache_data(ttl=60 * 60, show_spinner=False)
def download_monthly_prices(ticker: str, start: date, end: date) -> pd.Series:
    # Daily adjusted closes are resampled to month-end. This avoids inconsistent
    # Yahoo monthly interval labelling and gives an exact date-window treatment.
    fetch_start = (pd.Timestamp(start) - pd.offsets.MonthEnd(1)).date()
    fetch_end = (pd.Timestamp(end) + pd.Timedelta(days=1)).date()
    raw = yf.download(ticker, start=fetch_start, end=fetch_end, auto_adjust=True, progress=False, threads=False)
    if raw is None or raw.empty:
        raise ValueError(f"Yahoo Finance returned no price data for ‘{ticker}’.")
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)
    if "Close" not in raw:
        raise ValueError("Yahoo Finance did not return an adjusted close column.")
    close = pd.to_numeric(raw["Close"], errors="coerce").dropna()
    close.index = pd.to_datetime(close.index).tz_localize(None)
    monthly = close.resample("ME").last()
    monthly.name = "Adjusted close"
    # Keep the true last trading date: resampling labels the current partial month
    # with its future calendar month-end, which is not a valid date-picker value.
    monthly.attrs["last_observation_date"] = close.index.max().date()
    return monthly


def run_analysis(ticker: str, start: date, end: date, model_name: str, region_choice: str) -> Analysis:
    ticker = ticker.strip().upper()
    if not ticker:
        raise ValueError("Enter a Yahoo Finance ticker.")
    if start >= end:
        raise ValueError("The end date must be later than the start date.")
    region = infer_region(ticker) if region_choice == "Auto-detect" else region_choice
    factors_used = MODEL_FACTORS[model_name]
    prices = download_monthly_prices(ticker, start, end)
    returns = prices.pct_change().dropna()
    returns.index = returns.index.to_period("M")
    returns.name = "Asset return"
    factors = get_factors(region, factors_used)
    combined = pd.concat([returns, factors], axis=1, join="inner").dropna()
    start_period, end_period = pd.Period(start, "M"), pd.Period(end, "M")
    combined = combined.loc[(combined.index >= start_period) & (combined.index <= end_period)].copy()
    minimum = len(factors_used) + 8
    if len(combined) < minimum:
        raise ValueError(f"Only {len(combined)} overlapping monthly observations are available; at least {minimum} are required.")
    combined["Excess return"] = combined["Asset return"] - combined["RF"]
    result = sm.OLS(combined["Excess return"], sm.add_constant(combined[factors_used])).fit(cov_type="HC1")
    return Analysis(
        ticker, get_fund_name(ticker), region, model_name, combined, result, prices, factors_used
    )


def available_analysis_dates(
    tickers: list[str], model_name: str, region_choice: str
) -> tuple[date, date]:
    """Find the earliest usable regression month and latest common price date."""
    starts: list[pd.Period] = []
    price_ends: list[date] = []
    factors_used = MODEL_FACTORS[model_name]
    for ticker in tickers:
        region = infer_region(ticker) if region_choice == "Auto-detect" else region_choice
        prices = download_monthly_prices(ticker, date(1900, 1, 1), date.today())
        price_ends.append(prices.attrs["last_observation_date"])
        return_periods = prices.pct_change().dropna().index.to_period("M")
        factors = get_factors(region, factors_used)
        usable_periods = return_periods.intersection(factors.index)
        if usable_periods.empty:
            raise ValueError(f"No overlapping price and factor data are available for {ticker}.")
        starts.append(usable_periods.min())
    common_start = max(starts)
    common_end = min(price_ends)
    if common_start.start_time.date() > common_end:
        raise ValueError("The selected tickers do not share a common analysis period.")
    return common_start.start_time.date(), common_end


def coefficient_table(analysis: Analysis) -> pd.DataFrame:
    result = analysis.result
    names = ["Alpha"] + analysis.factors
    index = ["const"] + analysis.factors
    return pd.DataFrame({
        "Term": names,
        "Coefficient": [result.params[i] for i in index],
        "Robust SE": [result.bse[i] for i in index],
        "t-statistic": [result.tvalues[i] for i in index],
        "p-value": [result.pvalues[i] for i in index],
        "95% CI lower bound": [result.conf_int().loc[i, 0] for i in index],
        "95% CI upper bound": [result.conf_int().loc[i, 1] for i in index],
    })


def annualised_alpha(analysis: Analysis) -> float:
    alpha = float(analysis.result.params["const"])
    return (1 + alpha) ** 12 - 1 if alpha > -1 else np.nan


def performance_statistics(analysis: Analysis) -> dict[str, float]:
    """Calculate investor-focused return and risk measures from monthly returns."""
    returns = analysis.data["Asset return"].dropna()
    wealth = (1 + returns).cumprod()
    years = len(returns) / 12
    total_return = float(wealth.iloc[-1] - 1)
    annual_return = float(wealth.iloc[-1] ** (1 / years) - 1) if years > 0 else np.nan
    annual_volatility = float(returns.std(ddof=1) * np.sqrt(12))
    annual_rf = float(analysis.data.loc[returns.index, "RF"].mean() * 12)
    sharpe = (
        (annual_return - annual_rf) / annual_volatility
        if annual_volatility > 0 else np.nan
    )
    downside = returns[returns < 0].std(ddof=1) * np.sqrt(12)
    sortino = (annual_return - annual_rf) / downside if downside > 0 else np.nan
    drawdown = wealth / wealth.cummax() - 1
    max_drawdown = float(drawdown.min())
    calmar = annual_return / abs(max_drawdown) if max_drawdown < 0 else np.nan
    return {
        "Total return": total_return,
        "Annualised return": annual_return,
        "Annualised volatility": annual_volatility,
        "Sharpe ratio": sharpe,
        "Sortino ratio": sortino,
        "Maximum drawdown": max_drawdown,
        "Calmar ratio": calmar,
        "Best month": float(returns.max()),
        "Worst month": float(returns.min()),
        "Positive months": float((returns > 0).mean()),
        "Monthly VaR (95%)": float(returns.quantile(.05)),
    }


@st.cache_data(ttl=6 * 60 * 60, show_spinner=False)
def get_fund_info(ticker: str) -> dict[str, object]:
    try:
        return yf.Ticker(ticker).get_info() or {}
    except Exception:
        return {}


def fund_profile_summary(info: dict[str, object]) -> list[tuple[str, str]]:
    details: list[tuple[str, str]] = []
    for label, key in [
        ("Category", "category"),
        ("Fund family", "fundFamily"),
        ("Yield", "yield"),
    ]:
        value = info.get(key)
        if value in (None, "", "nan"):
            continue
        if isinstance(value, float):
            if label == "Yield":
                details.append((label, f"{value:.2%}"))
            else:
                details.append((label, f"{value:.2f}"))
        else:
            details.append((label, str(value)))

    return details


def render_performance(analysis: Analysis) -> None:
    returns = analysis.data["Asset return"].dropna()
    dates = returns.index.to_timestamp("M")
    wealth = (1 + returns).cumprod()
    drawdown = wealth / wealth.cummax() - 1
    stats = performance_statistics(analysis)

    first_row = st.columns(4)
    first_row[0].metric("Total return", f"{stats['Total return']:.2%}")
    first_row[1].metric("Annualised return", f"{stats['Annualised return']:.2%}")
    first_row[2].metric("Annualised volatility", f"{stats['Annualised volatility']:.2%}")
    first_row[3].metric("Maximum drawdown", f"{stats['Maximum drawdown']:.2%}")
    second_row = st.columns(4)
    second_row[0].metric("Sharpe ratio", f"{stats['Sharpe ratio']:.2f}")
    second_row[1].metric("Sortino ratio", f"{stats['Sortino ratio']:.2f}")
    second_row[2].metric("Best / worst month", f"{stats['Best month']:.2%} / {stats['Worst month']:.2%}")
    second_row[3].metric("Positive months", f"{stats['Positive months']:.1%}")

    chart_data = pd.DataFrame({
        "Date": dates,
        "Growth of 100": wealth.to_numpy() * 100,
        "Drawdown": drawdown.to_numpy(),
    })
    growth_chart = alt.Chart(chart_data).mark_line(color="#c83a30", strokeWidth=2.5).encode(
        x=alt.X("Date:T", title=None),
        y=alt.Y("Growth of 100:Q", title="Growth of 100", scale=alt.Scale(zero=False)),
        tooltip=[alt.Tooltip("Date:T", format="%b %Y"), alt.Tooltip("Growth of 100:Q", format=".2f")],
    ).properties(height=330, padding={"top": 20, "right": 20, "bottom": 20, "left": 20})
    st.altair_chart(growth_chart, use_container_width=True)

    drawdown_chart = alt.Chart(chart_data).mark_area(color="#c83a30", opacity=.7).encode(
        x=alt.X("Date:T", title=None),
        y=alt.Y("Drawdown:Q", title="Drawdown", axis=alt.Axis(format="%")),
        tooltip=[alt.Tooltip("Date:T", format="%b %Y"), alt.Tooltip("Drawdown:Q", format=".2%")],
    ).properties(height=220, padding={"top": 20, "right": 20, "bottom": 20, "left": 20})
    st.altair_chart(drawdown_chart, use_container_width=True)

    annual = returns.groupby(returns.index.year).apply(lambda values: (1 + values).prod() - 1)
    rolling = pd.DataFrame(index=returns.index)
    rolling["12-month return"] = (1 + returns).rolling(12).apply(np.prod, raw=True) - 1
    rolling["36-month annualised return"] = (
        (1 + returns).rolling(36).apply(np.prod, raw=True) ** (1 / 3) - 1
    )
    detail_left, detail_right = st.columns(2)
    with detail_left:
        st.markdown("#### Calendar-year returns")
        annual_table = annual.rename("Return").rename_axis("Year").reset_index()
        st.dataframe(annual_table.style.format({"Return": "{:.2%}"}), hide_index=True, use_container_width=True)
    with detail_right:
        st.markdown("#### Additional risk statistics")
        risk_table = pd.DataFrame({
            "Measure": ["Calmar ratio", "Monthly VaR (95%)"],
            "Value": [f"{stats['Calmar ratio']:.2f}", f"{stats['Monthly VaR (95%)']:.2%}"],
        })
        st.dataframe(risk_table, hide_index=True, use_container_width=True)
    st.markdown("#### Rolling returns")
    rolling_export = rolling.dropna(how="all").copy()
    rolling_export.index = rolling_export.index.astype(str)
    st.dataframe(rolling_export.style.format("{:.2%}"), use_container_width=True)
    st.caption("Returns use adjusted prices and include reinvested distributions where supplied by Yahoo Finance. Risk-free rates are from the selected French dataset.")


def render_analysis(analysis: Analysis) -> None:
    result = analysis.result
    st.subheader(f"{analysis.display_name} | {analysis.model_name}")
    st.caption(
        f"{analysis.region} factors | {analysis.data.index.min()} to "
        f"{analysis.data.index.max()} | HC1 robust standard errors"
    )
    m1, m2, m3 = st.columns(3)
    m1.metric("R²", f"{result.rsquared:.3f}")
    m2.metric("Adjusted R²", f"{result.rsquared_adj:.3f}")
    m3.metric("Observations", f"{int(result.nobs)} months")

    table = coefficient_table(analysis)
    plot_data = table.copy()
    plot_data["Significant"] = np.where(
        plot_data["p-value"] < .05, "p < 0.05", "Not significant"
    )
    chart = alt.Chart(plot_data).mark_bar(
        cornerRadiusTopLeft=4, cornerRadiusTopRight=4
    ).encode(
        x=alt.X(
            "Term:N",
            sort=None,
            title=None,
            axis=alt.Axis(labelAngle=0),
        ),
        y=alt.Y("Coefficient:Q", title="Monthly alpha / factor loading"),
        color=alt.Color(
            "Significant:N",
            scale=alt.Scale(
                domain=["p < 0.05", "Not significant"],
                range=["#c83a30", "#a8a8a8"],
            ),
            legend=alt.Legend(title=None),
        ),
        tooltip=[
            "Term",
            alt.Tooltip("Coefficient:Q", format=".4f"),
            alt.Tooltip("p-value:Q", format=".4f"),
        ],
    ).properties(height=360, padding={"top": 20, "right": 20, "bottom": 20, "left": 20})

    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(
        ["Factor Exposures", "Regression Table", "Info", "Performance", "Monthly Data", "Methodology"]
    )
    with tab1:
        st.altair_chart(chart, use_container_width=True)
        st.caption("Alpha is the monthly intercept. Factor coefficients are loadings, not percentages.")
    with tab2:
        number_columns = [column for column in table.columns if column != "Term"]
        formatted = table.set_index("Term").style.format(
            {column: "{:.4f}" for column in number_columns}
        )
        st.dataframe(formatted, use_container_width=True)
        st.caption("95% CI lower and upper bounds show the range of values compatible with the estimate at 95% confidence.")
        st.download_button(
            "Download regression results (CSV)",
            table.to_csv(index=False),
            f"{analysis.ticker}_regression.csv",
            "text/csv",
            key=f"results_{analysis.ticker}",
        )
    with tab3:
        info = get_fund_info(analysis.ticker)
        details = fund_profile_summary(info)

        if details:
            detail_items = "\n".join(f"- **{label}:** {value}" for label, value in details)
            st.markdown(detail_items)
        else:
            st.info("Category, fund family, and yield are not available from Yahoo Finance for this ticker.")
    with tab4:
        render_performance(analysis)
    with tab5:
        export = analysis.data.copy()
        export.index = export.index.astype(str)
        st.dataframe(export.style.format("{:.4%}"), use_container_width=True)
        st.download_button(
            "Download aligned monthly data (CSV)",
            export.to_csv(),
            f"{analysis.ticker}_monthly_data.csv",
            "text/csv",
            key=f"data_{analysis.ticker}",
        )
    with tab6:
        st.write(
            f"The dependent variable is {analysis.display_name}'s monthly adjusted-price "
            f"return minus the monthly risk-free rate. Explanatory variables are "
            f"{', '.join(analysis.factors)}. The model includes an intercept and uses "
            "HC1 heteroskedasticity-robust standard errors."
        )
        st.warning(
            "Educational information only, not investment advice. Results are historical "
            "estimates and may be sensitive to the period, market region, data quality "
            "and model choice."
        )


st.markdown(
    """<div class="hero"><h1>Factor Lens</h1>
    <p>This tool aims to get under the hood of funds via a factor regression to give investors greater insight and ability to make better investment decisions.</p></div>""",
    unsafe_allow_html=True,
)


if "selected_start_date" not in st.session_state:
    st.session_state.selected_start_date = date(2018, 1, 1)
if "selected_end_date" not in st.session_state:
    st.session_state.selected_end_date = date.today()


def use_custom_start_date() -> None:
    st.session_state.use_earliest = False


def use_custom_end_date() -> None:
    st.session_state.use_latest = False


with st.sidebar:
    st.image("assets/logo.jpg", use_container_width=True)
    st.header("Run an Analysis")
    st.markdown("**Enter Yahoo Finance Tickers Below:**")
    ticker_text = st.text_area(
        "Yahoo Finance Tickers",
        value="VWRL.L",
        height=92,
        placeholder="VWRL.L, SPY, AAPL",
        help="You can separate tickers with commas, spaces or new lines.",
    )
    st.caption(
        "Enter up to five tickers. Commas are recommended, for example: "
        "VWRL.L, SPY, AAPL. Spaces and separate lines also work."
    )
    model_name = st.selectbox("Regression Model", list(MODEL_FACTORS))
    region_choice = st.selectbox(
        "Factor Region",
        ["Auto-detect", "Europe", "United States"],
        help="Auto-detect uses Europe for common European exchange suffixes and the United States otherwise.",
    )
    use_earliest = st.checkbox(
        "Use earliest available start date",
        key="use_earliest",
        help="The analysis will begin at the first date shared by the price and factor data.",
    )
    use_latest = st.checkbox(
        "Use latest available end date",
        value=True,
        key="use_latest",
        help=(
            "The end date will follow the most recent price observation shared by the "
            "selected tickers. Regression results use the latest month for which matching "
            "factor data are also available."
        ),
    )
    availability_error = None
    selected_tickers = list(dict.fromkeys(
        ticker.upper() for ticker in re.split(r"[,\s;]+", ticker_text.strip()) if ticker
    ))
    if (use_earliest or use_latest) and selected_tickers:
        try:
            with st.spinner("Finding available dates..."):
                earliest_date, latest_date = available_analysis_dates(
                    selected_tickers, model_name, region_choice
                )
            if use_earliest:
                st.session_state.selected_start_date = earliest_date
            if use_latest:
                st.session_state.selected_end_date = latest_date
        except Exception as exc:
            availability_error = str(exc)
    selected_start_date = st.date_input(
        "Start Date",
        min_value=date(1900, 1, 1),
        max_value=date.today(),
        key="selected_start_date",
        on_change=use_custom_start_date,
    )
    selected_end_date = st.date_input(
        "End Date",
        min_value=date(1900, 1, 1),
        max_value=date.today(),
        key="selected_end_date",
        on_change=use_custom_end_date,
    )
    start_date = selected_start_date
    end_date = selected_end_date
    if availability_error:
        st.warning(f"Available dates could not be determined: {availability_error}")
    submitted = st.button(
        "Run Regressions",
        use_container_width=True,
        type="primary",
        disabled=availability_error is not None,
    )
    st.caption("Prices: Yahoo Finance | Factors: Kenneth R. French Data Library")

if not submitted:
    left, right = st.columns([1.4, 1])
    with left:
        st.subheader("A Clearer View of Investment Returns")
        st.write(
            "Choose up to five tickers, a period and a model. The app downloads adjusted "
            "prices and matching monthly factors, aligns the data, and estimates each "
            "regression using heteroskedasticity-robust standard errors."
        )
        st.markdown(
            '<div class="note">Start with at least three years of monthly data. Longer periods generally produce more stable estimates.</div>',
            unsafe_allow_html=True,
        )
    with right:
        st.subheader("What You Will See")
        st.write(
            "• Side-by-side comparison of model fit and exposures\n\n"
            "• Factor exposures with confidence intervals and significance\n\n"
            "• Performance summary and rolling returns\n\n"
            "• The exact aligned monthly datasets used for each regression"
        )
    st.stop()

tickers = list(dict.fromkeys(
    ticker.upper() for ticker in re.split(r"[,\s;]+", ticker_text.strip()) if ticker
))
if not tickers:
    st.error("Enter at least one Yahoo Finance ticker.")
    st.stop()
if len(tickers) > 5:
    st.error("You can analyse a maximum of five tickers at a time.")
    st.stop()

analyses: list[Analysis] = []
errors: dict[str, str] = {}
progress = st.progress(0, text="Preparing the analysis...")
for position, ticker in enumerate(tickers, start=1):
    progress.progress(
        (position - 1) / len(tickers), text=f"Analysing {ticker} ({position} of {len(tickers)})..."
    )
    try:
        analyses.append(
            run_analysis(ticker, start_date, end_date, model_name, region_choice)
        )
    except Exception as exc:
        errors[ticker] = str(exc)
progress.progress(1.0, text="Analysis complete")
progress.empty()

for ticker, message in errors.items():
    st.warning(f"{ticker} could not be analysed: {message}")
if not analyses:
    st.error("None of the requested analyses could be completed. Check the tickers, date range and factor region.")
    st.stop()

if len(analyses) > 1:
    st.subheader("Comparison")
    comparison = pd.DataFrame(
        {
            "Fund": analysis.display_name,
            "Region": analysis.region,
            "R²": analysis.result.rsquared,
            "Adjusted R²": analysis.result.rsquared_adj,
            "Observations": int(analysis.result.nobs),
        }
        for analysis in analyses
    )
    st.dataframe(
        comparison.style.format({"R²": "{:.3f}", "Adjusted R²": "{:.3f}"}),
        use_container_width=True,
        hide_index=True,
    )
    st.divider()

result_tabs = st.tabs([analysis.display_name for analysis in analyses])
for tab, analysis in zip(result_tabs, analyses):
    with tab:
        render_analysis(analysis)

st.caption("Investing in Evidence | Transparent analysis for better-informed decisions")
