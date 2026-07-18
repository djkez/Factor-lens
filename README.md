# Factor Lens

A public Streamlit application for monthly Fama-French factor regressions. It downloads adjusted price data from Yahoo Finance and factor data directly from the Kenneth French Data Library. Up to five tickers can be compared in one run.

## Run locally

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
streamlit run app.py
```

The app supports CAPM, Fama-French 3-factor, Carhart 4-factor, Fama-French 5-factor, and 5-factor plus momentum models. European Yahoo suffixes (including `.L`) automatically select European factors; all other tickers default to US factors. Users can override this choice.

## Deploy

Push this directory to a Git repository and deploy `app.py` on Streamlit Community Cloud or another Python host. No API keys are required.

## Important limitations

- Yahoo Finance is an unofficial data source and may change or omit observations.
- Kenneth French factors are regional portfolio factors, not ticker-specific benchmarks.
- Results are historical estimates for educational use, not investment advice.
