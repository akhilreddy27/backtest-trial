import streamlit as st
import yfinance as yf
import pandas as pd
import datetime
import numpy as np

st.set_page_config(page_title="Multi-Strategy Backtest", layout="wide")
st.title("📈 Multi-Strategy Backtest: SMA vs MACD vs Buy & Hold")

# --------- User Inputs ---------
symbol = st.text_input("Stock Symbol", "AAPL")
start_date = st.date_input("Start Date", datetime.date(2020, 1, 1))
end_date = datetime.date.today()
sma_short = st.number_input("SMA Short", 1, 200, 20)
sma_long = st.number_input("SMA Long", 1, 500, 50)

selected_strategies = st.multiselect(
    "📊 Select Strategies to Backtest",
    ["SMA Crossover", "MACD Crossover", "Buy & Hold"],
    default=["SMA Crossover", "MACD Crossover", "Buy & Hold"]
)

run_backtest = st.button("🚀 Run Backtest")

# --------- Download Data ---------
data = yf.download(symbol, start=start_date, end=end_date)
if isinstance(data.columns, pd.MultiIndex):
    data.columns = ['_'.join(col).strip() for col in data.columns.values]
close_col = next((col for col in data.columns if 'close' in col.lower()), None)

# --------- Backtest Function ---------
def run_strategy(df, strategy_name):
    df = df.copy()
    cash = 100000
    shares = 0
    trade_log = []
    portfolio_value = []

    if strategy_name == "SMA Crossover":
        df['SMA_Short'] = df[close_col].rolling(sma_short).mean()
        df['SMA_Long'] = df[close_col].rolling(sma_long).mean()

        for i in range(1, len(df)):
            today = df.index[i]
            price = df[close_col].iloc[i]
            s_prev, l_prev = df['SMA_Short'].iloc[i - 1], df['SMA_Long'].iloc[i - 1]
            s_curr, l_curr = df['SMA_Short'].iloc[i], df['SMA_Long'].iloc[i]

            if pd.notna(s_prev) and pd.notna(l_prev):
                if s_prev < l_prev and s_curr > l_curr and shares == 0:
                    shares = int(cash // price)
                    cash -= shares * price
                    trade_log.append({"type": "BUY", "date": today, "price": price, "shares": shares})
                elif s_prev > l_prev and s_curr < l_curr and shares > 0:
                    cash += shares * price
                    trade_log.append({"type": "SELL", "date": today, "price": price, "shares": shares})
                    shares = 0
            portfolio_value.append({"date": today, "portfolio": cash + shares * price})

    elif strategy_name == "MACD Crossover":
        ema12 = df[close_col].ewm(span=12).mean()
        ema26 = df[close_col].ewm(span=26).mean()
        df["MACD"] = ema12 - ema26
        df["Signal"] = df["MACD"].ewm(span=9).mean()

        for i in range(1, len(df)):
            today = df.index[i]
            price = df[close_col].iloc[i]
            m_prev, s_prev = df["MACD"].iloc[i - 1], df["Signal"].iloc[i - 1]
            m_curr, s_curr = df["MACD"].iloc[i], df["Signal"].iloc[i]

            if pd.notna(m_prev) and pd.notna(s_prev):
                if m_prev < s_prev and m_curr > s_curr and shares == 0:
                    shares = int(cash // price)
                    cash -= shares * price
                    trade_log.append({"type": "BUY", "date": today, "price": price, "shares": shares})
                elif m_prev > s_prev and m_curr < s_curr and shares > 0:
                    cash += shares * price
                    trade_log.append({"type": "SELL", "date": today, "price": price, "shares": shares})
                    shares = 0
            portfolio_value.append({"date": today, "portfolio": cash + shares * price})

    elif strategy_name == "Buy & Hold":
        entry_price = df[close_col].iloc[sma_long]
        shares = int(cash // entry_price)
        cash -= shares * entry_price
        for i in range(sma_long, len(df)):
            today = df.index[i]
            price = df[close_col].iloc[i]
            portfolio_value.append({"date": today, "portfolio": cash + shares * price})
        trade_log.append({"type": "BUY", "date": df.index[sma_long], "price": entry_price, "shares": shares})

    # Create portfolio DataFrame
    pf_df = pd.DataFrame(portfolio_value).set_index("date")
    pf_df['daily_return'] = pf_df['portfolio'].pct_change().fillna(0)

    # Metrics
    final_value = pf_df['portfolio'].iloc[-1]
    start_value = pf_df['portfolio'].iloc[0]
    total_return = (final_value - start_value) / start_value
    days = (pf_df.index[-1] - pf_df.index[0]).days
    annualized_return = (1 + total_return) ** (365 / days) - 1 if days > 0 else 0
    max_dd = ((pf_df['portfolio'].cummax() - pf_df['portfolio']) / pf_df['portfolio'].cummax()).max()
    sharpe = np.mean(pf_df['daily_return']) / np.std(pf_df['daily_return']) * np.sqrt(252) if np.std(pf_df['daily_return']) > 0 else 0

    return {
        "name": strategy_name,
        "final_value": final_value,
        "total_return": total_return,
        "annualized": annualized_return,
        "max_dd": max_dd,
        "sharpe": sharpe,
        "portfolio": pf_df['portfolio'],
        "trades": pd.DataFrame(trade_log)
    }

# --------- Run Strategies ---------
if run_backtest and close_col and not data.empty:
    st.markdown("---")
    results = {}
    for strategy in selected_strategies:
        results[strategy] = run_strategy(data, strategy)

    # --------- Metrics Table ---------
    st.subheader("📋 Strategy Performance Summary")
    metrics_df = pd.DataFrame([
        {
            "Strategy": r["name"],
            "Final Value ($)": round(r["final_value"], 2),
            "Total Return (%)": round(r["total_return"] * 100, 2),
            "Annualized Return (%)": round(r["annualized"] * 100, 2),
            "Max Drawdown (%)": round(r["max_dd"] * 100, 2),
            "Sharpe Ratio": round(r["sharpe"], 2)
        } for r in results.values()
    ])
    st.dataframe(metrics_df)

    # --------- Line Chart ---------
    st.subheader("📊 Equity Curve Comparison")
    chart_df = pd.concat([r["portfolio"] for r in results.values()], axis=1)
    chart_df.columns = results.keys()
    st.line_chart(chart_df)

    # --------- Trade Logs ---------
    for name, res in results.items():
        st.markdown(f"### 📑 {name} Trade Log")
        st.dataframe(res["trades"])
else:
    if run_backtest:
        st.warning("⚠️ Could not load price data.")
