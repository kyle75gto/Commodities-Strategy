import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Simuler des données de prix pour 6 mono-indices
np.random.seed(77)
dates = pd.date_range(start="2023-01-01", end="2024-12-31", freq="B")
n = len(dates)

commodities = {
    "BCOMGC": 1800,  # Or
    "BCOMAL": 2500,  # Aluminium
    "BCOMCL": 70,    # Pétrole brut
    "BCOMW": 600,    # Blé
    "BCOMHG": 4000,  # Cuivre
    "BCOMLN": 90     # Porc maigre
}

price_data = pd.DataFrame(index=dates)
for code, start_price in commodities.items():
    price_data[code] = start_price * (1 + np.random.normal(0, 0.01, size=n)).cumprod()

lookback = 20               # fenêtre de calcul des performances passées
rebalance_freq = 20         # fréquence de rebalancement (en jours ouvrés)
n_long_short = 3            # nombre de long et de short

index_values = []
weights_history = []
rebalance_dates = []
initial_index_value = 100
index_values.append(initial_index_value)



# Boucle principale pour construire l'indice
for i in range(lookback, len(price_data)):
    today = price_data.index[i]
    rebalance = ((i - lookback) % rebalance_freq == 0)

    if rebalance:
        perf = price_data.iloc[i - lookback:i].pct_change().sum()
        ranked = perf.sort_values()
        longs = ranked.head(n_long_short).index.tolist()
        shorts = ranked.tail(n_long_short).index.tolist()

        weights = {c: 0 for c in price_data.columns}
        for c in longs:
            weights[c] = 1 / n_long_short
        for c in shorts:
            weights[c] = -1 / n_long_short

        rebalance_UI = price_data.loc[today].copy()
        rebalance_weights = weights.copy()
        rebalance_dates.append(today)

    curr_UI = price_data.iloc[i]
    ER_t = index_values[-1] + sum(
        rebalance_weights[c] * ((curr_UI[c] / rebalance_UI[c]) - 1)
        for c in price_data.columns
    ) * index_values[-1]

    index_values.append(ER_t)
    weights_history.append(rebalance_weights.copy())

# Série temporelle finale de l'indice
index_dates = price_data.index[lookback:]
index_series = pd.Series(index=index_dates, data=index_values[1:])

# Calcul des indicateurs de performance
returns = index_series.pct_change().dropna()
cumulative_return = index_series.iloc[-1] / index_series.iloc[0] - 1
volatility = returns.std() * np.sqrt(252)
sharpe_ratio = returns.mean() / returns.std() * np.sqrt(252)
max_drawdown = ((index_series / index_series.cummax()) - 1).min()

# Affichage du track de l'indice
plt.figure(figsize=(10, 5))
plt.plot(index_series, label="Strategy Index")
plt.title("Mean Reversion Commodity Strategy Index")
plt.xlabel("Date")
plt.ylabel("Index Value")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

# Affichage des indicateurs
metrics = pd.DataFrame({
    "Cumulative Return": [f"{cumulative_return:.2%}"],
    "Annualized Volatility": [f"{volatility:.2%}"],
    "Sharpe Ratio": [f"{sharpe_ratio:.2f}"],
    "Max Drawdown": [f"{max_drawdown:.2%}"]
})

print(metrics)