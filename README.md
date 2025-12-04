м# StockAI – End-to-End Price Prediction MVP (Backend + Mobile App)

StockAI is an educational and portfolio project that implements an **end-to-end system** for forecasting financial asset prices and delivering signals to a **mobile application**.

The project covers the full lifecycle:

- data collection and feature engineering,
- model training and packaging,
- REST API for serving predictions (offline + online),
- mobile UI for retail-style users.

---

## 🎯 Project Goals

- **Educational goal**  
  Final capstone project for a price-prediction module (week 14).

- **Portfolio goal**  
  Demonstrate the ability to design and implement a small but complete product:
  - ML pipeline (features + models),
  - production-like backend (FastAPI + Postgres),
  - mobile client (React Native / Expo),
  - reproducible setup for future deployment (e.g., cloud).

---

## 📈 Supported Assets and Horizons

**Assets**

- Bitcoin (**BTC**)
- Ethereum (**ETH**)
- Apple (**AAPL**)
- Tesla (**TSLA**)

**Forecast horizons (T+N)**

- **T+1** — 1 day ahead  
- **T+3** — 3 days ahead  
- **T+7** — 7 days ahead  

For each asset and horizon, the system uses **hybrid XGBoost-based models** combining:

- a **classifier** (probability that the price will go up),
- a **regressor** (magnitude of the move),
- a **gating logic** that filters/adjusts the signal quality.
