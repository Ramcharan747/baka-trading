"""
NIFTY50 instruments for Phase 3.

Format: "NSE_EQ|{ISIN}" for stocks, "NSE_INDEX|Nifty 50" for index.
Deduplicated — 49 unique stocks + 1 index = 50 instruments.
"""

INSTRUMENTS = {
    # Financials
    "HDFCBANK":   "NSE_EQ|INE040A01034",
    "ICICIBANK":  "NSE_EQ|INE090A01021",
    "KOTAKBANK":  "NSE_EQ|INE237A01028",
    "SBIN":       "NSE_EQ|INE062A01020",
    "AXISBANK":   "NSE_EQ|INE238A01034",
    "BAJFINANCE": "NSE_EQ|INE296A01024",
    "BAJAJFINSV": "NSE_EQ|INE918I01026",
    "INDUSINDBK": "NSE_EQ|INE095A01012",
    "HDFCLIFE":   "NSE_EQ|INE795G01014",
    "SBILIFE":    "NSE_EQ|INE123W01016",
    "SHRIRAMFIN": "NSE_EQ|INE721A01013",
    # IT
    "TCS":        "NSE_EQ|INE467B01029",
    "INFY":       "NSE_EQ|INE009A01021",
    "HCLTECH":    "NSE_EQ|INE860A01027",
    "WIPRO":      "NSE_EQ|INE075A01022",
    "TECHM":      "NSE_EQ|INE669C01036",
    "LTIM":       "NSE_EQ|INE214T01019",
    # Energy
    "RELIANCE":   "NSE_EQ|INE002A01018",
    "ONGC":       "NSE_EQ|INE213A01029",
    "COALINDIA":  "NSE_EQ|INE522F01014",
    "BPCL":       "NSE_EQ|INE029A01011",
    # Materials
    "TATASTEEL":  "NSE_EQ|INE081A01020",
    "JSWSTEEL":   "NSE_EQ|INE019A01038",
    "HINDALCO":   "NSE_EQ|INE038A01020",
    "GRASIM":     "NSE_EQ|INE047A01021",
    "ULTRACEMCO": "NSE_EQ|INE481G01011",
    # Consumer & FMCG
    "HINDUNILVR": "NSE_EQ|INE030A01027",
    "ITC":        "NSE_EQ|INE154A01025",
    "NESTLEIND":  "NSE_EQ|INE239A01024",
    "BRITANNIA":  "NSE_EQ|INE216A01030",
    "TATACONSUM": "NSE_EQ|INE192A01025",
    "ASIANPAINT": "NSE_EQ|INE021A01026",
    # Auto
    "MARUTI":     "NSE_EQ|INE585B01010",
    "TATAMOTORS": "NSE_EQ|INE155A01022",
    "M&M":        "NSE_EQ|INE101A01026",
    "BAJAJ-AUTO": "NSE_EQ|INE917I01010",
    "EICHERMOT":  "NSE_EQ|INE066A01021",
    "HEROMOTOCO": "NSE_EQ|INE158A01026",
    # Pharma
    "SUNPHARMA":  "NSE_EQ|INE044A01036",
    "DRREDDY":    "NSE_EQ|INE089A01023",
    "CIPLA":      "NSE_EQ|INE059A01026",
    "DIVISLAB":   "NSE_EQ|INE361B01024",
    "APOLLOHOSP": "NSE_EQ|INE437A01024",
    # Infrastructure & Utilities
    "LT":         "NSE_EQ|INE018A01030",
    "POWERGRID":  "NSE_EQ|INE752E01010",
    "NTPC":       "NSE_EQ|INE733E01010",
    "ADANIPORTS": "NSE_EQ|INE742F01042",
    # Mixed
    "TITAN":      "NSE_EQ|INE280A01028",
    "BHARTIARTL": "NSE_EQ|INE397D01024",
    # Market factor (not predicted, used as feature only)
    "NIFTY50":    "NSE_INDEX|Nifty 50",
}

# Stocks to predict (everything except the index)
PREDICT_STOCKS = [k for k in INSTRUMENTS if k != "NIFTY50"]

# Sector mapping for sector-level features
SECTORS = {
    "FINANCIALS": ["HDFCBANK", "ICICIBANK", "KOTAKBANK", "SBIN", "AXISBANK",
                   "BAJFINANCE", "BAJAJFINSV", "INDUSINDBK", "HDFCLIFE",
                   "SBILIFE", "SHRIRAMFIN"],
    "IT":         ["TCS", "INFY", "HCLTECH", "WIPRO", "TECHM", "LTIM"],
    "ENERGY":     ["RELIANCE", "ONGC", "COALINDIA", "BPCL"],
    "MATERIALS":  ["TATASTEEL", "JSWSTEEL", "HINDALCO", "GRASIM", "ULTRACEMCO"],
    "CONSUMER":   ["HINDUNILVR", "ITC", "NESTLEIND", "BRITANNIA",
                   "TATACONSUM", "ASIANPAINT"],
    "AUTO":       ["MARUTI", "TATAMOTORS", "M&M", "BAJAJ-AUTO",
                   "EICHERMOT", "HEROMOTOCO"],
    "PHARMA":     ["SUNPHARMA", "DRREDDY", "CIPLA", "DIVISLAB", "APOLLOHOSP"],
    "INFRA":      ["LT", "POWERGRID", "NTPC", "ADANIPORTS"],
    "MIXED":      ["TITAN", "BHARTIARTL"],
}

# Reverse lookup: stock → sector
STOCK_SECTOR = {}
for sector, stocks in SECTORS.items():
    for s in stocks:
        STOCK_SECTOR[s] = sector
