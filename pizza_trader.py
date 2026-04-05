"""Here comes the Pizza Money! - Refactored with structured logging."""

import logging
import os
from datetime import datetime, timedelta
from typing import Any, Dict, List
import warnings

import pandas as pd
import requests
import talib
import yfinance as yf
from dotenv import load_dotenv
from rich.console import Console
from rich.table import Table

from datapizza.agents import Agent
from datapizza.clients.google import GoogleClient
from datapizza.tools import tool

warnings.simplefilter("ignore")
load_dotenv()

# ---------------------------------------------------------------------------
# Logging configuration
# ---------------------------------------------------------------------------
# Force UTF-8 on Windows (default console codec is CP1252 and chokes on
# non-ASCII characters like the checkmark used in log messages).
import sys
if sys.stdout.encoding and sys.stdout.encoding.upper() not in ("UTF-8", "UTF8"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if sys.stderr.encoding and sys.stderr.encoding.upper() not in ("UTF-8", "UTF8"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

logger = logging.getLogger("PizzaMoney")
# Guard against duplicate handlers when the module is reloaded
if not logger.handlers:
    logger.setLevel(logging.INFO)
    _formatter = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    # Console handler (UTF-8 already reconfigured above)
    _stream_handler = logging.StreamHandler(sys.stdout)
    _stream_handler.setFormatter(_formatter)
    logger.addHandler(_stream_handler)

    # File handler - one log file per run, UTF-8 explicit (safe on Windows)
    # Set LOG_DIR in your .env to control where log files are saved.
    # If not set, defaults to the same folder as this script.
    _log_dir = os.environ.get("LOG_DIR", os.path.dirname(os.path.abspath(__file__)))
    os.makedirs(_log_dir, exist_ok=True)
    _log_filename = os.path.join(
        _log_dir, f"pizza_money_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    )
    _file_handler = logging.FileHandler(_log_filename, encoding="utf-8")
    _file_handler.setFormatter(_formatter)
    logger.addHandler(_file_handler)
    logger.info("Log file: %s", _log_filename)

    logger.propagate = False  # prevent double-printing via root logger


# ---------------------------------------------------------------------------
# MarketAnalyzer
# ---------------------------------------------------------------------------
class MarketAnalyzer:
    """
    A comprehensive market analysis toolkit for AI agents.

    Exposes multiple analysis methods as LangChain-compatible tools for
    automated trading research, sector rotation analysis, technical signal
    detection, and news sentiment analysis.
    """

    def __init__(self) -> None:
        """Initialize the MarketAnalyzer with API credentials and stock universe."""
        logger.info("Initializing MarketAnalyzer…")

        self.finnhub_api_key: str | None = os.environ.get("FINNHUB_API_KEY")
        self.marketaux_api_key: str | None = os.environ.get("MARKETAUX_API_KEY")

        if not self.finnhub_api_key:
            logger.warning("FINNHUB_API_KEY not found in environment variables.")
        if not self.marketaux_api_key:
            logger.warning("MARKETAUX_API_KEY not found in environment variables.")

        self.finnhub_base_url = "https://finnhub.io/api/v1/company-news"
        self.sector_etfs = self._initialize_sector_etfs()
        logger.info("MarketAnalyzer initialized with %d sector ETFs.", len(self.sector_etfs))

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _initialize_sector_etfs() -> Dict[str, str]:
        """Return sector → ETF ticker mapping."""
        return {
            "Basic Materials": "XLB",
            "Communication Services": "XLC",
            "Consumer Cyclical": "XLY",
            "Consumer Defensive": "XLP",
            "Energy": "XLE",
            "Financial Services": "XLF",
            "Healthcare": "XLV",
            "Industrials": "XLI",
            "Real Estate": "VNQ",
            "Technology": "XLK",
            "Utilities": "XLU",
        }

    @staticmethod
    def _safe_float(value: Any, fallback: float = 0.0) -> float:
        """Convert *value* to float, returning *fallback* on failure."""
        try:
            return float(value)
        except (TypeError, ValueError):
            return fallback

    @staticmethod
    def _safe_int(value: Any, fallback: int = 0) -> int:
        """Convert *value* to int, returning *fallback* on NaN / None."""
        try:
            f = float(value)
            if pd.isna(f):
                return fallback
            return int(f)
        except (TypeError, ValueError):
            return fallback

    # ------------------------------------------------------------------
    # Sector rotation
    # ------------------------------------------------------------------

    def calculate_sector_rotation_signal(self) -> pd.DataFrame:
        """
        Calculate 1-month momentum signals for major market sectors via ETF data.

        Downloads the last 252 trading days of closing prices for each sector ETF
        and computes momentum as the percentage change over the last ~21 sessions.

        Returns:
            pd.DataFrame: Columns - ``ticker``, ``Sector``, ``momentum``.

        Notes:
            momentum = ((close_today / close_21_days_ago) − 1) × 100
        """
        logger.info("Calculating sector rotation signals for %d ETFs…", len(self.sector_etfs))
        rows: List[Dict[str, Any]] = []

        for sector_name, etf_ticker in self.sector_etfs.items():
            try:
                logger.debug("Fetching history for ETF: %s (%s)", etf_ticker, sector_name)
                data = yf.Ticker(etf_ticker).history(period="252d")

                if data.empty or len(data) < 22:
                    logger.warning(
                        "Insufficient data for ETF %s - skipping (rows: %d).",
                        etf_ticker,
                        len(data),
                    )
                    continue

                close_today = float(data["Close"].iloc[-1])
                close_21d_ago = float(data["Close"].iloc[-21])
                momentum = (close_today / close_21d_ago - 1) * 100

                rows.append(
                    {"ticker": etf_ticker, "Sector": sector_name, "momentum": momentum}
                )
                logger.debug("%s momentum: %.2f%%", etf_ticker, momentum)

            except Exception as exc:
                logger.error(
                    "Error calculating momentum for ETF %s (%s): %s",
                    etf_ticker,
                    sector_name,
                    exc,
                    exc_info=True,
                )
                continue

        logger.info("Sector rotation signal calculated for %d / %d ETFs.", len(rows), len(self.sector_etfs))
        return pd.DataFrame(rows)

    # ------------------------------------------------------------------
    # Analyze signals (tool)
    # ------------------------------------------------------------------

    @tool
    def analyze_signals(self, ticker_names: str) -> str:
        """
        Analyze tickers based on price gaps, volume spikes, options data, and sector momentum.

        For each ticker downloads 252 days of OHLCV data and computes:
        - Gap size % (last open vs previous close)
        - Volume spike (last vol > 4× 10-day average)
        - Float shares, current price, sector
        - Options chain metrics (nearest expiry): total OI / volume, put/call ratios, DTE
        - Sector momentum from ``calculate_sector_rotation_signal``

        Args:
            ticker_names: Comma-separated ticker symbols, e.g. ``'AAPL,MSFT'``.

        Returns:
            String representation of a list of result dicts.
        """
        logger.info("analyze_signals called with: %s", ticker_names)
        watchlist = [t.strip() for t in ticker_names.split(",") if t.strip()]
        results: List[Dict[str, Any]] = []

        logger.info("Computing sector rotation signals…")
        sector_momentum_df = self.calculate_sector_rotation_signal()

        for ticker in watchlist:
            logger.info("[analyze_signals] Processing ticker: %s", ticker)
            try:
                # ---- Price / volume data ----
                logger.debug("[%s] Downloading historical data…", ticker)
                data = yf.download(ticker, period="252d", interval="1d", progress=False)

                if data.empty:
                    logger.warning("[%s] Empty price history - skipping.", ticker)
                    continue

                # Multi-level columns when downloading a single ticker with newer yfinance
                if isinstance(data.columns, pd.MultiIndex):
                    data = data.xs(ticker, axis=1, level="Ticker")

                if len(data) < 2:
                    logger.warning("[%s] Not enough rows (%d) - skipping.", ticker, len(data))
                    continue

                last_open: float = float(data["Open"].iloc[-1])
                prev_close: float = float(data["Close"].iloc[-2])
                gap_size_pct: float = round((last_open - prev_close) / prev_close * 100, 2)
                logger.debug("[%s] Gap size: %.2f%%", ticker, gap_size_pct)

                avg_volume_10: float = float(data["Volume"].iloc[-11:-1].mean())
                last_volume: float = float(data["Volume"].iloc[-1])
                volume_spike: bool = last_volume > 4 * avg_volume_10
                logger.debug(
                    "[%s] Volume spike: %s (last: %.0f, avg10: %.0f)",
                    ticker, volume_spike, last_volume, avg_volume_10,
                )

                # ---- Ticker info ----
                logger.debug("[%s] Fetching ticker info…", ticker)
                info = yf.Ticker(ticker).info
                float_shares = info.get("floatShares") or info.get("float")
                price: float = info.get("regularMarketPrice") or last_open
                sector: str | None = info.get("sector")
                logger.debug("[%s] Price: %.2f | Sector: %s", ticker, price, sector)

                # ---- Options chain ----
                logger.debug("[%s] Fetching options chain…", ticker)
                ticker_obj = yf.Ticker(ticker)
                expirations = ticker_obj.options

                total_call_oi = total_put_oi = total_call_volume = total_put_volume = None
                put_call_oi_ratio = put_call_volume_ratio = None
                dte = None

                if expirations:
                    expiry = expirations[0]
                    logger.debug("[%s] Using nearest expiry: %s", ticker, expiry)
                    try:
                        option_chain = ticker_obj.option_chain(expiry)
                        calls = option_chain.calls
                        puts = option_chain.puts

                        total_call_oi = self._safe_int(calls["openInterest"].sum())
                        total_put_oi = self._safe_int(puts["openInterest"].sum())
                        total_call_volume = self._safe_int(calls["volume"].sum())
                        total_put_volume = self._safe_int(puts["volume"].sum())

                        put_call_oi_ratio = (
                            round(total_put_oi / total_call_oi, 2) if total_call_oi else None
                        )
                        put_call_volume_ratio = (
                            round(total_put_volume / total_call_volume, 2)
                            if total_call_volume
                            else None
                        )

                        expiry_date = datetime.strptime(expiry, "%Y-%m-%d").date()
                        dte = (expiry_date - datetime.today().date()).days
                        logger.debug(
                            "[%s] Options - call OI: %d | put OI: %d | DTE: %d",
                            ticker, total_call_oi, total_put_oi, dte,
                        )
                    except Exception as opt_exc:
                        logger.error(
                            "[%s] Error fetching options chain for expiry %s: %s",
                            ticker, expiry, opt_exc, exc_info=True,
                        )
                else:
                    logger.debug("[%s] No options expirations available.", ticker)

                # ---- Sector momentum ----
                sector_momentum: float = 0.0
                if sector and sector in sector_momentum_df["Sector"].values:
                    try:
                        sector_momentum = float(
                            sector_momentum_df.loc[
                                sector_momentum_df["Sector"] == sector, "momentum"
                            ].iloc[0]
                        )
                    except (KeyError, IndexError) as sm_exc:
                        logger.warning(
                            "[%s] Could not retrieve sector momentum for '%s': %s",
                            ticker, sector, sm_exc,
                        )

                results.append(
                    {
                        "ticker": ticker,
                        "gap_size_pct": gap_size_pct,
                        "volume_spike": float(volume_spike),
                        "price": float(price),
                        "float_shares": float(float_shares) if float_shares else None,
                        "sector": sector,
                        "sector_momentum": round(sector_momentum, 2),
                        "total_call_open_interest": total_call_oi,
                        "total_put_open_interest": total_put_oi,
                        "total_call_volume": total_call_volume,
                        "total_put_volume": total_put_volume,
                        "put_call_oi_ratio": put_call_oi_ratio,
                        "put_call_volume_ratio": put_call_volume_ratio,
                        "days_to_expiry": dte,
                    }
                )
                logger.info("[analyze_signals] [OK] %s processed successfully.", ticker)

            except Exception as exc:
                logger.error(
                    "[analyze_signals] Unhandled error for ticker %s: %s",
                    ticker, exc, exc_info=True,
                )
                continue

        logger.info(
            "analyze_signals completed: %d / %d tickers processed.", len(results), len(watchlist)
        )
        return str(results)

    # ------------------------------------------------------------------
    # Expert signals (tool)
    # ------------------------------------------------------------------

    @tool
    def expert_signals(self, ticker_names: str) -> str:
        """
        Generate expert-level technical trading signals for each ticker.

        Downloads 252 days of daily OHLCV data and uses TA-Lib to compute:
        - Trend: SMA50/200 crossover, ADX, Parabolic SAR
        - Momentum: RSI, MACD, CCI, Stochastic, Williams %R
        - Volatility: ATR, Bollinger Bands, ROC, TRIX, Ultimate Oscillator
        - Volume: OBV trend, MFI
        - Support/Resistance: 50-day rolling min/max
        - Candlestick patterns: Engulfing, Hammer, Shooting Star, Doji, Morning/Evening Star,
          Harami, Piercing Line, Dark Cloud Cover

        Args:
            ticker_names: Comma-separated ticker symbols, e.g. ``'AAPL,MSFT'``.

        Returns:
            String representation of a list of signal dicts.
        """
        logger.info("expert_signals called with: %s", ticker_names)
        watchlist = [t.strip() for t in ticker_names.split(",") if t.strip()]
        results: List[Dict[str, Any]] = []

        for ticker in watchlist:
            logger.info("[expert_signals] Processing ticker: %s", ticker)
            try:
                logger.debug("[%s] Downloading historical data…", ticker)
                data = yf.download(ticker, period="252d", interval="1d", progress=False)

                if data.empty:
                    logger.warning("[%s] Empty price history - skipping.", ticker)
                    continue

                if isinstance(data.columns, pd.MultiIndex):
                    data = data.xs(ticker, axis=1, level="Ticker")

                if len(data) < 200:
                    logger.warning(
                        "[%s] Insufficient rows (%d < 200) for indicator calculation - skipping.",
                        ticker, len(data),
                    )
                    continue

                close = data["Close"]
                high = data["High"]
                low = data["Low"]
                open_ = data["Open"]
                volume = data["Volume"]

                # ---- Technical indicators ----
                logger.debug("[%s] Computing technical indicators…", ticker)
                try:
                    sma50 = talib.SMA(close, 50)
                    sma200 = talib.SMA(close, 200)
                    adx = talib.ADX(high, low, close, 14)
                    sar = talib.SAR(high, low, 0.02, 0.2)
                    rsi = talib.RSI(close, 14)
                    macd, macdsignal, macdhist = talib.MACD(close, 12, 26, 9)
                    cci = talib.CCI(high, low, close, 14)
                    slowk, slowd = talib.STOCH(high, low, close, 14, 3, 0, 3, 0)
                    willr = talib.WILLR(high, low, close, 14)
                    atr = talib.ATR(high, low, close, 14)
                    upperband, middleband, lowerband = talib.BBANDS(close, 20, 2, 2, 0)
                    obv = talib.OBV(close, volume)
                    mfi = talib.MFI(high, low, close, volume, 14)
                    roc = talib.ROC(close, 10)
                    trix = talib.TRIX(close, 14)
                    ultosc = talib.ULTOSC(high, low, close, 7, 14, 28)
                except Exception as ind_exc:
                    logger.error(
                        "[%s] Error computing TA-Lib indicators: %s",
                        ticker, ind_exc, exc_info=True,
                    )
                    continue

                # ---- Candlestick patterns ----
                logger.debug("[%s] Computing candlestick patterns…", ticker)
                try:
                    patterns = {
                        "CDL_ENGULFING": talib.CDLENGULFING(open_, high, low, close),
                        "CDL_HAMMER": talib.CDLHAMMER(open_, high, low, close),
                        "CDL_SHOOTING_STAR": talib.CDLSHOOTINGSTAR(open_, high, low, close),
                        "CDL_DOJI": talib.CDLDOJI(open_, high, low, close),
                        "CDL_MORNING_STAR": talib.CDLMORNINGSTAR(open_, high, low, close),
                        "CDL_EVENING_STAR": talib.CDLEVENINGSTAR(open_, high, low, close),
                        "CDL_HARAMI": talib.CDLHARAMI(open_, high, low, close),
                        "CDL_PIERCING": talib.CDLPIERCING(open_, high, low, close),
                        "CDL_DARK_CLOUD_COVER": talib.CDLDARKCLOUDCOVER(open_, high, low, close),
                    }
                except Exception as pat_exc:
                    logger.error(
                        "[%s] Error computing candlestick patterns: %s",
                        ticker, pat_exc, exc_info=True,
                    )
                    patterns = {}

                def pattern_signal(arr) -> str:
                    val = arr.iloc[-1]
                    if val > 0:
                        return "Bullish"
                    elif val < 0:
                        return "Bearish"
                    return "None"

                # ---- Signal interpretation ----
                logger.debug("[%s] Interpreting signals…", ticker)

                last_close = float(close.iloc[-1])
                last_sma50 = float(sma50.iloc[-1])
                last_sma200 = float(sma200.iloc[-1])
                last_adx = float(adx.iloc[-1])
                last_sar = float(sar.iloc[-1])
                last_rsi = float(rsi.iloc[-1])
                last_macdhist = float(macdhist.iloc[-1])
                prev_macdhist = float(macdhist.iloc[-2])
                last_cci = float(cci.iloc[-1])
                last_slowk = float(slowk.iloc[-1])
                last_slowd = float(slowd.iloc[-1])
                last_willr = float(willr.iloc[-1])
                last_upperband = float(upperband.iloc[-1])
                last_lowerband = float(lowerband.iloc[-1])
                last_mfi = float(mfi.iloc[-1])
                last_atr = float(atr.iloc[-1])
                last_roc = float(roc.iloc[-1])
                last_trix = float(trix.iloc[-1])
                last_ultosc = float(ultosc.iloc[-1])

                bullish_trend: bool = (
                    last_close > last_sma50
                    and last_close > last_sma200
                    and last_adx > 25
                    and last_close > last_sar
                )
                bearish_trend: bool = (
                    last_close < last_sma50
                    and last_close < last_sma200
                    and last_adx > 25
                    and last_close < last_sar
                )

                # RSI
                if last_rsi < 30:
                    rsi_signal = "Oversold"
                elif last_rsi > 70:
                    rsi_signal = "Overbought"
                else:
                    rsi_signal = "Neutral"

                # MACD
                if prev_macdhist < 0 < last_macdhist:
                    macd_signal = "Bullish Crossover"
                elif prev_macdhist > 0 > last_macdhist:
                    macd_signal = "Bearish Crossover"
                else:
                    macd_signal = "Neutral"

                # CCI
                if last_cci < -100:
                    cci_signal = "Oversold"
                elif last_cci > 100:
                    cci_signal = "Overbought"
                else:
                    cci_signal = "Neutral"

                # Stochastic
                if last_slowk < 20 and last_slowk > last_slowd:
                    stoch_signal = "Bullish Crossover"
                elif last_slowk > 80 and last_slowk < last_slowd:
                    stoch_signal = "Bearish Crossover"
                else:
                    stoch_signal = "Neutral"

                # Williams %R
                if last_willr < -80:
                    willr_signal = "Oversold"
                elif last_willr > -20:
                    willr_signal = "Overbought"
                else:
                    willr_signal = "Neutral"

                # Bollinger Bands
                if last_close >= last_upperband:
                    boll_signal = "Touch Upper Band"
                elif last_close <= last_lowerband:
                    boll_signal = "Touch Lower Band"
                else:
                    boll_signal = "Within Bands"

                # OBV trend (last 3 sessions all positive delta)
                vol_confirm: bool = bool(pd.Series(obv).diff().iloc[-3:].gt(0).all())

                # MFI
                if last_mfi < 20:
                    mfi_signal = "Oversold"
                elif last_mfi > 80:
                    mfi_signal = "Overbought"
                else:
                    mfi_signal = "Neutral"

                # Support / Resistance
                support: float = float(close.rolling(50).min().iloc[-1])
                resistance: float = float(close.rolling(50).max().iloc[-1])
                support_break: bool = last_close < support
                resistance_break: bool = last_close > resistance

                candlestick_signals = {
                    key: pattern_signal(val) for key, val in patterns.items()
                }

                results.append(
                    {
                        "azienda": ticker,
                        "Bullish_Trend": bullish_trend,
                        "Bearish_Trend": bearish_trend,
                        "RSI_Signal": rsi_signal,
                        "MACD_Signal": macd_signal,
                        "CCI_Signal": cci_signal,
                        "Stochastic_Signal": stoch_signal,
                        "WilliamsR_Signal": willr_signal,
                        "Bollinger_Signal": boll_signal,
                        "Volume_Confirmation": vol_confirm,
                        "MFI_Signal": mfi_signal,
                        "Support_Break": support_break,
                        "Resistance_Break": resistance_break,
                        "ATR": round(last_atr, 4),
                        "ROC": round(last_roc, 4),
                        "TRIX": round(last_trix, 4),
                        "Ultimate_Oscillator": round(last_ultosc, 4),
                        **candlestick_signals,
                    }
                )
                logger.info("[expert_signals] [OK] %s processed successfully.", ticker)

            except Exception as exc:
                logger.error(
                    "[expert_signals] Unhandled error for ticker %s: %s",
                    ticker, exc, exc_info=True,
                )
                continue

        logger.info(
            "expert_signals completed: %d / %d tickers processed.", len(results), len(watchlist)
        )
        return str(results)

    # ------------------------------------------------------------------
    # Finnhub news (tool)
    # ------------------------------------------------------------------

    @tool
    def finnhub_news_analysis(self, ticker_names: str) -> str:
        """
        Fetch recent news articles (last 3 days) via the Finnhub API.

        Args:
            ticker_names: Comma-separated ticker symbols, e.g. ``'AAPL,MSFT'``.

        Returns:
            String representation of a dict mapping each ticker to its article list.
        """
        logger.info("finnhub_news_analysis called with: %s", ticker_names)

        if not self.finnhub_api_key:
            logger.error("FINNHUB_API_KEY is not set - cannot fetch news.")
            return str({})

        summaries: Dict[str, List[Dict[str, Any]]] = {}
        end_date = datetime.now()
        start_date = end_date - timedelta(days=3)
        date_from = start_date.strftime("%Y-%m-%d")
        date_to = end_date.strftime("%Y-%m-%d")

        watchlist = [t.strip() for t in ticker_names.split(",") if t.strip()]

        for ticker in watchlist:
            logger.info("[finnhub_news] Fetching news for: %s", ticker)
            summaries[ticker] = []
            try:
                params = {
                    "symbol": ticker,
                    "from": date_from,
                    "to": date_to,
                    "token": self.finnhub_api_key,
                }

                logger.debug("[%s] GET %s params=%s", ticker, self.finnhub_base_url, params)
                response = requests.get(self.finnhub_base_url, params=params, timeout=10)

                if response.status_code != 200:
                    logger.error(
                        "[%s] Finnhub API returned HTTP %d: %s",
                        ticker, response.status_code, response.text[:200],
                    )
                    continue

                articles = response.json()
                logger.debug("[%s] Received %d articles from Finnhub.", ticker, len(articles))

                articles_sorted = sorted(
                    articles, key=lambda x: x.get("datetime", 0), reverse=True
                )

                for article in articles_sorted:
                    try:
                        ts = article.get("datetime", 0)
                        summaries[ticker].append(
                            {
                                "datetime": datetime.fromtimestamp(ts).strftime(
                                    "%Y-%m-%d %H:%M:%S"
                                ),
                                "headline": article.get("headline", ""),
                                "summary": article.get("summary", ""),
                                "url": article.get("url", ""),
                            }
                        )
                    except Exception as art_exc:
                        logger.warning(
                            "[%s] Could not parse article entry: %s", ticker, art_exc
                        )
                        continue

                logger.info(
                    "[finnhub_news] [OK] %s - %d articles fetched.", ticker, len(summaries[ticker])
                )

            except requests.exceptions.Timeout:
                logger.error("[%s] Finnhub request timed out.", ticker)
            except requests.exceptions.RequestException as req_exc:
                logger.error("[%s] Finnhub request error: %s", ticker, req_exc, exc_info=True)
            except Exception as exc:
                logger.error(
                    "[finnhub_news] Unhandled error for ticker %s: %s",
                    ticker, exc, exc_info=True,
                )

        logger.info(
            "finnhub_news_analysis completed for %d tickers.", len(summaries)
        )
        return str(summaries)

    # ------------------------------------------------------------------
    # Marketaux news (tool)
    # ------------------------------------------------------------------

    @tool
    def marketaux_news_analysis(self, ticker_names: str) -> str:
        """
        Fetch recent news for Italian and European tickers via the Marketaux API (up to 10 articles/ticker).

        Args:
            ticker_names: Comma-separated ticker symbols, e.g. ``'ENI.MI,ISP.MI'``.

        Returns:
            String representation of a dict mapping each ticker to its article list.
        """
        logger.info("marketaux_news_analysis called with: %s", ticker_names)

        if not self.marketaux_api_key:
            logger.error("MARKETAUX_API_KEY is not set - cannot fetch news.")
            return str({})

        summaries: Dict[str, List[Dict[str, Any]]] = {}
        watchlist = [t.strip() for t in ticker_names.split(",") if t.strip()]

        for ticker in watchlist:
            logger.info("[marketaux_news] Fetching news for: %s", ticker)
            summaries[ticker] = []
            try:
                params = {
                    "symbols": ticker,
                    "filter_entities": "true",
                    "limit": 10,
                    "api_token": self.marketaux_api_key,
                }

                logger.debug("[%s] GET https://api.marketaux.com/v1/news/all params=%s", ticker, params)
                response = requests.get(
                    "https://api.marketaux.com/v1/news/all",
                    params=params,
                    timeout=10,
                )

                if response.status_code != 200:
                    logger.error(
                        "[%s] Marketaux API returned HTTP %d: %s",
                        ticker, response.status_code, response.text[:200],
                    )
                    continue

                payload = response.json()
                articles = payload.get("data", [])
                logger.debug("[%s] Received %d articles from Marketaux.", ticker, len(articles))

                for article in articles:
                    try:
                        summaries[ticker].append(
                            {
                                "datetime": article.get("published_at", ""),
                                "headline": article.get("title", ""),
                                "summary": article.get("description", "")
                                or article.get("snippet", ""),
                                "url": article.get("url", ""),
                            }
                        )
                    except Exception as art_exc:
                        logger.warning(
                            "[%s] Could not parse article entry: %s", ticker, art_exc
                        )
                        continue

                logger.info(
                    "[marketaux_news] [OK] %s - %d articles fetched.", ticker, len(summaries[ticker])
                )

            except requests.exceptions.Timeout:
                logger.error("[%s] Marketaux request timed out.", ticker)
            except requests.exceptions.RequestException as req_exc:
                logger.error("[%s] Marketaux request error: %s", ticker, req_exc, exc_info=True)
            except Exception as exc:
                logger.error(
                    "[marketaux_news] Unhandled error for ticker %s: %s",
                    ticker, exc, exc_info=True,
                )

        logger.info(
            "marketaux_news_analysis completed for %d tickers.", len(summaries)
        )
        return str(summaries)


# ---------------------------------------------------------------------------
# Rich table rendering
# ---------------------------------------------------------------------------

def render_table_with_separators(response: str) -> None:
    """
    Parse a Markdown pipe-delimited table from *response* and render it with Rich.

    Also prints any trailing disclaimer line.

    Args:
        response: Raw string output from the agent, expected to contain a Markdown table.
    """
    logger.info("Rendering output table…")
    lines = [line.strip() for line in response.splitlines() if line.strip().startswith("|")]

    if not lines:
        logger.warning("No pipe-delimited table found in agent output.")
        print("No table found in output.")
        return

    # Header row
    header = [h.strip() for h in lines[0].split("|") if h.strip()]
    table = Table(title="Trading Report", show_lines=True)
    for col in header:
        table.add_column(col, style="bold cyan")

    # Data rows (skip header + markdown separator line)
    added = 0
    for row_line in lines[2:]:
        fields = [f.strip() for f in row_line.split("|") if f.strip()]
        if len(fields) == len(header):
            table.add_row(*fields)
            added += 1
        else:
            logger.debug(
                "Skipping row with mismatched column count (%d vs %d): %s",
                len(fields), len(header), row_line,
            )

    logger.info("Table rendered with %d data rows.", added)
    console = Console()
    console.print(table)

    # Disclaimer
    for line in response.splitlines():
        if line.lower().startswith("disclaimer:"):
            console.print(f"[bold yellow]{line}[/bold yellow]")
            break


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logger.info("Starting Pizza Money agent…")

    analyzer = MarketAnalyzer()

    agent = Agent(
        name="Pizza Trader",
        system_prompt="""
        You are Pizza Trader, a senior quantitative trader and AI research partner.
        Your goal is to provide clear and elegant trading reports combining technical, sentiment, and news-based insights.
        When a user asks something about stocks, always convert the company name to its ticker symbol.
        When using tools, provide ticker names as comma-separated values (e.g. 'AAPL,MSFT').

        For each user query:
        1. Identify which tool(s) are most suitable (AnalyzeSignals, ExpertSignals, or finnhub_news).
        2. Call and aggregate results for each ticker.
        3. Respond with a detailed and well-structured table, where each ticker is a row and the following columns are included.
        Each cell must provide clear, exhaustive insights, not generic comments. Always ensure:
        - Ticker: The stock or ETF symbol in uppercase (e.g. "AAPL").
        - Company Name: Full official company name (e.g. "Apple Inc").
        - Technical Analysis: Comprehensive summary including specific price levels, recent momentum indicators, relevant technical signals (such as RSI, MACD, CCI, trend direction), major patterns from the last week, and any observable volume or options anomalies. Clearly explain all signals referenced.
        - News & Sentiment: Detailed overview of the most important recent news (from the last 7 days), including major events, analyst upgrades/downgrades, earnings, product releases, or significant controversies. Explicitly note the sentiment (positive, negative, neutral), relevant catalysts, or anticipated volatility.
        - Final Recommendation: Precise trading stance ("Buy", "Hold", "Sell", "Short") and a brief rationale. Explicitly state a recommended entry price, based on current market data or technical setups, and ensure alignment with subsequent stop loss and take profit values.
        - Stop Loss: Specific price level(s) where an immediate exit is advised to limit risk. Justify the stop placement based on chart levels, volatility, or observed technical support.
        - Take Profit: Specific price level(s) to target for profit taking, clearly justified based on resistance levels, average move, or anticipated catalyst effect.

        Example Table Headers:
        | Ticker | Company | Technical Analysis | News & Sentiment | Final Recommendation | Entry Price | Stop Loss | Take Profit |

        OUTPUT FORMAT:
        Return ONLY the answer as a table in **pipe-delimited format** (Markdown table), with no extra text, no comments, no introductions, and no explanations. Always use the following headers in English:

        | Ticker | Company | Technical Analysis | News & Sentiment | Final Recommendation | Stop Loss | Take Profit |

        After the table, print the following disclaimer below as plain text (with NO extra newlines before or after):

        Disclaimer: The results are for informational purposes only and do not represent financial advice or guaranteed investment outcomes.

        The report should be complete and easy to parse programmatically or render using a table library in Python.
        Answer in the same language as the user's query and omit any markdown formatting or decoration outside of the table and disclaimer.
        """,
        client=GoogleClient(api_key=os.getenv("GOOGLE_API_KEY"), model="gemini-2.5-flash"),
        tools=[
            analyzer.analyze_signals,
            analyzer.expert_signals,
            analyzer.finnhub_news_analysis,
            analyzer.marketaux_news_analysis,
        ],
    )

    QUESTION = input("Ask me anything about stock market:\n").strip()
    if not QUESTION:
        logger.info("No question provided - using default fallback.")
        QUESTION = "Trading Intraday: apple, amazon or tesla?"

    logger.info("Running agent with question: %s", QUESTION)
    res = agent.run(QUESTION, tool_choice="auto")
    render_table_with_separators(res.text)
    logger.info("Done.")
