import os
import pandas as pd
import yfinance as yf
import warnings
from datetime import datetime, timedelta
import talib
import requests
from typing import Dict
from dotenv import load_dotenv
from datapizza.clients.google import GoogleClient
from datapizza.agents import Agent
from datapizza.tools import tool
from rich.console import Console
from rich.table import Table

warnings.simplefilter("ignore")
load_dotenv()


class MarketAnalyzer:
    """
    A comprehensive market analysis toolkit for AI agents.
    
    This class provides multiple analysis methods that can be exposed as LangChain tools
    for automated trading research, sector rotation analysis, technical signals detection,
    and news sentiment analysis.
    """
    
    def __init__(self):
        """Initialize the MarketAnalyzer with API credentials and stock universe."""
        self.finnhub_api_key = os.environ.get('FINNHUB_API_KEY')
        self.finnhub_base_url = "https://finnhub.io/api/v1/company-news"
        self.sector_etfs = self._initialize_sector_etfs()
        
    
    def _initialize_sector_etfs(self) -> Dict[str, str]:
        """Initialize sector ETF mappings."""
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
            "Utilities": "XLU"
        }
    
    def calculate_sector_rotation_signal(self) -> pd.DataFrame:
        """
        Calculate momentum signals for major market sectors using ETF data.

        This function retrieves the last 252 days of historical closing prices for a predefined set of sector ETFs,
        representing sectors like Technology, Financials, Energy, Healthcare, and more. It then computes the momentum
        for each sector, defined as the percentage change in closing price over the past approximately 21 trading days
        (about one month).

        Returns:
            pd.DataFrame: A DataFrame with columns 'ticker' (ETF ticker symbol), 'Sector' (sector name),
            and 'momentum' (percentage change over the past month).

        Use case:
            Intended for use in sector rotation strategies where an AI agent or trading algorithm needs to identify
            which sectors exhibit recent strength or weakness.
            The momentum values of ETFs serve as a proxy for the recent performance of their respective sectors.

        Notes:
            The momentum is calculated as ((close_today / close_21_days_ago) - 1) * 100.
        """
        rows = []
        
        for sector_name, etf_ticker in self.sector_etfs.items():
            try:
                data = yf.Ticker(etf_ticker).history(period="252d")
                momentum = (data['Close'][-1] / data['Close'][-21] - 1) * 100
                rows.append({
                    'ticker': etf_ticker,
                    'Sector': sector_name,
                    'momentum': momentum
                })
            except Exception:
                continue
                
        return pd.DataFrame(rows)
    
    @tool
    def analyze_signals(self, ticker_names: str) -> str:
        """
        Analyze a list of ticker symbols based on price gaps, volume spikes, options data, and sector momentum.

        This function takes a dictionary whose values are ticker symbols representing stocks to be analyzed.
        For each ticker it downloads historical price data (last 252 days) and calculates:
        - Gap size percentage between last open price and previous close
        - Volume spike compared to 10-day average volume
        It fetches company info including float shares, current price, and sector.

        The function also retrieves options data for the nearest expiry to compute, e.g.:
        - Total open interest and volume for calls and puts
        - Put/Call ratios (open interest and volume)
        - Days to expiry (DTE)

        It obtains sector momentum from the sector_rotation_signal() function results, linking each ticker's sector momentum.
        Input should be comma-separated ticker names like 'AAPL,MSFT'
        """
        results = []
        watchlist =  [n.strip() for n in ticker_names.split(',')]
        sector_momentum_df = self.calculate_sector_rotation_signal()
        
        for ticker in watchlist:
            try:
                data = yf.download(ticker, period="252d", interval="1d", progress=False)
                data = data.xs(ticker, axis=1, level='Ticker')
                
                last_open = data['Open'][-1]
                prev_close = data['Close'][-2]
                gap_size_pct = float(round((last_open - prev_close) / prev_close * 100, 2))
                
                avg_volume_10 = data['Volume'][-11:-1].mean()
                last_volume = data['Volume'][-1]
                volume_spike = last_volume > 4 * avg_volume_10
                
                info = yf.Ticker(ticker).info
                float_shares = info.get('floatShares') or info.get('float')
                price = info.get('regularMarketPrice') or last_open
                sector = info.get('sector')
                
                ticker_obj = yf.Ticker(ticker)
                expirations = ticker_obj.options
                
                if expirations:
                    expiry = expirations[0]
                    option_chain = ticker_obj.option_chain(expiry)
                    calls = option_chain.calls
                    puts = option_chain.puts
                    
                    total_call_oi = calls['openInterest'].sum()
                    total_put_oi = puts['openInterest'].sum()
                    total_call_volume = calls['volume'].sum()
                    total_put_volume = puts['volume'].sum()
                    
                    put_call_oi_ratio = total_put_oi / total_call_oi if total_call_oi > 0 else None
                    put_call_volume_ratio = total_put_volume / total_call_volume if total_call_volume > 0 else None
                    
                    expiry_date = datetime.strptime(expiry, '%Y-%m-%d').date()
                    today_date = datetime.today().date()
                    dte = (expiry_date - today_date).days
                else:
                    total_call_oi = total_put_oi = total_call_volume = total_put_volume = None
                    put_call_oi_ratio = put_call_volume_ratio = None
                    dte = None
                
                try:
                    if sector in sector_momentum_df['Sector'].values:
                        sector_momentum = sector_momentum_df.loc[
                            sector_momentum_df['Sector'] == sector, 'momentum'
                        ].iloc[0]
                    else:
                        sector_momentum = 0
                except (KeyError, IndexError):
                    sector_momentum = 0
                
                results.append({
                    'ticker': ticker,
                    'gap_size_pct': float(gap_size_pct),
                    'volume_spike': float(volume_spike),
                    'price': float(price),
                    'float_shares': float(float_shares) if float_shares else None,
                    'sector': sector,
                    'sector_momentum': round(float(sector_momentum),2),
                    'total_call_open_interest': int(total_call_oi),
                    'total_put_open_interest': int(total_put_oi),
                    'total_call_volume': int(total_call_volume),
                    'total_put_volume': int(total_put_volume),
                    'put_call_oi_ratio': round(float(put_call_oi_ratio),2),
                    'put_call_volume_ratio': round(float(put_call_volume_ratio),2),
                    'days_to_expiry': dte
                })
            except Exception:
                continue
                
        return str(results)
    
    @tool
    def expert_signals(self, ticker_names: str) -> str:
        """
        Generate a comprehensive set of expert-level technical trading signals including trend, momentum, volatility,
        volume, support/resistance, and candlestick pattern recognition for each ticker in the input dictionary.

        For each ticker, downloads 252 days of daily OHLCV data via yFinance, uses TA-Lib for indicators and candlestick pattern detection,
        and constructs a rich profile of signals typically used by professional traders for entry/exit decisions.

        Signals include:
        - Trend: SMA50, SMA200 crossover, ADX strength, Parabolic SAR position
        - Momentum: RSI levels, MACD crossovers, CCI, Stochastic oscillators, Williams %R
        - Volatility: ATR, Bollinger Bands levels/touches, Chaikin volatility
        - Volume: OBV trend, MFI extremes, Volume spikes
        - Support/Resistance: rolling min/max (50 days), price proximity/breaks
        - Candlestick patterns: bullish/bearish engulfing, hammer, shooting star, doji, morning/evening star, harami, piercing line,
            dark cloud cover, etc. (TA-Lib pattern functions included)
        - Additional: ROC, TRIX, Ultimate Oscillator signals
        Input should be comma-separated ticker names like 'AAPL,MSFT'
        """

        results = []
        watchlist =  [n.strip() for n in ticker_names.split(',')]
        for ticker in watchlist:
            try:
                data = yf.download(ticker, period="252d", interval="1d", progress=False)
                data = data.xs(ticker, axis=1, level='Ticker')
                
                if data.empty or len(data) < 200:
                    continue
                
                close = data['Close']
                high = data['High']
                low = data['Low']
                open_ = data['Open']
                volume = data['Volume']
                last_idx = -1
                
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
                
                patterns = {
                    'CDL_ENGULFING': talib.CDLENGULFING(open_, high, low, close),
                    'CDL_HAMMER': talib.CDLHAMMER(open_, high, low, close),
                    'CDL_SHOOTING_STAR': talib.CDLSHOOTINGSTAR(open_, high, low, close),
                    'CDL_DOJI': talib.CDLDOJI(open_, high, low, close),
                    'CDL_MORNING_STAR': talib.CDLMORNINGSTAR(open_, high, low, close),
                    'CDL_EVENING_STAR': talib.CDLEVENINGSTAR(open_, high, low, close),
                    'CDL_HARAMI': talib.CDLHARAMI(open_, high, low, close),
                    'CDL_PIERCING': talib.CDLPIERCING(open_, high, low, close),
                    'CDL_DARK_CLOUD_COVER': talib.CDLDARKCLOUDCOVER(open_, high, low, close),
                }
                
                def pattern_signal(pattern_array):
                    val = pattern_array[last_idx]
                    if val > 0:
                        return 'Bullish'
                    elif val < 0:
                        return 'Bearish'
                    else:
                        return 'None'
                
                bullish_trend = (
                    (close[last_idx] > sma50[last_idx]) and 
                    (close[last_idx] > sma200[last_idx]) and 
                    (adx[last_idx] > 25) and 
                    (close[last_idx] > sar[last_idx])
                )
                bearish_trend = (
                    (close[last_idx] < sma50[last_idx]) and 
                    (close[last_idx] < sma200[last_idx]) and 
                    (adx[last_idx] > 25) and 
                    (close[last_idx] < sar[last_idx])
                )
                
                rsi_signal = 'Neutral'
                if rsi[last_idx] < 30:
                    rsi_signal = 'Oversold'
                elif rsi[last_idx] > 70:
                    rsi_signal = 'Overbought'
                
                macd_signal = 'Neutral'
                macd_hist_prev = macdhist[last_idx - 1]
                macd_hist_curr = macdhist[last_idx]
                if macd_hist_prev < 0 < macd_hist_curr:
                    macd_signal = 'Bullish Crossover'
                elif macd_hist_prev > 0 > macd_hist_curr:
                    macd_signal = 'Bearish Crossover'
                
                cci_signal = 'Neutral'
                if cci[last_idx] < -100:
                    cci_signal = 'Oversold'
                elif cci[last_idx] > 100:
                    cci_signal = 'Overbought'
                
                stoch_signal = 'Neutral'
                if slowk[last_idx] < 20 and slowk[last_idx] > slowd[last_idx]:
                    stoch_signal = 'Bullish Crossover'
                elif slowk[last_idx] > 80 and slowk[last_idx] < slowd[last_idx]:
                    stoch_signal = 'Bearish Crossover'
                
                willr_signal = 'Neutral'
                if willr[last_idx] < -80:
                    willr_signal = 'Oversold'
                elif willr[last_idx] > -20:
                    willr_signal = 'Overbought'
                
                if close[last_idx] >= upperband[last_idx]:
                    boll_signal = 'Touch Upper Band'
                elif close[last_idx] <= lowerband[last_idx]:
                    boll_signal = 'Touch Lower Band'
                else:
                    boll_signal = 'Within Bands'
                
                vol_confirm = obv.diff().iloc[-3:].gt(0).all()
                
                mfi_signal = 'Neutral'
                if mfi[last_idx] < 20:
                    mfi_signal = 'Oversold'
                elif mfi[last_idx] > 80:
                    mfi_signal = 'Overbought'
                
                support = close.rolling(50).min().iloc[last_idx]
                resistance = close.rolling(50).max().iloc[last_idx]
                support_break = close[last_idx] < support
                resistance_break = close[last_idx] > resistance
                
                candlestick_signals = {key: pattern_signal(val) for key, val in patterns.items()}
                
                results.append({
                    'azienda': ticker,
                    'Bullish_Trend': bool(bullish_trend),
                    'Bearish_Trend': bool(bearish_trend),
                    'RSI_Signal': rsi_signal,
                    'MACD_Signal': macd_signal,
                    'CCI_Signal': cci_signal,
                    'Stochastic_Signal': stoch_signal,
                    'WilliamsR_Signal': willr_signal,
                    'Bollinger_Signal': boll_signal,
                    'Volume_Confirmation': bool(vol_confirm),
                    'MFI_Signal': mfi_signal,
                    'Support_Break': bool(support_break),
                    'Resistance_Break': bool(resistance_break),
                    'ATR': round(float(atr[last_idx]), 4),
                    'ROC': round(float(roc[last_idx]), 4),
                    'TRIX': round(float(trix[last_idx]), 4),
                    'Ultimate_Oscillator': round(float(ultosc[last_idx]), 4),
                    **candlestick_signals
                })
                
            except Exception as Error:
                print(Error)
                continue
        
        return str(results)
    
    @tool
    def finnhub_news_analysis(self, ticker_names: str) -> str:
        """
        Search online for recent news, sentiment, and buzz metrics using Finnhub API.
        
        Designed for AI agents to gather external market intelligence on tickers.
        Fetches recent news (last 3 days), analyzes sentiment
        
        Args:
            ticker_names:  ticker symbols.
            
        Returns:
                A string containing the article summaries.
        """
        summaries = {}
        
        end_date = datetime.now()
        start_date = end_date - timedelta(days=3)
        date_from = start_date.strftime('%Y-%m-%d')
        date_to = end_date.strftime('%Y-%m-%d')
        
        watchlist =  [n.strip() for n in ticker_names.split(',')]
        for ticker in watchlist:
            try:
                params = {
                    'symbol': ticker,
                    'from': date_from,
                    'to': date_to,
                    'token': self.finnhub_api_key
                }
                
                response = requests.get(self.finnhub_base_url, params=params, timeout=10)
                
                if response.status_code != 200:
                    raise Exception(f"API error: {response.status_code}")
                
                articles = response.json()
                summaries[ticker] = []
                
                articles_sorted = sorted(articles, key=lambda x: x.get('datetime', 0), reverse=True)
                
                for idx, article in enumerate(articles_sorted):
                    headline = article.get('headline', '')
                    summary = article.get('summary', '')
                    article_datetime = article.get('datetime', 0)
                    url = article.get('url', '')
                    
                    summaries[ticker].append({
                        'datetime': datetime.fromtimestamp(article_datetime).strftime('%Y-%m-%d %H:%M:%S'),
                        'headline': headline,
                        'summary': summary,
                        'url': url
                    }) 
                
            except Exception:
                summaries[ticker] = []
                continue
        
        return str(summaries)

def render_table_with_separators(response: str):
    """Create table."""
    lines = [line.strip() for line in response.splitlines() if line.strip().startswith('|')]
    if not lines:
        print("No table found in output.")
        return

    # Headers:
    header = [h.strip() for h in lines[0].split('|') if h.strip()]
    table = Table(title="Trading Report", show_lines=True)  # <--- show_lines!
    for col in header:
        table.add_column(col, style="bold cyan")

    # Righe dati (dopo header + separatore markdown)
    for row in lines[2:]:
        fields = [f.strip() for f in row.split('|') if f.strip()]
        if len(fields) == len(header):
            table.add_row(*fields)

    console = Console()
    console.print(table)

    # Disclaimer
    for line in response.splitlines():
        if line.lower().startswith("disclaimer:"):
            console.print(f"[bold yellow]{line}[/bold yellow]")

if __name__ == '__main__':
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
        client = GoogleClient(api_key=os.getenv("GOOGLE_API_KEY"), model="gemini-2.5-flash"),
        tools=[analyzer.analyze_signals, analyzer.expert_signals, analyzer.finnhub_news_analysis],
    )
    QUESTION = input('Ask me anything about stock market: \n')
    #QUESTION = "Trading Intraday: apple, amazon or tesla?"
    if not QUESTION:
        QUESTION="Trading Intraday: apppple, amzon or tesla?"
    res = agent.run(QUESTION, tool_choice="auto")
    render_table_with_separators(res.text)
