"""
Market Data Provider - Enhanced with Pro-Level Indicators.

Indicators included:
- Core: RSI, MACD, Bollinger Bands, ADX, ATR, EMA (9/21/50)
- Volume: OBV with trend detection
- Crypto-specific: BTC Correlation, Funding Rate, Fear & Greed

Removed/Simplified:
- Ichimoku: Replaced with simple cloud position signal
- Stochastic RSI: Redundant with RSI, simplified
- VWAP: Less useful for swing trading
"""

import ccxt.async_support as ccxt
import pandas as pd
import pandas_ta as ta
import aiohttp
import asyncio
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional
import logging

logger = logging.getLogger(__name__)


class MarketDataProvider:
    """
    Fetches real-time market data and calculates technical indicators.
    Enhanced with crypto-specific signals.
    """
    
    def __init__(self, exchange_id: str = 'binance'):
        self.exchange_id = exchange_id
        self.exchange = getattr(ccxt, exchange_id)()
        
        # Cache for BTC data (refreshed every 5 minutes)
        self._btc_cache: Dict[str, Any] = {}
        self._btc_cache_time: Optional[datetime] = None
        self._btc_cache_ttl = 300  # 5 minutes
        
        # Cache for Fear & Greed (refreshed every hour)
        self._fng_cache: Optional[Dict[str, Any]] = None
        self._fng_cache_time: Optional[datetime] = None
        
    async def close(self):
        await self.exchange.close()

    async def get_ohlcv(self, symbol: str, timeframe: str = '1h', limit: int = 100) -> pd.DataFrame:
        """Fetch OHLCV data from the exchange."""
        try:
            ohlcv = await self.exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            return df
        except Exception as e:
            logger.error(f"Error fetching data for {symbol}: {e}")
            return pd.DataFrame()

    async def get_btc_trend(self) -> Dict[str, Any]:
        """
        Get BTC trend data. CRITICAL for alt trading.
        Never buy alts when BTC is dumping.
        """
        # Check cache
        if self._btc_cache_time and (datetime.now() - self._btc_cache_time).seconds < self._btc_cache_ttl:
            return self._btc_cache
        
        try:
            df = await self.get_ohlcv("BTC/USDT", "1h", 50)
            if df.empty:
                return {"trend": "unknown", "change_1h": 0, "change_24h": 0}
            
            # Ensure proper index
            if not isinstance(df.index, pd.DatetimeIndex):
                df.set_index('timestamp', inplace=True)
            
            latest = df.iloc[-1]['close']
            
            # Calculate changes
            change_1h = ((latest - df.iloc[-2]['close']) / df.iloc[-2]['close']) * 100 if len(df) > 1 else 0
            change_24h = ((latest - df.iloc[-24]['close']) / df.iloc[-24]['close']) * 100 if len(df) > 24 else 0
            
            # Calculate EMAs
            df['ema_9'] = df['close'].ewm(span=9, adjust=False).mean()
            df['ema_21'] = df['close'].ewm(span=21, adjust=False).mean()
            
            # Calculate RSI
            delta = df['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            
            btc_rsi = rsi.iloc[-1]
            ema_bullish = df['ema_9'].iloc[-1] > df['ema_21'].iloc[-1]
            
            # Determine trend
            if change_24h > 3 and ema_bullish:
                trend = "strong_bullish"
            elif change_24h > 0 and ema_bullish:
                trend = "bullish"
            elif change_24h < -3 and not ema_bullish:
                trend = "strong_bearish"
            elif change_24h < 0 and not ema_bullish:
                trend = "bearish"
            else:
                trend = "neutral"
            
            result = {
                "trend": trend,
                "price": latest,
                "change_1h": round(change_1h, 2),
                "change_24h": round(change_24h, 2),
                "rsi": round(btc_rsi, 1),
                "ema_bullish": ema_bullish,
                "is_safe_for_alts": trend not in ["bearish", "strong_bearish"]
            }
            
            # Update cache
            self._btc_cache = result
            self._btc_cache_time = datetime.now()
            
            return result
            
        except Exception as e:
            logger.error(f"Error getting BTC trend: {e}")
            return {"trend": "unknown", "change_1h": 0, "change_24h": 0, "is_safe_for_alts": True}

    async def get_funding_rate(self, symbol: str) -> Dict[str, Any]:
        """
        Get funding rate for perpetual futures.
        Positive = longs pay shorts (crowded long)
        Negative = shorts pay longs (crowded short)
        """
        try:
            # Try to use Binance futures for funding rate
            futures_exchange = ccxt.binance({
                'options': {'defaultType': 'future'}
            })
            
            # Convert spot symbol to futures
            futures_symbol = symbol.replace("/", "")
            
            funding = await futures_exchange.fetch_funding_rate(futures_symbol)
            await futures_exchange.close()
            
            rate = funding.get('fundingRate', 0) * 100  # Convert to percentage
            
            # Interpret funding rate
            if rate > 0.05:
                sentiment = "extreme_long"  # Crowded long, bearish signal
            elif rate > 0.01:
                sentiment = "bullish"
            elif rate < -0.05:
                sentiment = "extreme_short"  # Crowded short, bullish signal
            elif rate < -0.01:
                sentiment = "bearish"
            else:
                sentiment = "neutral"
            
            return {
                "rate": round(rate, 4),
                "sentiment": sentiment,
                "interpretation": "Crowded longs (bearish)" if rate > 0.03 else "Crowded shorts (bullish)" if rate < -0.03 else "Normal"
            }
            
        except Exception as e:
            logger.debug(f"Funding rate not available for {symbol}: {e}")
            return {"rate": 0, "sentiment": "unknown", "interpretation": "N/A"}

    async def get_fear_greed_index(self) -> Dict[str, Any]:
        """
        Get Crypto Fear & Greed Index.
        0-25: Extreme Fear (buy signal)
        25-45: Fear
        45-55: Neutral
        55-75: Greed
        75-100: Extreme Greed (sell signal)
        """
        # Check cache (1 hour TTL)
        if self._fng_cache_time and (datetime.now() - self._fng_cache_time).seconds < 3600:
            return self._fng_cache
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get("https://api.alternative.me/fng/?limit=1") as response:
                    if response.status == 200:
                        data = await response.json()
                        fng_data = data.get('data', [{}])[0]
                        
                        value = int(fng_data.get('value', 50))
                        classification = fng_data.get('value_classification', 'Neutral')
                        
                        # Trading signal based on FNG
                        if value <= 25:
                            signal = "strong_buy"
                            interpretation = "Extreme fear = buying opportunity"
                        elif value <= 40:
                            signal = "buy"
                            interpretation = "Fear in market, consider buying"
                        elif value >= 75:
                            signal = "strong_sell"
                            interpretation = "Extreme greed = take profits"
                        elif value >= 60:
                            signal = "sell"
                            interpretation = "Greed in market, be cautious"
                        else:
                            signal = "neutral"
                            interpretation = "Market sentiment neutral"
                        
                        result = {
                            "value": value,
                            "classification": classification,
                            "signal": signal,
                            "interpretation": interpretation
                        }
                        
                        self._fng_cache = result
                        self._fng_cache_time = datetime.now()
                        
                        return result
        except Exception as e:
            logger.debug(f"Fear & Greed API error: {e}")
        
        return {"value": 50, "classification": "Neutral", "signal": "neutral", "interpretation": "Data unavailable"}

    def _calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate all technical indicators."""
        if df.empty:
            return df
        
        # Ensure DatetimeIndex
        if not isinstance(df.index, pd.DatetimeIndex):
            df = df.set_index('timestamp')
        
        # ========== CORE INDICATORS ==========
        
        # RSI (14)
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['rsi'] = 100 - (100 / (1 + rs))
        
        # MACD (12, 26, 9)
        exp1 = df['close'].ewm(span=12, adjust=False).mean()
        exp2 = df['close'].ewm(span=26, adjust=False).mean()
        df['macd'] = exp1 - exp2
        df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
        df['macd_hist'] = df['macd'] - df['macd_signal']
        
        # Bollinger Bands (20, 2)
        df['bb_mid'] = df['close'].rolling(window=20).mean()
        bb_std = df['close'].rolling(window=20).std()
        df['bb_upper'] = df['bb_mid'] + (bb_std * 2)
        df['bb_lower'] = df['bb_mid'] - (bb_std * 2)
        df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_mid'] * 100
        
        # EMAs (9, 21, 50)
        df['ema_9'] = df['close'].ewm(span=9, adjust=False).mean()
        df['ema_21'] = df['close'].ewm(span=21, adjust=False).mean()
        df['ema_50'] = df['close'].ewm(span=50, adjust=False).mean()
        
        # ========== TREND STRENGTH ==========
        
        # ADX (14)
        try:
            adx_df = df.ta.adx(length=14)
            if adx_df is not None and not adx_df.empty:
                df['adx'] = adx_df['ADX_14']
                df['di_plus'] = adx_df['DMP_14']
                df['di_minus'] = adx_df['DMN_14']
            else:
                df['adx'] = 0
                df['di_plus'] = 0
                df['di_minus'] = 0
        except:
            df['adx'] = 0
            df['di_plus'] = 0
            df['di_minus'] = 0
        
        # ATR (14) - for position sizing
        try:
            df['atr'] = df.ta.atr(length=14)
        except:
            df['atr'] = df['close'] * 0.02  # Fallback 2%
        
        # ========== VOLUME ANALYSIS ==========
        
        # OBV with trend
        try:
            df['obv'] = df.ta.obv()
            df['obv_ema'] = df['obv'].ewm(span=20, adjust=False).mean()
        except:
            df['obv'] = 0
            df['obv_ema'] = 0
        
        # Volume SMA for comparison
        df['volume_sma'] = df['volume'].rolling(window=20).mean()
        
        # ========== MOMENTUM ==========
        
        # Stochastic (simplified - just K value)
        low_14 = df['low'].rolling(window=14).min()
        high_14 = df['high'].rolling(window=14).max()
        df['stoch_k'] = ((df['close'] - low_14) / (high_14 - low_14)) * 100
        
        return df

    def _get_trend_signal(self, row: pd.Series) -> str:
        """Determine overall trend signal from indicators."""
        signals = []
        
        # EMA alignment
        if row['ema_9'] > row['ema_21'] > row['ema_50']:
            signals.append(1)  # Strong bullish
        elif row['ema_9'] > row['ema_21']:
            signals.append(0.5)  # Bullish
        elif row['ema_9'] < row['ema_21'] < row['ema_50']:
            signals.append(-1)  # Strong bearish
        elif row['ema_9'] < row['ema_21']:
            signals.append(-0.5)  # Bearish
        else:
            signals.append(0)
        
        # MACD
        if row['macd'] > row['macd_signal'] and row['macd_hist'] > 0:
            signals.append(1)
        elif row['macd'] < row['macd_signal'] and row['macd_hist'] < 0:
            signals.append(-1)
        else:
            signals.append(0)
        
        # ADX + DI
        if row['adx'] > 25:
            if row['di_plus'] > row['di_minus']:
                signals.append(1)
            else:
                signals.append(-1)
        else:
            signals.append(0)
        
        avg_signal = sum(signals) / len(signals)
        
        if avg_signal > 0.6:
            return "strong_bullish"
        elif avg_signal > 0.2:
            return "bullish"
        elif avg_signal < -0.6:
            return "strong_bearish"
        elif avg_signal < -0.2:
            return "bearish"
        else:
            return "neutral"

    def _get_momentum_signal(self, row: pd.Series) -> Dict[str, Any]:
        """Get momentum signals."""
        rsi = row.get('rsi', 50)
        stoch = row.get('stoch_k', 50)
        
        # RSI zones
        if rsi < 25:
            rsi_signal = "extreme_oversold"
        elif rsi < 35:
            rsi_signal = "oversold"
        elif rsi > 75:
            rsi_signal = "extreme_overbought"
        elif rsi > 65:
            rsi_signal = "overbought"
        else:
            rsi_signal = "neutral"
        
        # Combined momentum
        if rsi < 30 and stoch < 20:
            momentum = "strong_buy"
        elif rsi < 40 and stoch < 30:
            momentum = "buy"
        elif rsi > 70 and stoch > 80:
            momentum = "strong_sell"
        elif rsi > 60 and stoch > 70:
            momentum = "sell"
        else:
            momentum = "neutral"
        
        return {
            "rsi": round(rsi, 1),
            "rsi_signal": rsi_signal,
            "stoch_k": round(stoch, 1),
            "momentum_signal": momentum
        }

    def _get_volume_signal(self, row: pd.Series) -> Dict[str, Any]:
        """Get volume analysis."""
        obv = row.get('obv', 0)
        obv_ema = row.get('obv_ema', 0)
        volume = row.get('volume', 0)
        volume_sma = row.get('volume_sma', 1)
        
        # OBV trend
        if obv > obv_ema * 1.05:
            obv_trend = "rising"
        elif obv < obv_ema * 0.95:
            obv_trend = "falling"
        else:
            obv_trend = "flat"
        
        # Volume vs average
        volume_ratio = volume / volume_sma if volume_sma > 0 else 1
        if volume_ratio > 2:
            volume_status = "very_high"
        elif volume_ratio > 1.5:
            volume_status = "high"
        elif volume_ratio < 0.5:
            volume_status = "very_low"
        elif volume_ratio < 0.75:
            volume_status = "low"
        else:
            volume_status = "normal"
        
        return {
            "obv_trend": obv_trend,
            "volume_status": volume_status,
            "volume_ratio": round(volume_ratio, 2),
            "confirms_trend": obv_trend == "rising"  # True if volume confirms bullish move
        }

    async def get_market_snapshot(self, symbol: str) -> Dict[str, Any]:
        """
        Get comprehensive market snapshot with all indicators.
        """
        timeframes = ['15m', '1h', '4h']
        
        # Get BTC trend first (critical for alts)
        btc_trend = await self.get_btc_trend() if symbol != "BTC/USDT" else None
        
        # Get Fear & Greed
        fear_greed = await self.get_fear_greed_index()
        
        snapshot = {
            "symbol": symbol,
            "last_updated": datetime.now().isoformat(),
            "btc_trend": btc_trend,
            "fear_greed": fear_greed,
            "timeframes": {}
        }

        for tf in timeframes:
            df = await self.get_ohlcv(symbol, timeframe=tf, limit=100)
            if df.empty:
                continue
            
            # Calculate all indicators
            df = self._calculate_indicators(df)
            
            latest = df.iloc[-1]
            prev = df.iloc[-2] if len(df) > 1 else latest
            
            # Calculate changes
            change_pct = ((latest['close'] - prev['close']) / prev['close']) * 100
            
            # Get 24h change
            lookback = {'15m': 96, '1h': 24, '4h': 6}.get(tf, 24)
            if len(df) >= lookback:
                price_ago = df.iloc[-lookback]['close']
                change_24h = ((latest['close'] - price_ago) / price_ago) * 100
            else:
                change_24h = change_pct
            
            # Get signals
            trend_signal = self._get_trend_signal(latest)
            momentum = self._get_momentum_signal(latest)
            volume = self._get_volume_signal(latest)
            
            # Bollinger Band position
            if latest['close'] > latest['bb_upper']:
                bb_position = "above_upper"
            elif latest['close'] < latest['bb_lower']:
                bb_position = "below_lower"
            elif latest['close'] > latest['bb_mid']:
                bb_position = "upper_half"
            else:
                bb_position = "lower_half"
            
            # ADX interpretation
            adx_val = latest['adx'] if pd.notna(latest['adx']) else 0
            if adx_val > 40:
                trend_strength = "very_strong"
            elif adx_val > 25:
                trend_strength = "strong"
            elif adx_val > 15:
                trend_strength = "weak"
            else:
                trend_strength = "no_trend"
            
            snapshot["timeframes"][tf] = {
                "price": round(latest['close'], 2),
                "change_pct": round(change_pct, 2),
                "change_24h": round(change_24h, 2),
                
                # Trend
                "trend": trend_signal,
                "trend_strength": trend_strength,
                "ema_alignment": "bullish" if latest['ema_9'] > latest['ema_21'] else "bearish",
                
                # Momentum
                "momentum": momentum,
                
                # Volume
                "volume": volume,
                
                # Indicators (for detailed analysis)
                "indicators": {
                    "rsi": round(latest['rsi'], 1) if pd.notna(latest['rsi']) else 50,
                    "macd": "bullish" if latest['macd'] > latest['macd_signal'] else "bearish",
                    "macd_hist": round(latest['macd_hist'], 4) if pd.notna(latest['macd_hist']) else 0,
                    "adx": round(adx_val, 1),
                    "atr": round(latest['atr'], 2) if pd.notna(latest['atr']) else 0,
                    "bb_position": bb_position,
                    "bb_width": round(latest['bb_width'], 1) if pd.notna(latest['bb_width']) else 0,
                    "ema_9": round(latest['ema_9'], 2),
                    "ema_21": round(latest['ema_21'], 2),
                    "ema_50": round(latest['ema_50'], 2),
                    "stoch_k": round(latest['stoch_k'], 1) if pd.notna(latest['stoch_k']) else 50,
                    "obv_trend": volume['obv_trend']
                }
            }
            
            # Set main price from 1h
            if tf == '1h':
                snapshot['price'] = round(latest['close'], 2)
                snapshot['change_24h'] = round(change_24h, 2)
                snapshot['volume'] = latest['volume']

        # Add funding rate for futures
        try:
            funding = await self.get_funding_rate(symbol)
            snapshot['funding_rate'] = funding
        except:
            snapshot['funding_rate'] = {"rate": 0, "sentiment": "unknown"}

        # Add BTC safety check for alts
        if btc_trend:
            snapshot['btc_safe'] = btc_trend.get('is_safe_for_alts', True)
            if not snapshot['btc_safe']:
                snapshot['warning'] = "⚠️ BTC is bearish - avoid buying alts"

        return snapshot
