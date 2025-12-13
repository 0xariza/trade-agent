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
    
    Supports multiple exchanges:
    - binance, binanceus
    - kraken
    - kucoin
    - bybit
    - okx
    - coinbase
    """
    
    # Exchange-specific configurations
    EXCHANGE_CONFIGS = {
        'kraken': {
            'options': {
                'adjustForTimeDifference': True
            },
            'rateLimit': 1000
        },
        'kucoin': {
            'options': {
                'adjustForTimeDifference': True
            },
            'rateLimit': 2000
        },
        'binance': {
            'options': {
                'defaultType': 'spot'
            },
            'rateLimit': 1200
        },
        'binanceus': {
            'options': {
                'defaultType': 'spot'
            },
            'rateLimit': 1200
        },
        'bybit': {
            'options': {
                'defaultType': 'spot'
            },
            'rateLimit': 1000
        },
        'okx': {
            'options': {
                'defaultType': 'spot'
            },
            'rateLimit': 1000
        },
        'coinbase': {
            'rateLimit': 1000
        }
    }
    
    def __init__(self, exchange_id: str = 'binance', api_key: str = None, api_secret: str = None, passphrase: str = None):
        """
        Initialize market data provider.
        
        Args:
            exchange_id: Exchange name (binance, kraken, kucoin, etc.)
            api_key: Optional API key (not needed for public data)
            api_secret: Optional API secret (not needed for public data)
            passphrase: Optional passphrase (required for KuCoin API)
        """
        self.exchange_id = exchange_id.lower()
        
        # Get exchange class
        exchange_class = getattr(ccxt, self.exchange_id, None)
        if not exchange_class:
            raise ValueError(f"Exchange '{exchange_id}' not supported by CCXT")
        
        # Get exchange-specific config
        config = self.EXCHANGE_CONFIGS.get(self.exchange_id, {}).copy()
        
        # Add API credentials if provided
        if api_key and api_secret:
            config['apiKey'] = api_key
            config['secret'] = api_secret
            config['enableRateLimit'] = True
            
            # KuCoin requires passphrase
            if self.exchange_id == 'kucoin' and passphrase:
                config['password'] = passphrase
        
        # Initialize exchange
        self.exchange = exchange_class(config)
        
        logger.info(f"MarketDataProvider initialized with {self.exchange_id}")
        
        # Cache for BTC data (refreshed every 5 minutes)
        self._btc_cache: Dict[str, Any] = {}
        self._btc_cache_time: Optional[datetime] = None
        self._btc_cache_ttl = 300  # 5 minutes
        
        # Cache for Fear & Greed (refreshed every hour)
        self._fng_cache: Optional[Dict[str, Any]] = None
        self._fng_cache_time: Optional[datetime] = None
        
        # Performance: Market data cache (TTL-based)
        self._market_data_cache: Dict[str, Dict[str, Any]] = {}
        self._market_data_cache_times: Dict[str, datetime] = {}
        # Cache TTL from settings (default 60 seconds)
        try:
            from backend.config import settings
            self._market_data_cache_ttl = settings.market_data_cache_ttl_seconds
        except:
            self._market_data_cache_ttl = 60  # Default fallback
        
        # Performance: OHLCV cache (reduces API calls)
        self._ohlcv_cache: Dict[str, Dict[str, pd.DataFrame]] = {}
        self._ohlcv_cache_times: Dict[str, datetime] = {}
        self._ohlcv_cache_ttl = 30  # 30 second cache for OHLCV data
    
    @classmethod
    async def create_with_fallback(
        cls,
        primary_exchange: str,
        fallback_exchanges: List[str],
        api_credentials: Dict[str, Dict[str, str]] = None
    ) -> 'MarketDataProvider':
        """
        Create MarketDataProvider with automatic fallback.
        Tries primary exchange first, then fallbacks if it fails.
        
        Args:
            primary_exchange: Primary exchange to try
            fallback_exchanges: List of fallback exchanges to try in order
            api_credentials: Dict mapping exchange_id to {'api_key', 'api_secret', 'passphrase'}
        
        Returns:
            MarketDataProvider instance using the first working exchange
        
        Raises:
            Exception: If all exchanges fail
        """
        api_credentials = api_credentials or {}
        fallback_exchanges = fallback_exchanges or []
        all_exchanges = [primary_exchange] + fallback_exchanges
        
        last_error = None
        provider = None
        for exchange_id in all_exchanges:
            try:
                # Get credentials for this exchange
                creds = api_credentials.get(exchange_id, {})
                api_key = creds.get('api_key')
                api_secret = creds.get('api_secret')
                passphrase = creds.get('passphrase')
                
                # Create provider
                provider = cls(
                    exchange_id=exchange_id,
                    api_key=api_key,
                    api_secret=api_secret,
                    passphrase=passphrase
                )
                
                # Test connection with a simple fetch
                test_ohlcv = await provider.get_ohlcv("BTC/USDT", limit=1)
                if not test_ohlcv.empty:
                    logger.info(f"✓ Successfully connected to {exchange_id}")
                    return provider
                else:
                    logger.warning(f"✗ {exchange_id} returned empty data, trying next...")
                    if provider:
                        await provider.close()
                    provider = None
                    
            except Exception as e:
                last_error = e
                logger.warning(f"✗ {exchange_id} failed: {str(e)[:100]}, trying next...")
                if provider:
                    try:
                        await provider.close()
                    except:
                        pass
                provider = None
                continue
        
        # All exchanges failed
        error_msg = f"All exchanges failed. Last error: {last_error}"
        logger.error(error_msg)
        raise Exception(error_msg)
    
    async def test_connection(self, symbol: str = "BTC/USDT") -> bool:
        """Test if the exchange connection works."""
        try:
            ohlcv = await self.get_ohlcv(symbol, limit=1)
            return not ohlcv.empty
        except Exception as e:
            logger.error(f"Connection test failed: {e}")
            return False
        
    async def close(self):
        await self.exchange.close()

    async def get_ohlcv(self, symbol: str, timeframe: str = '1h', limit: int = 100) -> pd.DataFrame:
        """Fetch OHLCV data from the exchange with caching."""
        # Check cache first
        cache_key = f"{symbol}:{timeframe}:{limit}"
        now = datetime.now()
        
        if cache_key in self._ohlcv_cache:
            cache_time = self._ohlcv_cache_times.get(cache_key)
            if cache_time and (now - cache_time).total_seconds() < self._ohlcv_cache_ttl:
                logger.debug(f"Using cached OHLCV for {cache_key}")
                return self._ohlcv_cache[cache_key].copy()
        
        try:
            ohlcv = await self.exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            
            # Cache the result
            self._ohlcv_cache[cache_key] = df.copy()
            self._ohlcv_cache_times[cache_key] = now
            
            # Clean old cache entries (keep last 100)
            if len(self._ohlcv_cache) > 100:
                oldest_key = min(self._ohlcv_cache_times.items(), key=lambda x: x[1])[0]
                del self._ohlcv_cache[oldest_key]
                del self._ohlcv_cache_times[oldest_key]
            
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
        """
        Calculate all technical indicators.
        PERFORMANCE: Uses vectorized pandas operations for speed.
        """
        if df.empty or len(df) < 2:
            return df
        
        # PERFORMANCE: Work on copy to avoid modifying original
        df = df.copy()
        
        # Ensure DatetimeIndex
        if not isinstance(df.index, pd.DatetimeIndex):
            if 'timestamp' in df.columns:
                df = df.set_index('timestamp')
            else:
                df.index = pd.to_datetime(df.index)
        
        # ========== CORE INDICATORS ==========
        
        # PERFORMANCE: RSI calculation optimized
        # Use pandas_ta for better performance if available, otherwise manual
        try:
            # Try using pandas_ta for faster RSI calculation
            import pandas_ta as ta
            df['rsi'] = ta.rsi(df['close'], length=14)
        except:
            # Fallback to manual calculation
            delta = df['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14, min_periods=1).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14, min_periods=1).mean()
            rs = gain / (loss + 1e-10)  # Add small epsilon to avoid division by zero
            df['rsi'] = 100 - (100 / (1 + rs))
        
        # PERFORMANCE: MACD calculation optimized
        try:
            import pandas_ta as ta
            macd_data = ta.macd(df['close'], fast=12, slow=26, signal=9)
            if macd_data is not None and isinstance(macd_data, pd.DataFrame) and not macd_data.empty:
                # pandas_ta returns DataFrame with columns like MACD_12_26_9, MACDs_12_26_9, MACDh_12_26_9
                cols = macd_data.columns
                if len(cols) >= 1:
                    df['macd'] = macd_data.iloc[:, 0]
                if len(cols) >= 2:
                    df['macd_signal'] = macd_data.iloc[:, 1]
                if len(cols) >= 3:
                    df['macd_hist'] = macd_data.iloc[:, 2]
            else:
                # Fallback
                exp1 = df['close'].ewm(span=12, adjust=False).mean()
                exp2 = df['close'].ewm(span=26, adjust=False).mean()
                df['macd'] = exp1 - exp2
                df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
                df['macd_hist'] = df['macd'] - df['macd_signal']
        except Exception as e:
            # Fallback to manual calculation
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
        Optimized with caching and parallel timeframe fetching.
        """
        # Check cache first
        now = datetime.now()
        if symbol in self._market_data_cache:
            cache_time = self._market_data_cache_times.get(symbol)
            if cache_time and (now - cache_time).total_seconds() < self._market_data_cache_ttl:
                logger.debug(f"Using cached market snapshot for {symbol}")
                return self._market_data_cache[symbol].copy()
        
        timeframes = ['15m', '1h', '4h']
        
        # PERFORMANCE: Fetch BTC trend and Fear & Greed in parallel
        btc_task = self.get_btc_trend() if symbol != "BTC/USDT" else None
        fng_task = self.get_fear_greed_index()
        
        # Wait for both
        if btc_task:
            btc_trend, fear_greed = await asyncio.gather(btc_task, fng_task)
        else:
            fear_greed = await fng_task
            btc_trend = None
        
        snapshot = {
            "symbol": symbol,
            "last_updated": datetime.now().isoformat(),
            "btc_trend": btc_trend,
            "fear_greed": fear_greed,
            "timeframes": {}
        }

        # PERFORMANCE: Fetch all timeframes in parallel instead of sequentially
        timeframe_tasks = {
            tf: self.get_ohlcv(symbol, timeframe=tf, limit=100)
            for tf in timeframes
        }
        
        # Wait for all timeframes
        timeframe_results = await asyncio.gather(*timeframe_tasks.values(), return_exceptions=True)
        
        for tf, df in zip(timeframes, timeframe_results):
            if isinstance(df, Exception):
                logger.error(f"Error fetching {tf} for {symbol}: {df}")
                continue
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
                "price": round(float(latest['close']), 2),
                "change_pct": round(change_pct, 2),
                "change_24h": round(change_24h, 2),
                
                # Trend
                "trend": trend_signal,
                "trend_strength": trend_strength,
                "ema_alignment": "bullish" if float(latest['ema_9']) > float(latest['ema_21']) else "bearish",
                
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

        # Cache the snapshot
        self._market_data_cache[symbol] = snapshot.copy()
        self._market_data_cache_times[symbol] = now
        
        # Clean old cache entries (keep last 50 symbols)
        if len(self._market_data_cache) > 50:
            oldest_symbol = min(self._market_data_cache_times.items(), key=lambda x: x[1])[0]
            del self._market_data_cache[oldest_symbol]
            del self._market_data_cache_times[oldest_symbol]

        return snapshot
