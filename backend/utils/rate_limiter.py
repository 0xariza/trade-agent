"""
API Rate Limiter and Error Handler.

Features:
- Token bucket rate limiting
- Exponential backoff retry
- Circuit breaker pattern
- Request queuing
- Error classification and handling
"""

import asyncio
import time
import logging
from datetime import datetime, timedelta
from typing import Dict, Any, Callable, Optional, TypeVar, Awaitable
from dataclasses import dataclass, field
from functools import wraps
from enum import Enum
import random

logger = logging.getLogger(__name__)

T = TypeVar('T')


class ErrorSeverity(Enum):
    """Classification of API errors."""
    TRANSIENT = "transient"      # Retry immediately
    RATE_LIMITED = "rate_limited"  # Wait and retry
    AUTHENTICATION = "auth"       # Stop - needs intervention
    PERMANENT = "permanent"       # Don't retry
    UNKNOWN = "unknown"           # Retry with backoff


@dataclass
class RateLimitConfig:
    """Configuration for rate limiting."""
    requests_per_second: float = 10.0
    requests_per_minute: float = 600.0
    burst_limit: int = 20
    
    # Backoff settings
    initial_backoff_ms: int = 100
    max_backoff_ms: int = 60000
    backoff_multiplier: float = 2.0
    max_retries: int = 5
    
    # Circuit breaker
    failure_threshold: int = 5
    recovery_timeout_seconds: int = 30


@dataclass
class RequestStats:
    """Statistics for API requests."""
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    retried_requests: int = 0
    rate_limited_requests: int = 0
    
    avg_response_time_ms: float = 0.0
    last_request_time: Optional[datetime] = None
    
    # Errors
    errors_by_type: Dict[str, int] = field(default_factory=dict)


class TokenBucket:
    """
    Token bucket rate limiter.
    
    Allows bursting while maintaining average rate.
    """
    
    def __init__(
        self,
        rate: float,  # Tokens per second
        capacity: int  # Maximum tokens (burst capacity)
    ):
        self.rate = rate
        self.capacity = capacity
        self.tokens = capacity
        self.last_update = time.monotonic()
        self._lock = asyncio.Lock()
    
    async def acquire(self, tokens: int = 1) -> float:
        """
        Acquire tokens. Returns wait time if needed.
        
        Args:
            tokens: Number of tokens to acquire
            
        Returns:
            Time to wait in seconds (0 if immediate)
        """
        async with self._lock:
            now = time.monotonic()
            
            # Add tokens based on time passed
            elapsed = now - self.last_update
            self.tokens = min(
                self.capacity,
                self.tokens + elapsed * self.rate
            )
            self.last_update = now
            
            if self.tokens >= tokens:
                self.tokens -= tokens
                return 0.0
            
            # Calculate wait time
            wait_time = (tokens - self.tokens) / self.rate
            return wait_time
    
    async def wait_for_token(self, tokens: int = 1):
        """Wait until tokens are available."""
        wait_time = await self.acquire(tokens)
        if wait_time > 0:
            await asyncio.sleep(wait_time)


class CircuitBreaker:
    """
    Circuit breaker pattern for failing APIs.
    
    States:
    - CLOSED: Normal operation
    - OPEN: Blocking all requests
    - HALF_OPEN: Testing recovery
    """
    
    class State(Enum):
        CLOSED = "closed"
        OPEN = "open"
        HALF_OPEN = "half_open"
    
    def __init__(
        self,
        failure_threshold: int = 5,
        recovery_timeout: int = 30,
        half_open_requests: int = 3
    ):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.half_open_requests = half_open_requests
        
        self.state = self.State.CLOSED
        self.failures = 0
        self.successes = 0
        self.last_failure_time: Optional[datetime] = None
        self.half_open_successes = 0
    
    def record_success(self):
        """Record a successful request."""
        if self.state == self.State.HALF_OPEN:
            self.half_open_successes += 1
            if self.half_open_successes >= self.half_open_requests:
                self._close()
        else:
            self.failures = 0
        
        self.successes += 1
    
    def record_failure(self):
        """Record a failed request."""
        self.failures += 1
        self.last_failure_time = datetime.now()
        
        if self.state == self.State.HALF_OPEN:
            self._open()
        elif self.failures >= self.failure_threshold:
            self._open()
    
    def can_execute(self) -> bool:
        """Check if request can proceed."""
        if self.state == self.State.CLOSED:
            return True
        
        if self.state == self.State.OPEN:
            # Check if recovery timeout has passed
            if self.last_failure_time:
                elapsed = (datetime.now() - self.last_failure_time).total_seconds()
                if elapsed >= self.recovery_timeout:
                    self._half_open()
                    return True
            return False
        
        # HALF_OPEN - allow limited requests
        return True
    
    def _open(self):
        """Open the circuit (block requests)."""
        self.state = self.State.OPEN
        logger.warning("Circuit breaker OPENED - blocking requests")
    
    def _half_open(self):
        """Try partial recovery."""
        self.state = self.State.HALF_OPEN
        self.half_open_successes = 0
        logger.info("Circuit breaker HALF-OPEN - testing recovery")
    
    def _close(self):
        """Resume normal operation."""
        self.state = self.State.CLOSED
        self.failures = 0
        self.half_open_successes = 0
        logger.info("Circuit breaker CLOSED - normal operation resumed")
    
    def get_status(self) -> Dict[str, Any]:
        return {
            'state': self.state.value,
            'failures': self.failures,
            'successes': self.successes,
            'last_failure': self.last_failure_time.isoformat() if self.last_failure_time else None
        }


class APIRateLimiter:
    """
    Complete rate limiter with retry logic and circuit breaker.
    """
    
    def __init__(self, config: RateLimitConfig = None):
        self.config = config or RateLimitConfig()
        
        # Token buckets for different time windows
        self.per_second_bucket = TokenBucket(
            rate=self.config.requests_per_second,
            capacity=self.config.burst_limit
        )
        self.per_minute_bucket = TokenBucket(
            rate=self.config.requests_per_minute / 60,
            capacity=int(self.config.requests_per_minute / 10)
        )
        
        # Circuit breaker
        self.circuit = CircuitBreaker(
            failure_threshold=self.config.failure_threshold,
            recovery_timeout=self.config.recovery_timeout_seconds
        )
        
        # Statistics
        self.stats = RequestStats()
        
        logger.info(
            f"APIRateLimiter initialized. "
            f"Rate: {self.config.requests_per_second}/sec, "
            f"Burst: {self.config.burst_limit}"
        )
    
    async def acquire(self) -> bool:
        """
        Acquire permission to make a request.
        
        Returns:
            True if request can proceed, False if blocked
        """
        # Check circuit breaker
        if not self.circuit.can_execute():
            logger.warning("Request blocked by circuit breaker")
            return False
        
        # Wait for rate limit tokens
        await self.per_second_bucket.wait_for_token()
        await self.per_minute_bucket.wait_for_token()
        
        return True
    
    async def execute_with_retry(
        self,
        func: Callable[..., Awaitable[T]],
        *args,
        **kwargs
    ) -> T:
        """
        Execute a function with retry logic.
        
        Args:
            func: Async function to execute
            *args, **kwargs: Arguments to pass to function
            
        Returns:
            Result of function call
            
        Raises:
            Exception: If all retries exhausted
        """
        last_error = None
        backoff_ms = self.config.initial_backoff_ms
        
        for attempt in range(self.config.max_retries):
            try:
                # Acquire rate limit permission
                if not await self.acquire():
                    raise Exception("Circuit breaker open")
                
                # Record attempt
                start_time = time.monotonic()
                self.stats.total_requests += 1
                
                # Execute
                result = await func(*args, **kwargs)
                
                # Record success
                elapsed_ms = (time.monotonic() - start_time) * 1000
                self._record_success(elapsed_ms)
                
                return result
                
            except Exception as e:
                last_error = e
                severity = self._classify_error(e)
                self._record_failure(e, severity)
                
                # Don't retry permanent errors
                if severity == ErrorSeverity.PERMANENT:
                    raise
                
                # Don't retry auth errors
                if severity == ErrorSeverity.AUTHENTICATION:
                    raise
                
                # Calculate backoff
                if severity == ErrorSeverity.RATE_LIMITED:
                    # Wait longer for rate limits
                    wait_time = min(backoff_ms * 2, self.config.max_backoff_ms)
                else:
                    wait_time = backoff_ms
                
                # Add jitter
                jitter = random.uniform(0.5, 1.5)
                wait_time = wait_time * jitter
                
                if attempt < self.config.max_retries - 1:
                    self.stats.retried_requests += 1
                    logger.warning(
                        f"Request failed (attempt {attempt + 1}): {e}. "
                        f"Retrying in {wait_time:.0f}ms"
                    )
                    await asyncio.sleep(wait_time / 1000)
                    backoff_ms = min(
                        backoff_ms * self.config.backoff_multiplier,
                        self.config.max_backoff_ms
                    )
        
        # All retries exhausted
        raise last_error
    
    def _classify_error(self, error: Exception) -> ErrorSeverity:
        """Classify an error to determine retry behavior."""
        error_msg = str(error).lower()
        
        # Rate limiting
        if any(x in error_msg for x in ['rate limit', 'too many', '429', 'throttl']):
            return ErrorSeverity.RATE_LIMITED
        
        # Authentication
        if any(x in error_msg for x in ['auth', 'api key', 'invalid key', '401', '403']):
            return ErrorSeverity.AUTHENTICATION
        
        # Network/transient
        if any(x in error_msg for x in ['timeout', 'connection', 'network', '502', '503', '504']):
            return ErrorSeverity.TRANSIENT
        
        # Permanent
        if any(x in error_msg for x in ['invalid', 'not found', '400', '404']):
            return ErrorSeverity.PERMANENT
        
        return ErrorSeverity.UNKNOWN
    
    def _record_success(self, elapsed_ms: float):
        """Record successful request."""
        self.stats.successful_requests += 1
        self.stats.last_request_time = datetime.now()
        
        # Update average response time
        total = self.stats.successful_requests
        self.stats.avg_response_time_ms = (
            (self.stats.avg_response_time_ms * (total - 1) + elapsed_ms) / total
        )
        
        self.circuit.record_success()
    
    def _record_failure(self, error: Exception, severity: ErrorSeverity):
        """Record failed request."""
        self.stats.failed_requests += 1
        
        error_type = type(error).__name__
        self.stats.errors_by_type[error_type] = (
            self.stats.errors_by_type.get(error_type, 0) + 1
        )
        
        if severity == ErrorSeverity.RATE_LIMITED:
            self.stats.rate_limited_requests += 1
        
        self.circuit.record_failure()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get rate limiter statistics."""
        return {
            'total_requests': self.stats.total_requests,
            'successful': self.stats.successful_requests,
            'failed': self.stats.failed_requests,
            'retried': self.stats.retried_requests,
            'rate_limited': self.stats.rate_limited_requests,
            'avg_response_ms': round(self.stats.avg_response_time_ms, 2),
            'circuit_breaker': self.circuit.get_status(),
            'errors': self.stats.errors_by_type
        }


def with_rate_limit(limiter: APIRateLimiter):
    """Decorator to apply rate limiting to async functions."""
    def decorator(func: Callable[..., Awaitable[T]]) -> Callable[..., Awaitable[T]]:
        @wraps(func)
        async def wrapper(*args, **kwargs) -> T:
            return await limiter.execute_with_retry(func, *args, **kwargs)
        return wrapper
    return decorator


# Default limiters for common APIs
class APILimiters:
    """Pre-configured rate limiters for common APIs."""
    
    @staticmethod
    def binance() -> APIRateLimiter:
        """Binance API limits."""
        return APIRateLimiter(RateLimitConfig(
            requests_per_second=10,
            requests_per_minute=1200,
            burst_limit=20,
            max_retries=3
        ))
    
    @staticmethod
    def gemini_api() -> APIRateLimiter:
        """Google Gemini API limits."""
        return APIRateLimiter(RateLimitConfig(
            requests_per_second=2,  # Conservative
            requests_per_minute=60,
            burst_limit=5,
            max_retries=3,
            initial_backoff_ms=1000
        ))
    
    @staticmethod
    def crypto_news() -> APIRateLimiter:
        """Crypto news API limits."""
        return APIRateLimiter(RateLimitConfig(
            requests_per_second=1,
            requests_per_minute=30,
            burst_limit=3,
            max_retries=2
        ))


