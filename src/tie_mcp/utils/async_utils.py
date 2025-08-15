"""
Async utilities for TIE MCP Server
"""

import asyncio
import functools
import logging
import threading
from collections.abc import Awaitable, Callable
from concurrent.futures import ThreadPoolExecutor
from typing import Any, TypeVar

logger = logging.getLogger(__name__)

T = TypeVar("T")

# Global thread pool for CPU-bound tasks
_thread_pool = None
_thread_pool_lock = threading.Lock()


def get_thread_pool() -> ThreadPoolExecutor:
    """Get or create the global thread pool"""
    global _thread_pool

    if _thread_pool is None:
        with _thread_pool_lock:
            if _thread_pool is None:
                _thread_pool = ThreadPoolExecutor(
                    max_workers=4,  # Adjust based on your needs
                    thread_name_prefix="tie_mcp_worker",
                )

    return _thread_pool


async def run_in_thread(func: Callable[..., T], *args, **kwargs) -> T:
    """
    Run a synchronous function in a thread pool

    Args:
        func: Function to run
        *args: Function arguments
        **kwargs: Function keyword arguments

    Returns:
        Function result
    """
    loop = asyncio.get_event_loop()
    executor = get_thread_pool()

    try:
        if kwargs:
            # If we have kwargs, we need to use partial
            partial_func = functools.partial(func, **kwargs)
            result = await loop.run_in_executor(executor, partial_func, *args)
        else:
            result = await loop.run_in_executor(executor, func, *args)

        return result

    except Exception as e:
        logger.error(f"Error running function {func.__name__} in thread: {e}")
        raise


async def run_with_timeout(
    coro: Awaitable[T],
    timeout_seconds: float,
    timeout_message: str = "Operation timed out",
) -> T:
    """
    Run a coroutine with a timeout

    Args:
        coro: Coroutine to run
        timeout_seconds: Timeout in seconds
        timeout_message: Message for timeout exception

    Returns:
        Coroutine result

    Raises:
        asyncio.TimeoutError: If operation times out
    """
    try:
        return await asyncio.wait_for(coro, timeout=timeout_seconds)
    except asyncio.TimeoutError:
        logger.error(f"Operation timed out after {timeout_seconds} seconds")
        raise asyncio.TimeoutError(timeout_message) from None


async def gather_with_concurrency(coroutines: list[Awaitable[T]], max_concurrency: int = 10) -> list[T]:
    """
    Run coroutines with limited concurrency

    Args:
        coroutines: List of coroutines to run
        max_concurrency: Maximum number of concurrent operations

    Returns:
        List of results in the same order as input coroutines
    """
    semaphore = asyncio.Semaphore(max_concurrency)

    async def run_with_semaphore(coro: Awaitable[T]) -> T:
        async with semaphore:
            return await coro

    limited_coroutines = [run_with_semaphore(coro) for coro in coroutines]
    return await asyncio.gather(*limited_coroutines)


class AsyncContextManager:
    """Base class for async context managers"""

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        pass


class AsyncLock:
    """Async lock with timeout support"""

    def __init__(self, timeout: float = 30.0):
        self._lock = asyncio.Lock()
        self._timeout = timeout

    async def __aenter__(self):
        try:
            await asyncio.wait_for(self._lock.acquire(), timeout=self._timeout)
            return self
        except asyncio.TimeoutError:
            raise asyncio.TimeoutError(f"Could not acquire lock within {self._timeout} seconds") from None

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        self._lock.release()


class RateLimiter:
    """Simple async rate limiter"""

    def __init__(self, max_calls: int, time_window: float):
        self.max_calls = max_calls
        self.time_window = time_window
        self.calls = []
        self._lock = asyncio.Lock()

    async def wait_if_needed(self):
        """Wait if rate limit would be exceeded"""
        async with self._lock:
            now = asyncio.get_event_loop().time()

            # Remove old calls outside the time window
            self.calls = [call_time for call_time in self.calls if now - call_time < self.time_window]

            if len(self.calls) >= self.max_calls:
                # Calculate how long to wait
                oldest_call = min(self.calls)
                wait_time = self.time_window - (now - oldest_call)

                if wait_time > 0:
                    await asyncio.sleep(wait_time)

            # Record this call
            self.calls.append(now)


async def retry_with_backoff(
    func: Callable[..., Awaitable[T]],
    max_retries: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 60.0,
    backoff_factor: float = 2.0,
    exceptions: tuple = (Exception,),
    *args,
    **kwargs,
) -> T:
    """
    Retry an async function with exponential backoff

    Args:
        func: Async function to retry
        max_retries: Maximum number of retries
        base_delay: Initial delay between retries
        max_delay: Maximum delay between retries
        backoff_factor: Factor to multiply delay by after each retry
        exceptions: Tuple of exceptions to catch and retry on
        *args: Function arguments
        **kwargs: Function keyword arguments

    Returns:
        Function result

    Raises:
        Last exception if all retries exhausted
    """
    last_exception = None
    delay = base_delay

    for attempt in range(max_retries + 1):
        try:
            return await func(*args, **kwargs)
        except exceptions as e:
            last_exception = e

            if attempt == max_retries:
                logger.error(f"All {max_retries} retries exhausted for {func.__name__}")
                break

            logger.warning(f"Attempt {attempt + 1} failed for {func.__name__}: {e}. Retrying in {delay}s...")
            await asyncio.sleep(delay)
            delay = min(delay * backoff_factor, max_delay)

    raise last_exception


class AsyncBatch:
    """Process items in batches asynchronously"""

    def __init__(self, batch_size: int = 100, max_concurrency: int = 5):
        self.batch_size = batch_size
        self.max_concurrency = max_concurrency

    async def process(self, items: list[Any], processor: Callable[[list[Any]], Awaitable[list[Any]]]) -> list[Any]:
        """
        Process items in batches

        Args:
            items: List of items to process
            processor: Async function that processes a batch of items

        Returns:
            List of all processed results
        """
        # Split items into batches
        batches = [items[i : i + self.batch_size] for i in range(0, len(items), self.batch_size)]

        # Process batches with limited concurrency
        batch_coroutines = [processor(batch) for batch in batches]
        batch_results = await gather_with_concurrency(batch_coroutines, self.max_concurrency)

        # Flatten results
        results = []
        for batch_result in batch_results:
            results.extend(batch_result)

        return results


def async_lru_cache(maxsize: int = 128, ttl: float = 300.0):
    """
    LRU cache decorator for async functions with TTL support

    Args:
        maxsize: Maximum cache size
        ttl: Time-to-live in seconds
    """

    def decorator(func):
        cache = {}
        access_times = {}

        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            # Create cache key
            key = str(args) + str(sorted(kwargs.items()))
            now = asyncio.get_event_loop().time()

            # Check if cached result exists and is still valid
            if key in cache and (now - access_times[key]) < ttl:
                return cache[key]

            # Call function and cache result
            result = await func(*args, **kwargs)

            # Implement LRU eviction if cache is full
            if len(cache) >= maxsize and key not in cache:
                # Remove oldest entry
                oldest_key = min(access_times.keys(), key=lambda k: access_times[k])
                del cache[oldest_key]
                del access_times[oldest_key]

            cache[key] = result
            access_times[key] = now

            return result

        return wrapper

    return decorator


async def shutdown_thread_pool():
    """Shutdown the global thread pool"""
    global _thread_pool

    if _thread_pool is not None:
        _thread_pool.shutdown(wait=True)
        _thread_pool = None
        logger.info("Thread pool shut down")
