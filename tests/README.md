# Alpha Arena Test Suite

## Overview

This directory contains unit and integration tests for the Alpha Arena trading bot.

## Structure

```
tests/
├── conftest.py              # Pytest fixtures and configuration
├── unit/                    # Unit tests (fast, isolated)
│   ├── test_risk_manager.py
│   └── ...
└── integration/            # Integration tests (slower, require setup)
    ├── test_trading_cycle.py
    └── ...
```

## Running Tests

### Run all tests
```bash
pytest
```

### Run only unit tests
```bash
pytest tests/unit/
```

### Run only integration tests
```bash
pytest tests/integration/
```

### Run with coverage
```bash
pytest --cov=backend --cov-report=html
```

### Run specific test file
```bash
pytest tests/unit/test_risk_manager.py
```

### Run with verbose output
```bash
pytest -v
```

## Test Markers

Tests are marked with categories:

- `@pytest.mark.unit` - Fast unit tests
- `@pytest.mark.integration` - Integration tests
- `@pytest.mark.slow` - Slow running tests
- `@pytest.mark.requires_db` - Tests requiring database
- `@pytest.mark.requires_api` - Tests requiring external APIs

Run specific markers:
```bash
pytest -m unit
pytest -m "not slow"
```

## Fixtures

Common fixtures available in `conftest.py`:

- `mock_market_data` - Sample market data for testing
- `mock_llm_response` - Mock LLM agent response
- `mock_paper_exchange` - Mock paper exchange
- `mock_market_data_provider` - Mock market data provider
- `mock_risk_manager` - Mock risk manager
- `sample_settings` - Sample settings configuration
- `mock_agent` - Mock trading agent

## Writing Tests

### Example Unit Test

```python
import pytest
from backend.risk.risk_manager import RiskManager

def test_calculate_position_size(mock_risk_manager):
    """Test position size calculation."""
    size = mock_risk_manager.calculate_position_size(
        portfolio_value=10000.0,
        entry_price=50000.0,
        stop_loss=49000.0
    )
    assert size > 0
    assert size * 50000.0 <= 1000.0  # Max 10% position
```

### Example Integration Test

```python
import pytest

@pytest.mark.asyncio
async def test_trading_cycle(mock_agent, mock_market_data):
    """Test complete trading cycle."""
    result = await mock_agent.analyze_market(mock_market_data)
    assert result is not None
```

## Continuous Integration

Tests should pass before merging PRs. Target: 80%+ code coverage.

## Notes

- All async tests must use `@pytest.mark.asyncio`
- Mock external dependencies (LLM, exchange, database)
- Keep tests fast and isolated
- Use descriptive test names

