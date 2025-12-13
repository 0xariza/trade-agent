# Gemini API Fallback Keys

## Overview

The trading bot now supports automatic rotation between multiple Gemini API keys. If the primary key hits a quota limit or rate limit, the bot will automatically switch to the next available fallback key.

## Configuration

Add your fallback keys to your `.env` file:

```env
# Primary Gemini API Key
GEMINI_API_KEY=your_primary_key_here

# Fallback Gemini API Keys (comma-separated)
GEMINI_FALLBACK_KEYS=AIzaSyDXhEYXxCd_Ma6iRSp-SjFeXClpgouZ7qw,AIzaSyCUAsBJT2YfrR7X12UJuCLhK7Kes9Jv6ys,AIzaSyBn8_77NlAvw556ZpSzBpyVqm04M1rpInc
```

## How It Works

1. **Primary Key**: The bot starts with your primary `GEMINI_API_KEY`
2. **Quota Detection**: If the API returns a quota/rate limit error, the bot detects it
3. **Automatic Rotation**: The bot automatically switches to the next available fallback key
4. **Retry**: The request is retried with the new key
5. **Logging**: All key rotations are logged for monitoring

## Error Detection

The bot automatically detects these error types:
- Quota exceeded
- Rate limit (429)
- Resource exhausted (503)
- Usage limit reached
- Billing issues
- Invalid API key

## Example Log Output

```
[INFO] Gemini Agent initialized with model: models/gemini-2.0-flash
[INFO] Total API keys available: 4 (primary + 3 fallback)
[WARNING] API key #1 hit quota limit. Rotating...
[WARNING] Rotated to fallback API key #2/4
```

## Benefits

- **No Manual Intervention**: Automatic key rotation when limits are hit
- **Continuous Trading**: Bot continues operating even if one key is exhausted
- **Better Reliability**: Reduces downtime from API quota issues
- **Cost Distribution**: Spreads API usage across multiple keys

## Notes

- Keys are rotated in order: Primary → Fallback 1 → Fallback 2 → ...
- Failed keys are tracked and skipped in future rotations
- If all keys are exhausted, the bot will fall back to the rule-based agent (if enabled)
- Key rotation happens automatically - no configuration needed beyond adding the keys

## Testing

To verify your fallback keys are configured:

```bash
python3 -c "from backend.config import settings; print(f'Primary: {bool(settings.gemini_api_key)}'); print(f'Fallbacks: {len([k for k in settings.gemini_fallback_keys.split(\",\") if k.strip()]) if settings.gemini_fallback_keys else 0}')"
```

