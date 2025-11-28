import os
import asyncio
from unittest.mock import MagicMock, patch, AsyncMock
from backend.agents import DeepSeekAgent, QwenAgent

async def test_agents():
    print("Setting up mock environment variables...")
    os.environ["DEEPSEEK_API_KEY"] = "mock_deepseek_key"
    os.environ["QWEN_API_KEY"] = "mock_qwen_key"

    print("Initializing agents...")
    deepseek = DeepSeekAgent()
    qwen = QwenAgent()
    
    print(f"DeepSeek Agent initialized: {deepseek.name}")
    print(f"Qwen Agent initialized: {qwen.name}")

    # Mock the OpenAI client calls
    mock_response = MagicMock()
    mock_response.choices = [
        MagicMock(message=MagicMock(content='{"trend": "bullish", "confidence": 0.8, "reasoning": "Upward momentum"}'))
    ]
    
    print("Testing DeepSeek analysis (mocked)...")
    # Use AsyncMock for the create method
    with patch.object(deepseek.client.chat.completions, 'create', new_callable=AsyncMock) as mock_create:
        mock_create.return_value = mock_response
        analysis = await deepseek.analyze_market({"price": 100})
        print(f"DeepSeek Analysis: {analysis}")
        assert "bullish" in analysis

    print("Testing Qwen analysis (mocked)...")
    with patch.object(qwen.client.chat.completions, 'create', new_callable=AsyncMock) as mock_create:
        mock_create.return_value = mock_response
        analysis = await qwen.analyze_market({"price": 100})
        print(f"Qwen Analysis: {analysis}")
        assert "bullish" in analysis

    print("Verification successful!")

if __name__ == "__main__":
    asyncio.run(test_agents())
