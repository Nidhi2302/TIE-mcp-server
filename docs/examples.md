# Examples

This page provides practical examples of using the TIE MCP Server.

## Basic Technique Prediction

### Example 1: Simple Prediction

Predict techniques based on observed phishing activity:

```python
import asyncio
from mcp import ClientSession
from mcp.client.stdio import stdio_client

async def predict_techniques():
    async with stdio_client(["tie-mcp-server"]) as (read, write):
        async with ClientSession(read, write) as session:
            result = await session.call_tool(
                "predict_techniques",
                {
                    "techniques": ["T1566.001", "T1204.002"],
                    "top_k": 5
                }
            )
            print(result)

asyncio.run(predict_techniques())
```

### Example 2: Advanced Prediction with Threat Intel

```python
async def advanced_prediction():
    async with stdio_client(["tie-mcp-server"]) as (read, write):
        async with ClientSession(read, write) as session:
            result = await session.call_tool(
                "predict_techniques",
                {
                    "techniques": ["T1059.001", "T1055", "T1082"],
                    "top_k": 10,
                    "include_threat_intel": True,
                    "include_reasoning": True
                }
            )
            
            for prediction in result["predicted_techniques"]:
                print(f"Technique: {prediction['technique_name']}")
                print(f"Score: {prediction['score']:.2f}")
                print(f"Reasoning: {prediction.get('reasoning', 'N/A')}")
                print("---")

asyncio.run(advanced_prediction())
```

## Technique Search

### Example 3: Search for Techniques

```python
async def search_techniques():
    async with stdio_client(["tie-mcp-server"]) as (read, write):
        async with ClientSession(read, write) as session:
            result = await session.call_tool(
                "search_techniques",
                {
                    "query": "lateral movement",
                    "limit": 5
                }
            )
            
            for technique in result["techniques"]:
                print(f"{technique['id']}: {technique['name']}")
                print(f"Description: {technique['description'][:100]}...")
                print("---")

asyncio.run(search_techniques())
```

## System Monitoring

### Example 4: Get System Metrics

```python
async def get_metrics():
    async with stdio_client(["tie-mcp-server"]) as (read, write):
        async with ClientSession(read, write) as session:
            result = await session.call_tool("get_system_metrics", {})
            
            print(f"CPU Usage: {result['cpu_usage_percent']:.1f}%")
            print(f"Memory Usage: {result['memory_usage_percent']:.1f}%")
            print(f"Active Models: {result['active_models']}")
            print(f"Total Predictions: {result['prediction_count']}")

asyncio.run(get_metrics())
```

## Working with Resources

### Example 5: Access Model Information

```python
async def get_models():
    async with stdio_client(["tie-mcp-server"]) as (read, write):
        async with ClientSession(read, write) as session:
            models = await session.read_resource("tie://models")
            
            for model in models["models"]:
                print(f"Model: {model['name']}")
                print(f"Type: {model['type']}")
                print(f"Accuracy: {model['metrics']['accuracy']:.2f}")
                print("---")

asyncio.run(get_models())
```

## Batch Processing

### Example 6: Process Multiple Threat Reports

```python
async def process_threat_reports():
    reports = [
        {"id": "report1", "techniques": ["T1566.001", "T1204.002"]},
        {"id": "report2", "techniques": ["T1059.003", "T1053.005"]},
        {"id": "report3", "techniques": ["T1055", "T1082", "T1105"]}
    ]
    
    async with stdio_client(["tie-mcp-server"]) as (read, write):
        async with ClientSession(read, write) as session:
            results = []
            
            for report in reports:
                prediction = await session.call_tool(
                    "predict_techniques",
                    {
                        "techniques": report["techniques"],
                        "top_k": 3
                    }
                )
                
                results.append({
                    "report_id": report["id"],
                    "input_techniques": report["techniques"],
                    "predictions": prediction["predicted_techniques"]
                })
            
            # Analyze results
            for result in results:
                print(f"Report {result['report_id']}:")
                print(f"  Input: {result['input_techniques']}")
                print("  Top Predictions:")
                for pred in result['predictions'][:3]:
                    print(f"    {pred['technique_id']}: {pred['score']:.2f}")
                print()

asyncio.run(process_threat_reports())
```

## Error Handling

### Example 7: Robust Error Handling

```python
async def robust_prediction():
    async with stdio_client(["tie-mcp-server"]) as (read, write):
        async with ClientSession(read, write) as session:
            try:
                result = await session.call_tool(
                    "predict_techniques",
                    {
                        "techniques": ["T1566.001", "INVALID_ID"],
                        "top_k": 5
                    }
                )
                print("Success:", result)
                
            except Exception as e:
                print(f"Error occurred: {e}")
                
                # Try with valid techniques only
                result = await session.call_tool(
                    "predict_techniques",
                    {
                        "techniques": ["T1566.001"],
                        "top_k": 5
                    }
                )
                print("Fallback result:", result)

asyncio.run(robust_prediction())
```

## Integration with Other Tools

### Example 8: Claude Desktop Integration

Configure the TIE MCP Server in Claude Desktop by adding to your `claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "tie-mcp-server": {
      "command": "tie-mcp-server",
      "args": [],
      "env": {
        "LOG_LEVEL": "INFO"
      }
    }
  }
}
```

Then use it in conversations:

```
@tie-mcp-server Can you predict likely next techniques after observing T1566.001 and T1204.002?