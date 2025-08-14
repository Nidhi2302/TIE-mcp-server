# API Reference

The TIE MCP Server provides the following MCP tools and resources for technique prediction and analysis.

## Tools

### predict_techniques

Predict likely MITRE ATT&CK techniques based on observed techniques.

**Parameters:**
- `techniques` (array): List of observed MITRE ATT&CK technique IDs
- `top_k` (integer, optional): Number of predictions to return (default: 10)
- `include_threat_intel` (boolean, optional): Include real-time threat intelligence (default: true)

**Example:**
```json
{
  "techniques": ["T1566.001", "T1204.002"],
  "top_k": 5,
  "include_threat_intel": true
}
```

**Response:**
```json
{
  "predicted_techniques": [
    {
      "technique_id": "T1055",
      "technique_name": "Process Injection",
      "score": 0.85,
      "confidence": "high"
    }
  ],
  "input_techniques": ["T1566.001", "T1204.002"],
  "execution_time_seconds": 0.1
}
```

### search_techniques

Search for MITRE ATT&CK techniques by name or description.

**Parameters:**
- `query` (string): Search query
- `limit` (integer, optional): Maximum results (default: 10)

**Example:**
```json
{
  "query": "phishing",
  "limit": 5
}
```

### get_technique_info

Get detailed information about a specific MITRE ATT&CK technique.

**Parameters:**
- `technique_id` (string): MITRE ATT&CK technique ID

**Example:**
```json
{
  "technique_id": "T1566.001"
}
```

### get_system_metrics

Get real-time system metrics and threat intelligence status.

**Parameters:** None

**Response:**
```json
{
  "cpu_usage_percent": 45.2,
  "memory_usage_percent": 62.1,
  "active_models": 3,
  "prediction_count": 1247,
  "threat_feeds_status": "active"
}
```

## Resources

### tie://models

Access to enhanced TIE models with threat intelligence integration.

### tie://techniques

Access to enhanced MITRE ATT&CK techniques database.

## Error Handling

All API endpoints return standard HTTP status codes and error messages:

- `200 OK`: Success
- `400 Bad Request`: Invalid parameters
- `404 Not Found`: Resource not found
- `500 Internal Server Error`: Server error

Error response format:
```json
{
  "error": "Invalid technique ID",
  "code": "INVALID_TECHNIQUE_ID",
  "details": {}
}