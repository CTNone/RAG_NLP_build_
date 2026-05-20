# Golden Answers Format

Place golden CSV files in `data/test_goldens/` with the same filename as in `data/test_questions/`.

Required columns:
- `Question`

Optional columns:
- `Expected` (substring match, case-insensitive)
- `ExpectedRegex` (regex match, case-insensitive)

Examples:
```csv
Question,Expected
DHQGHN duoc thanh lap khi nao?,10 thang 12 nam 1993
```

```csv
Question,ExpectedRegex
DHQGHN duoc thanh lap khi nao?,\\b10\\s+thang\\s+12\\s+nam\\s+1993\\b
```

