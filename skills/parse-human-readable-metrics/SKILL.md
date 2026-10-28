---
name: parse-human-readable-metrics
description: Use when parsing numeric values from text that include magnitude suffixes like 'k', 'M', or 'B' to represent scale.
---
# Parse Human-Readable Metric Suffixes

## When to use
Use when input strings contain numbers followed by unit suffixes that represent multipliers (e.g., 'k' for thousand, 'M' for million, 'B' for billion, 'T' for trillion).

## Procedure
1. Clean the input string by removing whitespace and converting it to lowercase.
2. Check if the last character of the string is a known alphabetic suffix.
3. Define a mapping of suffixes to their corresponding multipliers (e.g., `{'k': 1e3, 'm': 1e6, 'b': 1e9, 't': 1e12}`).
4. If a suffix is found, extract the numeric portion, convert it to a float, and multiply it by the mapped value.
5. If no suffix is found, convert the entire string to a float.
