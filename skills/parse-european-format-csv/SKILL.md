---
name: parse-european-format-csv
description: Use when parsing CSV data that uses semicolons as delimiters and European-style numeric formatting (dots for thousands and commas for decimals).
---
# Parse European Format CSV

## When to use
Use when the input CSV text uses semicolons (`;`) as the field separator and numbers are formatted with a comma as the decimal separator and dots as thousands separators (e.g., `1.234,56`).

## Procedure
1. Clean the raw text by stripping whitespace and removing the Byte Order Mark (BOM) using `.lstrip('\ufeff')`.
2. Initialize `csv.DictReader` specifying the correct `delimiter=';'`.
3. For numeric columns, remove the thousands separator (the dot) and replace the decimal separator (the comma) with a dot before converting to a float.
