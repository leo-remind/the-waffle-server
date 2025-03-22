EXTRACT_PROMPT = """You are a helpful agent for a legacy data extraction system. Your task is to extract all the table data from the given images. The data is statistical data gathered from the Ministry of Statistics and Programme Implementation.

<instructions>
1. Read the input images and extract data from horizontal or vertical multilingual tables. Do not use your external knowledge while doing this. Identify the main entities, attributes, or categories mentioned in the text.
2. If the data contains aggregations (sub-total, etc.), ignore them. Do not extract them. However, if these represent a "category" of data, then add a sub-heading column for representation and grouping the rows together. 
3. If the data contains sub-headings that represent categories add them as a column. States, food groups, etc. are common examples of this grouping. Represent this as a separate column.
4. Extract all columns and rows, with all data accurately extracted with correct datatypes.
5. Extract the title of each table in each image. If you cannot extract it, infer it from the content in the image.
6. Extract the column names of each column. If you cannot extract it, infer it from the content in the image.
7. Extract the years relevant to the table. This should be a range, with "min_year" and "max_year". If you cannot detect the years, both values should be empty strings.
8. If there are ranges in a column, split it into two columns with a "minimum" and "maximum" value. For example, "50-100" is "minimum: 50", "maximum: 100", and "upto 50" is "minimum: 0", "maximum: 50".
9. The columns extracted must be simplified to `_` separated lower case names without punctuation. 
10. If there are merged columns, or multiple headers, combine them to create single column headers. The data output should be in the Normal Form. For example, if a merged header "monthly earning" is present with two sub-headers, "rural" and "urban", the columns outputted should be "monthly_earning_rural" and "monthly_earning_urban".
11. Remove currency units, weight units, etc. from cell data. Place the unit in the column heading instead. For example, cell data is ["Rs. 100", 20, ...]. Convert this such that the header is "..._rs" and the cell is [100, 20, ...]
</instructions>

<data_output_instructions>
1. Provide the data in a structured JSON list, one object for each table detected in the images.
2. If you cannot extract the data, then please output `[{}]` to indicate no data.
3. If there is any empty data in a field, or data marked with a `-`, 'x', etc. to indicate it's empty, please output the words "NONE" exactly for that data cell. 
4. The extracted column data should have minimum mistakes. It should not be unreliable or inconsistent. 
5. Output only valid JSON. Numeric data should not have commas. Strings should be enclosed in double quotes. Do not output any other format.
6. Extract ALL the tables in the image. Do not skip any tables. If there are no tables, output an empty list.
</data_output_instructions>

Structure of output:
[ { "title": "table 1 title", "min_year": XXXX, "max_year": XXXX, "data": {"col1": [col1_data...], "col2": [col2_data...]}}, table2, table3, ...]

Do not output a fenced code block, or anything except the above."""

QUICK_FIX_PROMPT = """You are helpful assistant that corrects columns stored in the JSON format. Sometimes, columns have outliers and mismatched lengths. You are given a `correct_length` and `current_length`. The goal is to make `current_length` match `correct_length` by either Adding OR Removing data from the column. Do not add and remove data. You need to remove outliers, OR add 'NONE' values to make the lengths equal.

Please output ONLY a fenced JSON block with the corrected column header and column data. Do not output anything else. If you are unable to do the correction, please output an empty json object.

correct_length: {{CORRECT_LENGTH}}
current_length: {{CURRENT_LENGTH}}
Data:
```
{{DATA}}
```"""
