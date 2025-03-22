import json 
from extractdata.utils import convert_response_to_df, get_command_from
from rich import print
import pandas as pd

class FakeResponse:
    def __init__(self, json_data):
        self.text = json_data

with open("cook.json", "r") as f:
    cooked = json.load(f)

# print(cooked)

f = json.loads(cooked["result"]["message"]["content"][0]["text"])[0]["data"]
# print(f)
df = pd.DataFrame(f)

print(df)

a, insc, dft = get_command_from(df, "e")

print(a)