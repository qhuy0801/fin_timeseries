"""
This module provides functionality to fetch data from specified URLs using the
requests library. It facilitates making HTTP GET or POST requests with options
for authorization, custom headers, and a request body. The response can be
either returned as a pandas DataFrame (if in CSV format and requested) or
saved directly to a CSV file given a filename.

Functions:
- fetch_data: Fetches data from a URL, optionally converting to a DataFrame or
              saving as a CSV.

Example usage:
```python
url = "https://www.example.com/api/data"
params = {"query": "value"}
df = fetch_data(url, params, to_pandas=True)
if df is not None:
    print(df.head())
"""
from typing import Optional, Dict, Any, Union
from io import StringIO
import requests
import pandas as pd


def fetch_data(
    url: str,
    params: Dict[str, Any],
    method: str = "GET",
    authorization: Optional[str] = None,
    headers: Optional[Dict[str, str]] = None,
    body: Optional[Dict[str, Any]] = None,
    reverse_data: bool = True,
) -> Optional[Union[pd.DataFrame, dict]]:
    """
    Fetch data from a specified URL with given parameters and options for authorization, headers, and body.
    If to_pandas is False, checks for a csv_filename to save the data as a CSV file.

    Parameters:
    - url (str): URL to fetch data from.
    - params (Dict[str, Any]): Dictionary of parameters to pass in the request.
    - to_pandas (bool): If True, attempts to return data as a pandas DataFrame. If False, requires csv_filename.
    - method (str): HTTP method to use ('GET', 'POST', etc.).
    - authorization (Optional[str]): Optional authorization token for the request.
    - headers (Optional[Dict[str, str]]): Optional HTTP headers to send with the request.
    - body (Optional[Dict[str, Any]]): Optional request body, useful for POST requests.
    - csv_filename (Optional[str]): Filename for saving the output CSV. Required if to_pandas is False.

    Returns:
    - Optional[pd.DataFrame]: A pandas DataFrame if to_pandas is True and data is in a compatible format; otherwise, None.
    """
    # Initialize headers if not provided
    if headers is None:
        headers = {}
    if authorization is not None:
        headers["Authorization"] = authorization

    if method.upper() == "POST":
        response = requests.post(url, params=params, headers=headers, json=body)
    else:
        response = requests.get(url, params=params, headers=headers)

    # Check if the request was successful
    if response.status_code == 200:
        if response.headers.get("Content-Type") == "application/x-download":
            try:
                csv_data = StringIO(response.text)
                df = pd.read_csv(csv_data)
                return df.iloc[::-1].reset_index(drop=True) if reverse_data else df
            except Exception as e:
                print(f"Failed to convert to DataFrame: {e}")
                return None
        elif response.headers.get("Content-Type") == "application/json":
            return response.json()
    else:
        print(f"Failed to retrieve data: {response.status_code}")
        return None
