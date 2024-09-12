import json
import re


def extract_and_parse_json(response: str):
    """
    Tries to parse the response directly as JSON. If it fails,
    attempts to extract the JSON part from the response and parse it.
    """
    try:
        # Try direct parsing first
        return json.loads(response)
    except json.JSONDecodeError:
        # If direct parsing fails, extract JSON part using regex
        json_match = re.search(r'\{.*?\}', response, re.DOTALL)
        if json_match:
            try:
                # Attempt to parse the extracted JSON part
                return json.loads(json_match.group(0))
            except json.JSONDecodeError:
                raise ValueError("Extracted content is not valid JSON.")
        else:
            raise ValueError("No JSON found in the response.")