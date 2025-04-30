# """
# Module containing user-defined functions that can be called by the AI agent.
# """
# import json
# import datetime
# from typing import Dict, Optional
# from zoneinfo import ZoneInfo


# async def fetch_current_datetime(format: Optional[str] = None) -> str:
#     """
#     Get the current time as a JSON string, optionally formatted.

#     :param format (Optional[str]): The format in which to return the current time. Defaults to None, which uses a standard format.
#     :return: The current time in JSON format.
#     :rtype: str
#     """
#     current_time = datetime.datetime.now()

#     # Use the provided format if available, else use a default format
#     if format:
#         time_format = format
#     else:
#         time_format = "%Y-%m-%d %H:%M:%S"

#     time_json = json.dumps({"current_time": current_time.strftime(time_format)})
#     return time_json


import datetime
import json
from typing import Optional
from zoneinfo import ZoneInfo
from geopy.geocoders import Nominatim
from timezonefinder import TimezoneFinder

# Initialize geocoder and timezone resolver
geolocator = Nominatim(user_agent="azure_agent_time_tool")
tz_finder = TimezoneFinder()


async def fetch_current_datetime(city: Optional[str] = "UTC", format: Optional[str] = None) -> str:
    """
    Get the current time in the user's requested city using geolocation and timezone resolution.

    Args:
        city: Any city name (e.g., "Amsterdam", "Tokyo").
        format: Optional datetime format string.

    Returns:
        A JSON string with the current time in that city.
    """
    if not city:
        city = "UTC"

    try:
        location = geolocator.geocode(city, timeout=10)
        if not location:
            return json.dumps({"error": f"Could not find city '{city}'."})

        timezone_str = tz_finder.timezone_at(lat=location.latitude, lng=location.longitude)
        if not timezone_str:
            return json.dumps({"error": f"Could not determine timezone for '{city}'."})

        now = datetime.datetime.now(ZoneInfo(timezone_str))
        time_format = format or "%Y-%m-%d %H:%M:%S"

        return json.dumps({
            "city": city.title(),
            "timezone": timezone_str,
            "current_time": now.strftime(time_format)
        })

    except Exception as e:
        return json.dumps({"error": f"Exception occurred: {str(e)}"})