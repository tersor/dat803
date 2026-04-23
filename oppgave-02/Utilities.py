"""Helper utilities for feature engineering in Assignment 2."""
import holidays

_HOLIDAYS = holidays.US()

def is_weekend(day_of_week):
    """Return 1 when day_of_week is Saturday (5) or Sunday (6), else 0."""
    return 1 if day_of_week >= 5 else 0


def is_holiday(date_value, holiday_calendar=None):
    """Return 1 when date_value is in a Norwegian holiday calendar, else 0."""
    calendar = holiday_calendar if holiday_calendar is not None else _HOLIDAYS
    return 1 if date_value in calendar else 0
