"""
Date utility functions
"""
from datetime import datetime
from fastapi import HTTPException


def parse_date(date_str: str) -> datetime:
    """Parse date string into datetime object"""
    try:
        parsed_date = datetime.strptime(date_str, '%Y-%m-%d')
        if parsed_date.year < 2020 or parsed_date.year > 2030:
            raise ValueError("Date must be between 2020 and 2030")
        return parsed_date
    except ValueError as e:
        if "does not match format" in str(e):
            raise HTTPException(status_code=400, detail="Invalid date format. Use YYYY-MM-DD")
        else:
            raise HTTPException(status_code=400, detail=f"Invalid date: {str(e)}")


def calculate_days_ahead(target_date: datetime) -> int:
    """Calculate days between now and target date"""
    current_date = datetime.now()
    return (target_date - current_date).days
