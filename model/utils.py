"""Utility functions for the trend analysis system."""

import pandas as pd
import numpy as np
import re
from typing import List, Dict, Any
import matplotlib.pyplot as plt
import seaborn as sns


def clean_text(text: str) -> str:
    """Clean text while preserving emojis and meaningful slang."""
    if pd.isna(text):
        return ""
    
    text = str(text)
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    text = re.sub(r'[.]{3,}', '...', text)
    text = re.sub(r'[!]{2,}', '!!', text)
    text = re.sub(r'[?]{2,}', '??', text)
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'[^\w\s\U0001F600-\U0001F64F\U0001F300-\U0001F5FF\U0001F680-\U0001F6FF\U0001F1E0-\U0001F1FF,.!?;:\'"()-]', '', text)
    
    return text.strip()


def process_tags(tags) -> List[str]:
    """Process and clean video tags."""
    if pd.isna(tags) or tags == '' or tags is None:
        return []
    
    if isinstance(tags, str):
        if ',' in tags:
            tag_list = tags.split(',')
        elif ';' in tags:
            tag_list = tags.split(';')
        elif '|' in tags:
            tag_list = tags.split('|')
        else:
            tag_list = [tags] if ' ' not in tags else tags.split()
    elif isinstance(tags, list):
        tag_list = tags
    else:
        tag_list = str(tags).split(',')
    
    processed_tags = []
    for tag in tag_list:
        if isinstance(tag, str):
            clean_tag = tag.strip().strip('"\'').lower()
            clean_tag = re.sub(r'[^\w\s\-]', '', clean_tag)
            if clean_tag and len(clean_tag) > 1:
                processed_tags.append(clean_tag)
    
    return processed_tags


def setup_plotting_style():
    """Set up consistent plotting style."""
    plt.style.use('seaborn-v0_8')
    sns.set_palette("husl")
