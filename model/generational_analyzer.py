"""Enhanced generational language analyzer for beauty industry content."""

import re
import numpy as np
from typing import Dict, List
from collections import Counter

class EnhancedGenerationalLanguageAnalyzer:
    """
    Enhanced analyzer for detecting generational language patterns with beauty-specific terms.
    """
    
    def __init__(self):
        # Enhanced generational vocabulary patterns based on current social media usage
        self.generational_patterns = {
            'gen_z': {
                'slang': [
                    # Core Gen Z terms
                    'slay', 'periodt', 'no cap', 'fr', 'frfr', 'bussin', 'sheesh', 'sus',
                    'bet', 'lowkey', 'highkey', 'deadass', 'based', 'cringe', 'hits different',
                    'slaps', 'vibe check', 'stan', 'bestie', 'bestyyy', 'girlie', 'girly',
                    'queen', 'king', 'icon', 'iconic', 'legend', 'fire', 'lit', 'mid',
                    'cap', 'facts', 'say less', 'periodt pooh', 'chile', 'oop', 'and i oop',
                    'sksksk', 'vsco', 'simp', 'salty', 'tea', 'spill', 'mood', 'same',
                    
                    # Beauty-specific Gen Z terms
                    'snatched', 'beat', 'glow up', 'lewk', 'serve', 'serving looks',
                    'beat face', 'contour', 'highlight', 'brows on fleek', 'cut crease',
                    'wing', 'winged liner', 'blend', 'shade', 'transition', 'inner corner',
                    'lashes', 'falsies', 'mascara wand', 'setting spray', 'prime',
                    'dewy', 'matte', 'glossy', 'shimmer', 'pigmented', 'buildable',
                    'grwm', 'get ready with me', 'skincare routine', 'glass skin',
                    'no makeup makeup', 'fresh face', 'natural glam', 'everyday look',
                    'night out', 'date night', 'going out', 'soft glam', 'dramatic',
                    'smoky eye', 'nude lip', 'bold lip', 'red lip', 'glossy lip',
                ],
                'expressions': [
                    'literally me', 'this is everything', 'not me', 'the way i',
                    'tell me why', 'bestie this', 'girlie that', 'the audacity',
                    'i cannot', 'i cant even', 'this aint it', 'we been knew',
                    'it be like that', 'chile anyways', 'as you should',
                    'living for this', 'obsessed', 'im deceased', 'i am deceased',
                    'this look', 'that beat', 'those brows', 'the glow',
                    'your skin', 'that highlight', 'the blend', 'those lashes',
                    'tutorial please', 'drop the routine', 'what products',
                    'need this', 'want this', 'trying this'
                ],
                'beauty_expressions': [
                    'beat for the gods', 'face beat', 'mug is beat', 'serving face',
                    'your mug', 'that mug', 'face card', 'never declines',
                    'face is giving', 'look is giving', 'serving looks',
                    'main character energy', 'hot girl', 'that girl',
                    'clean girl', 'it girl', 'effortless', 'no effort'
                ]
            },
            'millennial': {
                'slang': [
                    # Core Millennial terms
                    'basic', 'bye felicia', 'cray', 'fleek', 'on fleek', 'ghosting',
                    'hashtag', 'jelly', 'savage', 'shade', 'throwing shade', 'squad',
                    'goals', 'relationship goals', 'thirsty', 'turnt', 'yasss', 'zero chill',
                    'dead', 'dying', 'literally cant', 'on point', 'mood', 'relatable',
                    'awkward', 'random', 'epic', 'fail', 'winning', 'adulting',
                    'bae', 'fam', 'woke', 'snatched', 'extra', 'pressed',
                    
                    # Beauty-specific Millennial terms
                    'contour', 'highlight', 'strobing', 'baking', 'cut crease',
                    'winged eyeliner', 'bold brow', 'power brow', 'ombre',
                    'balayage', 'lob', 'beach waves', 'no poo', 'bb cream',
                    'cc cream', 'primer', 'setting powder', 'bronzer', 'blush',
                    'lipstick', 'lip gloss', 'matte lips', 'liquid lipstick',
                    'eyeshadow palette', 'makeup haul', 'beauty guru', 'tutorial'
                ],
                'expressions': [
                    'i literally', 'so random', 'hot mess', 'train wreck',
                    'comfort zone', 'bucket list', 'netflix and chill',
                    'sorry not sorry', 'my bad', 'lets do this', 'game changer',
                    'life hack', 'pro tip', 'diy', 'holy grail', 'ride or die',
                    'must have', 'obsessed with', 'in love with', 'cant live without',
                    'beauty routine', 'morning routine', 'night routine',
                    'self care', 'treat yourself', 'me time'
                ],
                'internet_culture': [
                    'lol', 'omg', 'wtf', 'smh', 'tbh', 'imo', 'imho', 'rofl',
                    'lmao', 'brb', 'ttyl', 'irl', 'fomo', 'yolo', 'tbt',
                    'inspo', 'motd', 'fotd', 'ootd', 'notd'
                ]
            },
            'gen_x': {
                'slang': [
                    'whatever', 'as if', 'totally', 'tubular', 'rad', 'gnarly',
                    'dude', 'sweet', 'tight', 'sick', 'phat', 'da bomb',
                    'all that', 'bananas', 'bling', 'bouncing', 'chill',
                    'diss', 'fresh', 'funky', 'off the hook', 'trippin'
                ],
                'expressions': [
                    'talk to the hand', 'dont go there', 'been there done that',
                    'my bad', 'whats the deal', 'get real', 'not', 'psych',
                    'cowabunga', 'excellent', 'bogus', 'grody'
                ]
            },
            'boomer': {
                'formal_language': [
                    'wonderful', 'lovely', 'beautiful', 'amazing', 'fantastic',
                    'terrific', 'marvelous', 'delightful', 'charming', 'pleasant',
                    'gorgeous', 'stunning', 'pretty', 'nice', 'good'
                ],
                'expressions': [
                    'back in my day', 'when i was young', 'kids these days',
                    'in my time', 'years ago', 'old school', 'classic',
                    'traditional', 'proper', 'decent', 'respectable',
                    'elegant', 'sophisticated', 'timeless'
                ],
                'communication_style': [
                    'thank you', 'please', 'excuse me', 'pardon me',
                    'bless you', 'god bless', 'have a nice day',
                    'very nice', 'well done', 'good job'
                ]
            }
        }
        
        # Beauty-specific context indicators
        self.beauty_context = [
            'makeup', 'lipstick', 'eyeshadow', 'mascara', 'foundation', 'concealer',
            'blush', 'bronzer', 'highlighter', 'contour', 'primer', 'setting',
            'skincare', 'moisturizer', 'cleanser', 'serum', 'toner', 'sunscreen',
            'routine', 'tutorial', 'look', 'glam', 'natural', 'dramatic',
            'palette', 'brush', 'sponge', 'application', 'blend', 'shade'
        ]
        
        # Compile regex patterns with lower thresholds
        self.compiled_patterns = self._compile_patterns()
        
        # Adjust scoring weights
        self.category_weights = {
            'slang': 3.0,  # Increased weight
            'expressions': 2.5,
            'beauty_expressions': 4.0,  # Highest weight for beauty-specific
            'internet_culture': 2.0,
            'formal_language': 1.5,
            'communication_style': 1.0
        }
    
    def _compile_patterns(self):
        """Compile all patterns into regex for efficient matching."""
        compiled = {}
        
        for generation, patterns in self.generational_patterns.items():
            compiled[generation] = {}
            for category, terms in patterns.items():
                # Create more flexible patterns
                pattern_list = []
                for term in terms:
                    # Handle multi-word terms
                    if ' ' in term:
                        # Allow some variation in spacing and punctuation
                        flexible_term = term.replace(' ', r'\s+')
                        pattern_list.append(flexible_term)
                    else:
                        # Single word with word boundaries
                        pattern_list.append(r'\b' + re.escape(term) + r'\b')
                
                if pattern_list:
                    compiled[generation][category] = re.compile(
                        '|'.join(pattern_list), re.IGNORECASE
                    )
        
        return compiled
    
    def analyze_generational_language(self, text: str) -> Dict[str, float]:
        """Enhanced analysis with context awareness and flexible scoring."""
        if not text or len(text.strip()) < 3:
            return {gen: 0.0 for gen in self.generational_patterns.keys()}
        
        text_lower = text.lower()
        scores = {gen: 0.0 for gen in self.generational_patterns.keys()}
        
        # Check if it's beauty-related context
        has_beauty_context = any(term in text_lower for term in self.beauty_context)
        beauty_multiplier = 1.5 if has_beauty_context else 1.0
        
        # Analyze patterns for each generation
        for generation, patterns in self.compiled_patterns.items():
            for category, pattern in patterns.items():
                matches = len(pattern.findall(text))
                if matches > 0:
                    weight = self.category_weights.get(category, 1.0)
                    # Apply beauty context multiplier
                    if 'beauty' in category or has_beauty_context:
                        weight *= beauty_multiplier
                    
                    scores[generation] += matches * weight
        
        # Normalize by text length (words, not characters)
        word_count = len(text.split())
        if word_count > 0:
            for gen in scores:
                scores[gen] = scores[gen] / max(word_count, 1)
        
        return scores
    
    def classify_generation(self, text: str, threshold: float = 0.005) -> str:  # Lower threshold
        """Classify with more sensitive threshold."""
        scores = self.analyze_generational_language(text)
        
        if not any(score > 0 for score in scores.values()):
            return 'neutral'
        
        max_generation = max(scores.items(), key=lambda x: x[1])
        
        if max_generation[1] >= threshold:
            return max_generation[0]
        else:
            return 'neutral'
