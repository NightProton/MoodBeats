# mood_map.py
EMOTIONS = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']

def get_mood(emotion):
    mood_map = {
        "happy":    {"genre": "Pop, Indie, Dance", "energy": "High", "bpm": "110-130", "vibe": "Uplifting, Energetic"},
        "sad":      {"genre": "Indie, Acoustic, Lo-fi", "energy": "Low", "bpm": "60-80", "vibe": "Melancholic, Reflective"},
        "angry":    {"genre": "Rock, Metal, Hip-Hop", "energy": "High", "bpm": "140-160", "vibe": "Intense, Aggressive"},
        "surprise": {"genre": "Electronic, EDM, Pop", "energy": "High", "bpm": "120-140", "vibe": "Exciting, Dynamic"},
        "fear":     {"genre": "Ambient, Dark Electronic", "energy": "Medium", "bpm": "80-100", "vibe": "Tense, Suspenseful"},
        "disgust":  {"genre": "Alternative, Punk", "energy": "Medium", "bpm": "90-110", "vibe": "Raw, Rebellious"},
        "neutral":  {"genre": "Chill, Lo-fi, Jazz", "energy": "Medium", "bpm": "90-110", "vibe": "Calm, Balanced"}
    }
    return mood_map.get(emotion, mood_map["neutral"])