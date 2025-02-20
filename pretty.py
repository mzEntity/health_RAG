from typing import Union, List

font_color_map = {
    "black":    "30",
    "red":      "31",
    "green":    "32",
    "yellow":   "33",
    "blue":     "34",
    "purple":   "35",
    "cyan":     "36",
    "white":    "37"
}

background_color_map = {
    "black":    "40",
    "red":      "41",
    "green":    "42",
    "yellow":   "43",
    "blue":     "44",
    "purple":   "45",
    "cyan":     "46",
    "white":    "47"
}

effect_map = {
    "bold":       "1",
    "underline":  "4",
    "blink":      "5",
    "reverse":    "7",
    "hidden":     "8"
}

def colored_text(text: str, font_color: str=None, background_color: str=None, effect: Union[str, List[str]]=None) -> str:
    """print text with colors.

    Args:
        text (str): text to print
        font_color (str, optional): foreground color. Defaults to None.
        background_color (str, optional): background color. Defaults to None.
        effect (Union[str, List[str]], optional): other effects. Defaults to None.

    Returns:
        str: processed text
    """
    escape_code = "\033["
    
    codes = []
    if effect:
        if isinstance(effect, str):
            codes.append(effect_map[effect])
        elif isinstance(effect, list):
            codes.extend(effect_map[eff] for eff in effect)
    if font_color:
        codes.append(font_color_map[font_color])
    if background_color:
        codes.append(background_color_map[background_color])
    
    escape_code += ";".join(codes) + "m"
    reset_code = "\033[0m"
    
    return f"{escape_code}{text}{reset_code}"