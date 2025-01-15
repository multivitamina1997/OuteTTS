def clean_dashes(text: str) -> str:
    dashes = ['—', '–', '-']
    for dash in dashes:
        text = text.replace(dash, ' ')
    return text
