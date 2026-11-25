"""Publication labels; storage keys and experiment identities stay unchanged."""

CONTEXT_SYMBOLS = {
    'L': r'\sigma^{\mathrm{lead}}',
    'M': r'\sigma^{\mathrm{life}}',
    'K': r'\sigma^{\mathrm{back}}',
}


def context_math(prefix, value=None):
    symbol = CONTEXT_SYMBOLS[prefix]
    return '$' + symbol + (f' = {value}' if value is not None else '') + '$'
