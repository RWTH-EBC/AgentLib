from typing import Iterable, Union, List

try:
    from rapidfuzz import process as fuz_process
    from rapidfuzz import fuzz

    RAPIDFUZZ_IS_INSTALLED = True
except ImportError:
    RAPIDFUZZ_IS_INSTALLED = False


def fuzzy_match(target: str, choices: Iterable[str]) -> Union[None, List[str]]:
    if not RAPIDFUZZ_IS_INSTALLED:
        return None
    matches = fuz_process.extract(query=target, choices=choices, scorer=fuzz.WRatio)
    return [m[0] for m in matches]
