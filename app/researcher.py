import requests
from bs4 import BeautifulSoup
from ddgs import DDGS
from ddgs.exceptions import DDGSException

from .models import Source


def _run_search(query: str, max_results: int) -> list[Source]:

    with DDGS() as ddgs:

        search_results = ddgs.text(
            query,
            max_results=max_results
        )

        return [
            Source(
                title=result.get("title", ""),
                url=result.get("href", "")
            )
            for result in search_results
        ]


def search_web(
    query: str,
    max_results: int = 5
) -> list[Source]:

    # First attempt
    try:

        results = _run_search(
            query,
            max_results
        )

        if results:
            return results

    except DDGSException as e:

        print(f"  ⚠ Search failed: {e}")

    # Retry with simplified query
    try:

        simplified_query = " ".join(
            query.split()[:10]
        )

        print(
            f"  ↻ Retrying: {simplified_query}"
        )

        return _run_search(
            simplified_query,
            max_results
        )

    except Exception as e:

        print(
            f"  ⚠ Search failed after retry: {e}"
        )

        return []


def fetch_page(url: str) -> str:

    headers = {
        "User-Agent": (
            "Mozilla/5.0 "
            "(Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 "
            "Chrome/131.0 Safari/537.36"
        )
    }

    try:

        response = requests.get(
            url,
            headers=headers,
            timeout=15
        )

        response.raise_for_status()

        soup = BeautifulSoup(
            response.text,
            "html.parser"
        )

        # Remove elements that usually contain
        # navigation, scripts, styling, etc.
        for element in soup(
            [
                "script",
                "style",
                "nav",
                "footer",
                "header",
                "aside"
            ]
        ):
            element.decompose()

        text = soup.get_text(
            separator=" ",
            strip=True
        )

        # Limit the amount of text we keep.
        return text[:20000]

    except requests.RequestException as e:

        print(f"  ⚠ Could not fetch {url}")
        print(f"    {e}")

        return ""