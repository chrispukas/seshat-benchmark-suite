import requests

from typing import Dict, List, Any, Optional, Tuple

from llm_benchmark.utils import cache as c


def fetch_json_from_url(url: str,
                        args: Optional[Dict[str, Any]] = None
                        ) -> Dict[str, Any]:
    """
        Fetches JSON data from a specified URL.
        
        Args:
            url [str]: The request URL.
            args [Optional[Dict[str, Any]]]: Optional arguments for the request, automatically passed to requests.get().
        Returns:
            Dict[str, Any]: A JSON object parsed from the response.
    """

    try:
        response: requests.Response = requests.get(url, **(args or {}))
    except requests.exceptions.RequestException as e:
        print(f"An error occurred while making the request to {url}: {e}")
        return {}
    
    status_code: int = response.status_code
    if status_code != 200:
        print(f"Request to {url} failed with status code {status_code}.")
        return {}

    return response.json()

def get_polity_list(polity_url: str,
                    args: Optional[Dict[str, Any]] = None
                    ) -> Tuple[List[Dict], Optional[str]]:
    
    """
    Gets a list of items from the polity URL.
    
    Args:
        polity_url [str]: The request URL.
    Returns: 
        Tuple[
          List[Dict],   : A list of items retrieved from the URL. 
          Optional[str] : The next page URL if available, otherwise None.
        ]
    """

    # Returns the following items:
    # - count: total number of items across all pages
    # - next: URL of the next page (or None if there is no next page)
    # - previous: URL of the previous page (or None if there is no previous page)
    # - results: list of items on the current page

    json: Dict[str, Any] = fetch_json_from_url(polity_url, args)
    next_page: Optional[str] = json.get("next", None)
    results: List[Dict[str, Any]] = json.get("results", []) 
    return results, next_page

def traverse_polity(polity_url: str,
                    ) -> List[Dict[str, Any]]:
    """
    Traverses through all pages of a polity URL and collects all items.
    Args:
        polity_url [str]: The request URL."""
                    
    curr_polity_url: Optional[str] = polity_url
    all_items: List[Dict[str, Any]] = []
    current_page: Optional[str] = polity_url

    while current_page is not None:
        print("Fetching page:", current_page, end="\r")
        items, next_page = get_polity_list(current_page)
        all_items.extend(items)
        current_page: str = next_page

    return all_items

def root_search_url(root_url: str, 
                    use_cache: Optional[bool] = True,
                    cache_url: Optional[str] = "seshat_root_url.pkl") -> List[str]:
    """
    Extracts the root search URL from a given polity URL.
    
    Args:
        root_url [str]: The request URL.
    Returns:
        List[str]: A list of URL components.
    """

    if use_cache and c.exists_file(cache_url):
        return c.load_file(cache_url)

    json: Dict[str, Any] = fetch_json_from_url(root_url)

    if use_cache:
        c.save_file(json, cache_url)

    return json

def collapse_entry(entry: Dict[str, Any],
                   collapse_key: str = "polity"
                   ) -> Dict[str, Any]:
    """
    Collapses a nested dictionary entry by merging the contents of a specified key into the parent dictionary. This is an in-place operation.
    
    :param entry: Description
    :type entry: Dict[str, Any]
    :param collapse_key: Description
    :type collapse_key: str
    :return: Description
    :rtype: Dict[str, Any]
    """
    
    polity: Dict[str, Any] = entry.get(collapse_key, {})
    entry.pop(collapse_key, None)
    
    if polity:
        entry.update(polity)


    reset_keys_str: List[str] = [
        "shapefile_name",
        "long_name",
        "url_link",
        "name",
        "alternative_name",
        "comment",
        "description",
    ]

    for key in reset_keys_str:
        if key in entry and entry[key] is None:
            entry[key] = ""

    reset_keys_int: List[str] = [
        "year_from",
        "year_to",
    ]

    for key in reset_keys_int:
        if key in entry and entry[key] is None:
            entry[key] = -9999

    return entry
    
def collapse_all_entries(entries: List[Dict[str, Any]],
                         collapse_key: str = "polity"
                         ) -> List[Dict[str, Any]]:
    """
    Collapses a list of nested dictionary entries by merging the contents of a specified key into each parent dictionary. This is an in-place operation.
    
    :param entries: Description
    :type entries: List[Dict[str, Any]]
    :param collapse_key: Description
    :type collapse_key: str
    :return: Description
    :rtype: List[Dict[str, Any]]
    """
    for i, entry in enumerate(entries):
        collapsed_entry: Dict[str, Any] = collapse_entry(entry, collapse_key)
        entries[i] = collapsed_entry
    
    return entries