from typing import Annotated, TypedDict, List

class PreferenceState(TypedDict):
    # The 'Source of Truth' for user vibes
    preferences: dict 
    # Current itinerary draft
    itinerary: str
    # Chat history
    messages: Annotated[List[dict], "The conversation history"]
    # Internal critique of why the sync might be failing
    sync_status: str
