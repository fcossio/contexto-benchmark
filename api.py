"""Interface with the Contexto API"""

import time
import requests
import json
from typing import TypedDict, Literal
from loguru import logger

class AttemptResponse(TypedDict):
    """Response model when checking a word in the Contexto API"""
    word: str # the word that was checked.
    lemma: str # the "root" version of the word that is used for checking.
    distance: int # the distance of the lemma to the target word.

class ContextoApiClient:
    """Client for the Contexto API."""
    def __init__(self, language:Literal["en", "es", "pt-br"] = "en", host:str = "https://api.contexto.me"):
        self.language = language
        self.host = host
        self.url = f"{host}/machado/{language}/game/"

    def attempt(self, game_id: int, word: str, retries:int = 3) -> AttemptResponse | None:
        """Checks the distance of a word to the target word.
        Returns None if the word is not found.
        Raises an exception if there is an unknown error.
        """
        response = requests.get(f"{self.url}/{game_id}/{word}")
        
        if response.status_code == 200:
            try:
                return response.json()
            except json.JSONDecodeError:
                raise Exception(f"Invalid JSON in response: {response.text}")
        elif response.status_code == 503:
            # Contexto is busy, retry in 10 seconds
            if retries > 0:
                logger.warning(f"Contexto is busy. Retrying in 10 seconds. Retries left: {retries}")
                time.sleep(10)
                return self.attempt(game_id, word, retries - 1)
            else:
                raise Exception("Contexto is busy. Retried 3 times. Failed.")
        
        else:
            try:
                error_data = response.json()
                if response.status_code == 404 and error_data.get("error") in [
                    "I'm sorry, I don't know this word",
                    "This word doesn't count, it's too common",
                    "This word doesn't count",
                ]:
                    return None
                else:
                    raise Exception(f"Unknown error: {error_data.get('error', 'No error message provided')}")
            except json.JSONDecodeError:
                raise Exception(f"Error status code: {response.status_code}, Response not JSON: {response.text}")


    def top_words(self, game_id:int) -> list[str]:
        """Get the top words for a game. Position 0 is the target word.
        Obvously, don't use this method to get the target word when solving the game.
        """
        response = requests.get(f"{self.host}/machado/{self.language}/top/{game_id}")
        if response.status_code == 200:
            return response.json()["words"]
        else:
            raise Exception(f"Error status code: {response.status_code}, Response not JSON: {response.text}")

if __name__ == "__main__":
    client = ContextoApiClient()
    response = client.attempt(1, "hello")
    print(response)
