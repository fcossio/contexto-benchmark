from enum import Enum
import random
from typing import Literal
from typing_extensions import TypedDict
import shutil
from rich.console import Console
from rich.layout import Layout
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from rich.live import Live

from qdrant_client import QdrantClient
from qdrant_client.models import Distance
from glove import GloVeData
from api import ContextoApiClient, AttemptResponse
import numpy as np
from api import ContextoApiClient

from loguru import logger

Status = Literal["valid_guess", "invalid_guess", "target_found"]

class ContextoBase:
    """
    A bot that plays the Contexto game. 
    Create your own subclass to implement your own logic by overriding the `take_turn` method.
    """

    MAX_ATTEMPTS = 500
    
    def __init__(self,
        language:Literal["en", "es", "pt-br"] = "en",
        glove_fp:str = "data/glove.42B.300d.txt", 
        distance:Distance = Distance.COSINE,
        embedding_size:int = 300,
        word_list_fp:str = "data/included_words_unique.txt",
        common_words_fp:str = "data/common_words.csv",
        qdrant_kwargs:dict={"location":":memory:"},
        show_progress:bool = True,
    ):

        """Initialize the environment.
        
        Args:
            language: The language to play the game in.
            glove_fp: The path to the GloVe embeddings file.
            distance: The distance metric to use.
            embedding_size: The size of the embeddings.
            word_list_fp: The path to the word list file.
            common_words_fp: The path to the common words file.
            qdrant_kwargs: Arguments to pass to QdrantClient.
            show_progress: Whether to show progress.
        """
        logger.info("Initializing db")
        self.language = language
        self.game_client = ContextoApiClient(language=language)
        self.qdrant = QdrantClient(**qdrant_kwargs)
        self.api = ContextoApiClient(language=language)
        self.show_progress = show_progress
        self.live_display = None
        GloVeData.build(self.qdrant, glove_fp, word_list_fp, distance, embedding_size, common_words_fp)

    def _reset_game(self):
        # reset the game
        self.attempts = 0
        self.best_distance = np.inf
        self.ranking = Ranking()
        self.last_guess = ""
        GloVeData.restart_game(self.qdrant)

    def take_turn(self) -> Status:
        """Take a turn. This method should be implemented by the subclass."""
        #####################
        # Custom logic here #
        #####################
        raise NotImplementedError("This method should be implemented by the subclass.")
        status = self.make_guess("hello")
        return status

    def play_game(self, game_id:int) -> int:
        """Play the game."""
        self.game_id = game_id
        self._reset_game()
        
        # Initialize live display if showing progress
        if self.show_progress:
            self.live_display = Live(self._generate_dashboard(), auto_refresh=True, screen=False)
            self.live_display.start()

        try:
            # play the game
            while self.attempts < self.MAX_ATTEMPTS and self.best_distance > 0:
                status = self.take_turn()
                if status == "invalid_guess": # invalid word skip to the next guess
                    continue
                if status == "target_found":
                    break

                # Update the dashboard
                if self.show_progress and self.live_display:
                    self.live_display.update(self._generate_dashboard())

            # Final update
            if self.show_progress and self.live_display:
                self.live_display.update(self._generate_dashboard())
                
            return self.attempts
        
        finally:
            # Make sure to stop the live display
            if self.show_progress and self.live_display:
                self.live_display.stop()

    def make_guess(self, word:str) -> Status:
        """Make a guess."""
        self.last_guess = word
        
        logger.debug(f"Making guess: {word}")
        response = self.api.attempt(self.game_id, word)

        if response == None: # the word is not in the game
            GloVeData.disable_word(self.qdrant, word)
            return "invalid_guess"

        if word != response.get("lemma"): # the word maps to a different word in the game
            GloVeData.disable_word(self.qdrant, word)
            word = response.get("lemma") # valid guess that maps to another word

        # update the distance, valid_attempts, and the counter attempts
        distance = response.get("distance")
        GloVeData.set_distance(self.qdrant, word, distance)
        self.ranking.add(Attempt(word=word, distance=distance))
        self.attempts += 1
        
        if distance < self.best_distance:
            self.best_distance = distance
            logger.debug(f"New best distance: {word} {self.best_distance}")

        if distance == 0:
            return "target_found"
        
        return "valid_guess"
                    

    def _generate_dashboard(self):
        """Generate a Rich dashboard with game information."""
        layout = Layout()
        
        # Get terminal size
        terminal_width = shutil.get_terminal_size().columns
        
        # Create header panel with game info
        header = Panel(
            Text(f"Game #{self.game_id} | Attempts: {self.attempts}/{self.MAX_ATTEMPTS} | Best Distance: {self.best_distance if self.best_distance != np.inf else 'N/A'}", justify="center"),
            style="bold blue"
        )
        
        # Create latest attempt panel if we have a last guess
        if self.last_guess and self.last_guess in self.ranking:
            last_distance = self.ranking[self.last_guess]["distance"]
            
            latest_attempt = Panel(
                Text(f"{self.last_guess} → {last_distance}", justify="center", style="bold"),
                style="green" if last_distance <= self.best_distance else "yellow", 
                title="Latest Attempt"
            )
        else:
            latest_attempt = Panel(Text("No attempts yet", justify="center"), style="dim")
        
        # Create table for valid attempts - horizontal layout
        table = Table(box=None, expand=True)
        
        # Calculate how many columns we can fit (each column needs ~15 chars)
        max_cols = min(len(self.ranking), (terminal_width - 10) // 15)
        
        # Add sorted attempts to table
        sorted_attempts = self.ranking.sorted()[:max_cols]
        
        # Add column headers (one for each word)
        for attempt in sorted_attempts:
            table.add_column(attempt["word"], justify="left", style="cyan")
        
        # Add row of distances
        distances = [str(attempt["distance"]) for attempt in sorted_attempts]
        if distances:
            table.add_row(*distances)
        
        # Create layout with three sections
        layout.split(
            Layout(header, size=3),
            Layout(latest_attempt, size=3),
            Layout(Panel(table, title="Best Attempts", border_style="blue") if distances else None)
        )
        
        return layout
    
    def display_dashboard(self):
        """Display or update the dashboard."""
        if self.live_display:
            self.live_display.update(self._generate_dashboard())
        else:
            console = Console()
            console.print(self._generate_dashboard())

Attempt = TypedDict("Attempt", {"word":str, "distance":int})


class Ranking(dict):
    # TODO: make interface a simple dict
    """A dictionary of valid attempts"""
    def __init__(self):
        super().__init__()

    def add(self, attempt:Attempt):
        self[attempt["word"]] = attempt

    def sorted(self):
        return sorted(self.values(), key=lambda x: x["distance"])
    
    def log_best(self):
        records = self.sorted()[:min(len(self), 5)]
        for r in records:
            logger.success(f"{r['word']} \t\t {r['distance']:05d}")



if __name__ == "__main__":
    bot = ContextoBase(language="en")
    bot.play_game(game_id=2)