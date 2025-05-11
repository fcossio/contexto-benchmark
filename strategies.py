from collections import defaultdict
import itertools
import random
from typing import Type

import numpy as np
from environment import ContextoBase, Status
from glove import GloVeData

strategies:dict[str, Type[ContextoBase]] = {}

def register_strategy(name:str, strategy:Type[ContextoBase]):
    """Register a strategy."""
    strategies[name] = strategy

class NearestDescent(ContextoBase):
    """Simplest algorithm. 
    
    1. Choose a random word.
    2. Query for the NN of the nearest word, filter out the words that have been used before.
    3. Make a guess.
    """
    N = 1
    def take_turn(self) -> Status:
        if len(self.ranking) < self.N:
            # choose a random word
            random_word = GloVeData.sample(self.qdrant, 1)[0]
            return self.make_guess(random_word)

        closest = self.ranking.sorted()[0]
        neighbor = GloVeData.nearest_by_word(self.qdrant, closest["word"], 1)[0]
        return self.make_guess(neighbor)
register_strategy("nearest_descent", NearestDescent)


class NearestDescentMultipleStart(NearestDescent):
    """Same spirit as NearestDescent, but with a multiple start strategy."""
    N = 10
register_strategy("nearest_descent_10_starts", NearestDescentMultipleStart)


class ContextSearch(ContextoBase):
    """Use Qdrant's context search API to `siege` the target word.
    
    1. Start by choosing 5 random words.
    2. Use the head word as positive and the next 5 as negatives to create multiple hyperplanes to 
       guide the search.
    3. Factor in the frequency of the words to the score.
    4. When close enough, just do a greedy search, it usually converges faster.
    """
    N = 5
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.last_top_N = []

    def get_context(self, force:bool=False) -> tuple[list[str], list[float]]:
        """Check if the first N valid attempts changed since the last context search.
        If so, perform a context search and return the results.
        """
        sorted_words = [a['word'] for a in self.ranking.sorted()]
        if any(
            [sorted_words[i] != self.last_top_N[i] for i in range(
                min(self.N, len(self.last_top_N)))]
            ) or len(self.last_top_N) == 0 or force:
            self.last_top_N = sorted_words[:self.N]
            sorted_records = GloVeData.get_records_by_words(self.qdrant, sorted_words)
            positive = sorted_records[0]
            negatives = sorted_records[1:5]
            candidates, scores = GloVeData.context_search(self.qdrant, positive, negatives, 100)
            self.context_cache = candidates, scores
        else:
            candidates, scores = self.context_cache
            candidates = candidates[1:]
            scores = scores[1:]
            self.context_cache = candidates, scores
        return candidates, scores
        
    def take_turn(self) -> Status:
        if len(self.ranking) < self.N:
            random_word = GloVeData.sample(self.qdrant, 1)[0]
            return self.make_guess(random_word)
        if self.best_distance > 40:
            # we are far from the target word, use context search
            candidates, scores = self.get_context()
            if len(candidates) == 0:
                # force the context to be regenerated
                candidates, scores = self.get_context(force=True)
            return self.make_guess(candidates[0])
        else:
            # we are close to the target word, use the nearest descent strategy 80% of the time
            if random.random() < 0.8:
                closest = self.ranking.sorted()[0]
                neighbor = GloVeData.nearest_by_word(self.qdrant, closest["word"], 1)[0]
                return self.make_guess(neighbor)
            else:
                candidates, scores = self.get_context()
                return self.make_guess(candidates[0])
register_strategy("context_search", ContextSearch)

class ContextSearchStochastic(ContextoBase):
    """Same as above but this time not only the head can be used as positive, the only 
    condition is that the positive must have smaller distance than the negative.

    Drawback is that it needs to do the context search at every step.
    """
    N = 5
    N_PAIRS = 10

    def get_context(self) -> tuple[list[str], list[float]]:
        sorted_words = [a['word'] for a in self.ranking.sorted()]
        sorted_records = GloVeData.get_records_by_words(self.qdrant, sorted_words)

        head = sorted_records[0]
        pairs = []

        # pick 10 random pairs of words that are sorted (lower, higher)
        combinations = list(itertools.combinations(sorted_records, 2))
        pairs = np.random.choice(np.arange(len(combinations)), size=self.N_PAIRS, replace=False)
        pairs = [combinations[i] for i in pairs]

        candidates, scores = GloVeData.context_search_stochastic(self.qdrant, head, pairs, 100)
    
        return candidates, scores
        
    def take_turn(self) -> Status:
        if len(self.ranking) < self.N:
            random_word = GloVeData.sample(self.qdrant, 1)[0]
            return self.make_guess(random_word)
        if self.best_distance > 40:
            # we are far from the target word, use context search
            candidates, scores = self.get_context()
            return self.make_guess(candidates[0])
        else:
            # we are close to the target word, use the nearest descent strategy 80% of the time
            if random.random() < 0.8:
                closest = self.ranking.sorted()[0]
                neighbor = GloVeData.nearest_by_word(self.qdrant, closest["word"], 1)[0]
                return self.make_guess(neighbor)
            else:
                candidates, scores = self.get_context()
                return self.make_guess(candidates[0])

register_strategy("context_search_stochastic", ContextSearchStochastic)

        