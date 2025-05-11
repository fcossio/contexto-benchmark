from qdrant_client import QdrantClient
from qdrant_client.models import (
    VectorParams,
    PointStruct,
    Filter,
    FieldCondition,
    MatchValue,
    Distance,
    ExtendedPointId,
    Record,
    QueryRequest,
    ContextQuery,
    ContextPair,
    VectorParams,
    SampleQuery,
    Sample,
    HasIdCondition,
)
from tqdm import tqdm
import numpy as np
from typing import Sequence

from loguru import logger


class GloVeData:
    """A collection of IO methods for GloVe embeddings and the db collection."""

    @staticmethod
    def build(
        qdrant_client: QdrantClient,
        glove_fp: str,
        word_list_fp: str,
        distance: Distance,
        embedding_size: int,
        common_words_fp: str,
    ):
        """Build a qdrant collection with the GloVe embeddings in the given db handle."""

        glove_words = GloVeData._read_glove_words(glove_fp)
        words = GloVeData._read_word_list(word_list_fp)
        words_idx, vectors = GloVeData._read_glove_vectors(words, glove_words, glove_fp)
        commonness = GloVeData._read_common_words(common_words_fp)

        qdrant_client.create_collection(
            collection_name="glove",
            vectors_config=VectorParams(size=embedding_size, distance=distance),
        )
        qdrant_client.upsert(
            "glove",
            points=[
                PointStruct(
                    id=idx,
                    vector=vec.tolist(),
                    payload={"word": word, "in_game": True, "game_distance": 0, "freq": commonness.get(word, 0.0)},
                )
                for idx, (word, vec) in enumerate(zip(words_idx, vectors))
            ],
        )

    @staticmethod
    def get_vector_distances(qdrant_client: QdrantClient, word: str, top_candidates: list[str]) -> dict[str, float]:
        """Get the distances to all the words in the collection that have not been guessed yet."""
        records = GloVeData.get_records_by_words(qdrant_client, [word] + top_candidates)
        record = records[0]
        all_ids = [r.id for r in records]
        results = qdrant_client.search(
            collection_name="glove",
            query_vector=record.vector,
            with_vectors=False,
            with_payload=True,
            limit=len(all_ids),
            query_filter=Filter(
                must=[
                    FieldCondition(key="in_game", match=MatchValue(value=True)),
                    FieldCondition(key="game_distance", match=MatchValue(value=0)),
                    HasIdCondition(has_id=all_ids)]
            ),
        )
        return {r.payload["word"]: r.score for r in results if r.payload and "word" in r.payload}

    @staticmethod
    def nearest(
        qdrant_client: QdrantClient, vector: Sequence[float], limit: int = 10
    ) -> list[str]:
        """Get the nearest words to the given vector."""
        results = qdrant_client.search(
            collection_name="glove",
            query_vector=vector,
            limit=limit,
            with_vectors=False,
            with_payload=True,
            query_filter=Filter(
                must=[
                    FieldCondition(key="in_game", match=MatchValue(value=True)),
                    FieldCondition(key="game_distance", match=MatchValue(value=0)),
                ]
            ),
        )
        return [r.payload["word"] for r in results if r.payload and "word" in r.payload]

    @staticmethod
    def nearest_by_word(
        qdrant_client: QdrantClient, word: str, limit: int = 10
    ) -> list[str]:
        """Get the nearest words to the given word."""
        record = GloVeData.get_records_by_words(qdrant_client, [word])[0]
        return GloVeData.nearest(qdrant_client, record.vector, limit=limit)

    @staticmethod
    def set_distance(qdrant_client: QdrantClient, word: str, distance: int):
        """Set the distance for a word."""
        logger.info(f"Setting distance for word: {word} to {distance}")
        qdrant_client.set_payload(
            collection_name="glove",
            payload={"game_distance": distance},
            points=Filter(
                must=[FieldCondition(key="word", match=MatchValue(value=word))]
            ),
        )
    
    @staticmethod
    def sample(qdrant_client: QdrantClient, n: int):
        """Use Qdrant's random sampling API to get n random points."""
        sampled = qdrant_client.query_points(
            collection_name="glove",
            query=SampleQuery(sample=Sample.RANDOM),
            query_filter=Filter(
                must=[
                    FieldCondition(key="in_game", match=MatchValue(value=True)),
                    FieldCondition(key="game_distance", match=MatchValue(value=0)),
                ]
            ),
            with_payload=True,
            limit=n
        )
        return [p.payload["word"] for p in sampled.points if p.payload]

    @staticmethod
    def disable_word(qdrant_client: QdrantClient, word: str):
        """Disable a word."""
        logger.warning(f"Disabling word: {word}")
        qdrant_client.set_payload(
            collection_name="glove",
            payload={"in_game": False},
            points=Filter(
                must=[FieldCondition(key="word", match=MatchValue(value=word))]
            ),
        )
        # write to the file
        with open("data/excluded_words.txt", "a") as f:
            f.write(word + "\n")

    @staticmethod
    def get_ids(qdrant_client: QdrantClient):
        """Get all ids in the collection that are in the game."""
        # When the game starts, all words have game_distance=0, but they are in_game=True
        # We need to get words that are in_game but don't have a distance yet, or have a non-zero distance
        points, _ = qdrant_client.scroll(
            "glove",
            with_vectors=False,
            with_payload=True,
            limit=100,  # Start with a smaller batch
            scroll_filter=Filter(
                must=[
                    FieldCondition(key="in_game", match=MatchValue(value=True)),
                    FieldCondition(key="game_distance", match=MatchValue(value=0)),
                ]
            ),
        )

        return [point.id for point in points]

    @staticmethod
    def get_record_by_id(qdrant_client: QdrantClient, id: ExtendedPointId) -> Record:
        """Get the vector for a given id."""
        points = qdrant_client.retrieve("glove", [id])
        if len(points) == 0:
            raise Exception(f"No point found for id {id}")
        return points[0]

    @staticmethod
    def get_records_by_words(
        qdrant_client: QdrantClient, words: list[str]
    ) -> list[Record]:
        """Get the records for a given word."""
        records = []
        for word in words:
            r, _ = qdrant_client.scroll(
                "glove",
                with_vectors=True,
                with_payload=True,
                scroll_filter=Filter(
                    must=[FieldCondition(key="word", match=MatchValue(value=word))]
                ),
            )
            if len(r) == 0:
                raise Exception(f"No point found for word {word}")
            records.extend(r)
        return records

    @staticmethod
    def get_sorted_records(qdrant_client: QdrantClient) -> list[Record]:
        """Get all the records with a distance greater than 0."""
        records, _ = qdrant_client.scroll(
            "glove",
            with_vectors=False,
            with_payload=True,
            scroll_filter=Filter(
                must_not=[
                    FieldCondition(key="game_distance", match=MatchValue(value=0)),
                    FieldCondition(key="in_game", match=MatchValue(value=False)),
                ]
            ),
        )
        # sort points by game_distance
        records.sort(
            key=lambda x: x.payload["game_distance"] if x.payload is not None else 0
        )
        return records

    @staticmethod
    def context_search(
        qdrant_client: QdrantClient,
        positive: Record,
        negatives: list[Record],
        limit: int,
    ) -> tuple[list[str], list[float]]:
        """Use context search to rerank the nearest neighbors to the positive word.
        Since many get the best score (0.0). Use context inversely to get the best candidates.
        Use the commonness of the words to re
        """
        
        # search results close to the positive
        response = qdrant_client.query_points(
            query=positive.vector,
            collection_name="glove",
            limit=limit,
            with_vectors=False,
            with_payload=False,
            query_filter=Filter(
                must=[
                    FieldCondition(key="in_game", match=MatchValue(value=True)),
                    FieldCondition(key="game_distance", match=MatchValue(value=0)),
                ]
            ),
        )

        # rerank the results based on the context search and commonness of the words

        context_pairs = [ # this is inverted to be able to get some signal.
            ContextPair(positive=negative.id, negative=positive.id)
            for negative in negatives
        ]

        all_ids = [p.id for p in response.points]
        all_ids += [negative.id for negative in negatives]
        all_ids += [positive.id]


        query_request = QueryRequest(
            query=ContextQuery(context=context_pairs),
            with_payload=True,
            limit=limit,
            filter=Filter(
                must=[
                    FieldCondition(key="in_game", match=MatchValue(value=True)),
                    FieldCondition(key="game_distance", match=MatchValue(value=0)),
                    HasIdCondition(has_id=all_ids)
                ]
            ),
        )

        # Execute the query
        responses = qdrant_client.query_batch_points(
            collection_name="glove", requests=[query_request]
        )

        # get the scores and words
        # minus score because we inverted the context pairs
        scores = [-p.score for p in responses[0].points if p.score is not None]
        freqs = [r.payload["freq"] for r in responses[0].points if r.payload]
        words = [r.payload["word"] for r in responses[0].points if r.payload]



        # get the highest score to normalize the scores
        max_score = max(scores)
        scores = [s / max_score for s in scores]

        # normalize the freqs
        max_freq = max(freqs)
        freqs = [float(f > 0.0) for f in freqs]

        # weighted sorted scores
        scores = [s + 0.05 * f for s, f in zip(scores, freqs)]

        # sort the words by the scores
        order = np.argsort(scores)[::-1] # descending
        words = [words[i] for i in order]
        scores = [scores[i] for i in order]

        return words, scores


    @staticmethod
    def context_search_stochastic(
        qdrant_client: QdrantClient,
        head: Record,
        pairs: list[tuple[Record, Record]],
        limit: int,
    ) -> tuple[list[str], list[float]]:
        """Use context search to rerank the nearest neighbors to the positive word.
        Since many get the best score (0.0). Use context inversely to get the best candidates.
        Use the commonness of the words to re
        """
        
        # search results close to the positive
        response = qdrant_client.query_points(
            query=head.vector,
            collection_name="glove",
            limit=limit,
            with_vectors=False,
            with_payload=False,
            query_filter=Filter(
                must=[
                    FieldCondition(key="in_game", match=MatchValue(value=True)),
                    FieldCondition(key="game_distance", match=MatchValue(value=0)),
                ]
            ),
        )

        # rerank the results based on the context search and commonness of the words

        context_pairs = [ # this is inverted to be able to get some signal.
            ContextPair(positive=negative.id, negative=positive.id)
            for positive, negative in pairs
        ]

        all_ids = [p.id for p in response.points]
        all_ids += [negative.id for positive, negative in pairs]
        all_ids += [positive.id for positive, negative in pairs]
        all_ids += [head.id]
        all_ids = list(set(all_ids))


        query_request = QueryRequest(
            query=ContextQuery(context=context_pairs),
            with_payload=True,
            limit=limit,
            filter=Filter(
                must=[
                    FieldCondition(key="in_game", match=MatchValue(value=True)),
                    FieldCondition(key="game_distance", match=MatchValue(value=0)),
                    HasIdCondition(has_id=all_ids)
                ]
            ),
        )

        # Execute the query
        responses = qdrant_client.query_batch_points(
            collection_name="glove", requests=[query_request]
        )

        # get the scores and words
        # minus score because we inverted the context pairs
        scores = [-p.score for p in responses[0].points if p.score is not None]
        freqs = [r.payload["freq"] for r in responses[0].points if r.payload]
        words = [r.payload["word"] for r in responses[0].points if r.payload]



        # get the highest score to normalize the scores
        max_score = max(scores)
        scores = [s / max_score for s in scores]

        # normalize the freqs
        max_freq = max(freqs)
        freqs = [float(f > 0.0) for f in freqs]

        # weighted sorted scores
        scores = [s + 0.05 * f for s, f in zip(scores, freqs)]

        # sort the words by the scores
        order = np.argsort(scores)[::-1] # descending
        words = [words[i] for i in order]
        scores = [scores[i] for i in order]

        return words, scores

    @staticmethod
    def restart_game(qdrant_client: QdrantClient):
        """Reset the game_distance for all the words."""
        logger.info("Resetting game state in database")
        # Reset all words to be in_game and have game_distance=0
        qdrant_client.set_payload(
            collection_name="glove",
            payload={"game_distance": 0},
            points=Filter(),  # Empty filter to match all points
        )
        logger.info("Game state reset complete")

    @staticmethod
    def _read_glove_words(glove_fp) -> dict[str, int]:
        """Read only the words (not the vectors) from the GloVe file.
        Returns a dictionary with the word as the key and the line number as the value.
        """
        d = dict()
        with open(glove_fp) as f:
            for i, line in tqdm(
                enumerate(f), desc=f"Reading strings in GloVe file {glove_fp}"
            ):
                end = line.find(" ")
                word = line[:end]
                d[word] = i
        return d

    @staticmethod
    def _read_word_list(word_list_fp):
        """Read list of all the words that should be used"""
        logger.info(f"Reading word list from {word_list_fp}")
        words = set()
        with open(word_list_fp) as f:
            for line in f:
                words.add(line[:-1])
        with open("data/excluded_words.txt") as f:
            for line in f:
                words.discard(line[:-1])
        return words

    @staticmethod
    def _line_to_vector(line: str) -> tuple[str, np.ndarray]:
        """Convert a line from the GloVe file to a word and a vector."""
        end = line.find(" ")
        return line[:end], np.fromstring(line[end + 1 :], sep=" ")

    @staticmethod
    def _read_common_words(common_words_fp: str) -> dict[str, float]:
        commonness = {}
        if common_words_fp:
            with open(common_words_fp) as f:
                # skip the header
                next(f)
                for line in f:
                    word, freq = line.split(",")
                    commonness[word] = float(freq)
        return commonness

    @staticmethod
    def _read_glove_vectors(words, glove_strings, glove_fp):
        """Read the vectors for the words that are in the glove file and the word list."""
        logger.info(f"Reading vectors for {len(words)} words")
        available_words = words.intersection(glove_strings.keys())
        
        indices = []
        vectors = []
        words_idx = []
        for word in available_words:
            indices.append(glove_strings[word])
        indices = sorted(indices)
        idx_iter = iter(indices)
        next_idx = next(idx_iter)
        with open(glove_fp) as f:
            for i, line in enumerate(f):
                if i < next_idx:
                    continue
                elif i == next_idx:
                    word, vector = GloVeData._line_to_vector(line)
                    words_idx.append(word)
                    vectors.append(vector)
                    try:
                        next_idx = next(idx_iter)
                    except StopIteration:
                        break
                else:
                    raise Exception("This should not happen something is wrong.")
        return words_idx, np.stack(vectors)
