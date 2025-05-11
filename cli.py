import typer
import ray
import json
import numpy as np
import pandas as pd
from typing import Optional, List
from strategies import strategies
from environment import ContextoBase
from qdrant_client.models import Distance
from loguru import logger
import sys

# Configure logger to only show warnings and errors
logger.remove()  # Remove default handler
logger.add(sys.stderr, level="WARNING")  # Add new handler with WARNING level

app = typer.Typer()

def create_strategy(strategy_name: str, **kwargs) -> ContextoBase:
    """Create a strategy instance."""
    if strategy_name not in strategies:
        raise typer.BadParameter(f"Strategy {strategy_name} not found. Available strategies: {list(strategies.keys())}")
    return strategies[strategy_name](**kwargs)

@app.command()
def play(
    strategy_name: str = typer.Argument(..., help="Name of the strategy to use"),
    game_id: int = typer.Argument(..., help="Game ID to play"),
    language: str = typer.Option("en", help="Language to play in (en, es, pt-br)"),
    glove_fp: str = typer.Option("data/glove.42B.300d.txt", help="Path to GloVe embeddings file"),
    distance: str = typer.Option("COSINE", help="Distance metric to use (COSINE, EUCLID, DOT, MANHATTAN)"),
    embedding_size: int = typer.Option(300, help="Size of the embeddings"),
    word_list_fp: str = typer.Option("data/included_words_unique.txt", help="Path to word list file"),
    show_progress: bool = typer.Option(True, help="Whether to show progress"),
):
    """Play a single game and show progress."""
    strategy = create_strategy(
        strategy_name,
        language=language,
        glove_fp=glove_fp,
        distance=getattr(Distance, distance),
        embedding_size=embedding_size,
        word_list_fp=word_list_fp,
        show_progress=show_progress
    )
    attempts = strategy.play_game(game_id)
    print(f"Game completed in {attempts} attempts")

@ray.remote
def play_game_remote(strategy_name: str, game_id: int, **kwargs) -> int:
    """Remote function to play a game."""
    strategy = create_strategy(strategy_name, **kwargs)
    return strategy.play_game(game_id)

@app.command()
def eval(
    strategy_name: str = typer.Argument(..., help="Name of the strategy to use"),
    game_ids: List[int] = typer.Argument(..., help="List of game IDs to evaluate"),
    num_runs: int = typer.Option(3, help="Number of runs per game"),
    language: str = typer.Option("en", help="Language to play in (en, es, pt-br)"),
    glove_fp: str = typer.Option("data/glove.42B.300d.txt", help="Path to GloVe embeddings file"),
    distance: str = typer.Option("COSINE", help="Distance metric to use (COSINE, EUCLID, DOT, MANHATTAN)"),
    embedding_size: int = typer.Option(300, help="Size of the embeddings"),
    word_list_fp: str = typer.Option("data/included_words_unique.txt", help="Path to word list file"),
    show_progress: bool = typer.Option(False, help="Whether to show progress"),
):
    """Evaluate a strategy on multiple games in parallel."""
    # Initialize ray
    ray.init()
    
    # Create kwargs for strategy initialization
    kwargs = {
        "language": language,
        "glove_fp": glove_fp,
        "distance": getattr(Distance, distance),
        "embedding_size": embedding_size,
        "word_list_fp": word_list_fp,
        "show_progress": show_progress
    }
    
    # Create tasks for each game and run
    tasks = []
    for game_id in game_ids:
        for _ in range(num_runs):
            tasks.append(play_game_remote.remote(strategy_name, game_id, **kwargs))
    
    # Get results
    results = ray.get(tasks)
    
    # Process results
    game_results = []
    for i, game_id in enumerate(game_ids):
        game_attempts = results[i*num_runs:(i+1)*num_runs]
        game_results.append({
            "strategy": strategy_name,
            "game_id": game_id,
            "mean": float(np.mean(game_attempts)),
            "std": float(np.std(game_attempts)),
            "attempts": game_attempts
        })
    
    # Output results as JSON
    for r in game_results:
        print(json.dumps(r))
    
    # Shutdown ray
    ray.shutdown()

@app.command()
def report(
    input_file: str = typer.Option("data/benchmark-results.jsonl", help="Path to benchmark results JSONL file"),
    output_file: str = typer.Option("data/benchmark.md", help="Path to output markdown file"),
):
    """Create a markdown table from benchmark results."""
    # Read the JSONL file
    df = pd.read_json(input_file, lines=True)
    
    # Group by strategy and game_id, and calculate the mean of the mean and std
    summary = df.groupby(['strategy', 'game_id']).agg({
        'mean': 'mean',
        'std': 'mean'
    }).reset_index()
    
    # Pivot the table to have strategies as rows and game_ids as columns
    pivot_mean = summary.pivot(index='strategy', columns='game_id', values='mean')
    pivot_std = summary.pivot(index='strategy', columns='game_id', values='std')
    
    # Calculate the average across all games
    pivot_mean['avg'] = pivot_mean.mean(axis=1)
    
    # Format the table with mean ± std
    pivot_table = pd.DataFrame(index=pivot_mean.index)
    for col in pivot_mean.columns:
        if col == 'avg':
            pivot_table[f'Game Avg'] = pivot_mean[col].map(lambda x: f"{int(round(x))}")
        else:
            pivot_table[f'Game {col}'] = pivot_mean[col].map(lambda x: f"{int(round(x))}") + " ± " + pivot_std[col].map(lambda x: f"{int(round(x))}")
    
    # Create a numeric column for sorting
    pivot_table['sort_value'] = pivot_mean['avg']
    
    # Sort by the average score (lower is better)
    pivot_table = pivot_table.sort_values(by='sort_value', ascending=True)
    
    # Remove the sort column before generating markdown
    pivot_table = pivot_table.drop(columns=['sort_value'])
    
    # rename index to Strategy
    pivot_table.index.name = 'Strategy'
    # Create markdown table
    markdown = f"# Contexto Benchmark Results\n\n"
    markdown += f"Results show the average number of attempts needed to solve each game (lower is better).\n\n"
    markdown += pivot_table.to_markdown()
    
    # Write to file
    with open(output_file, 'w') as f:
        f.write(markdown)
    
    print(f"Benchmark table created at {output_file}")

if __name__ == "__main__":
    app()
