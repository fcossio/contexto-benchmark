# Download and prepare data for Contexto solver
# This script downloads GloVe 42B 300d embeddings and common words data

import os
import pandas as pd
import typer
from typing import Optional
from typing_extensions import Annotated

app = typer.Typer(help="Prepare data for the benchmark.")

@app.command()
def download_embeddings(
    force: Annotated[bool, typer.Option(help="Force download even if files exist")] = False
):
    """Download GloVe 42B 300d embeddings."""
    target_zip = "data/glove.42B.300d.zip"
    target_txt = "data/glove.42B.300d.txt"
    
    if os.path.exists(target_txt) and not force:
        typer.echo(f"Embeddings file already exists at {target_txt}. Use --force to download again.")
        return
    
    os.makedirs("data", exist_ok=True)
    typer.echo("Downloading GloVe 42B 300d embeddings...")
    result = os.system(f"wget https://huggingface.co/stanfordnlp/glove/resolve/main/glove.42B.300d.zip -O {target_zip}")
    
    if result != 0:
        typer.echo("Download failed!", err=True)
        raise typer.Exit(code=1)
    
    typer.echo("Extracting GloVe 42B 300d embeddings...")
    result = os.system(f"unzip -o {target_zip} -d data/")
    
    if result != 0:
        typer.echo("Extraction failed!", err=True)
        raise typer.Exit(code=1)
    
    # Delete zip file after extraction
    os.remove(target_zip)
    typer.echo(f"Deleted {target_zip}")
    
    typer.echo("GloVe embeddings downloaded and extracted successfully!")

@app.command()
def common_words(
    force: Annotated[bool, typer.Option(help="Force download even if files exist")] = False,
    process_only: Annotated[bool, typer.Option(help="Skip download and only process existing file")] = False
):
    """Download and process common words data."""
    target_xlsx = "data/common_words.xlsx"
    target_csv = "data/common_words.csv"
    
    os.makedirs("data", exist_ok=True)
    
    # Handle download
    if not process_only:
        if os.path.exists(target_xlsx) and not force:
            typer.echo(f"Common words Excel file already exists. Skipping download.")
        else:
            typer.echo("Downloading common words data...")
            result = os.system(f"wget https://www.wordfrequency.info/samples/wordFrequency.xlsx -O {target_xlsx}")
            
            if result != 0:
                typer.echo("Download failed!", err=True)
                raise typer.Exit(code=1)
            typer.echo("Common words data downloaded successfully!")
    
    # Process the data
    if not os.path.exists(target_xlsx):
        typer.echo(f"Common words file not found at {target_xlsx}. Run without --process-only first.", err=True)
        raise typer.Exit(code=1)
    
    if os.path.exists(target_csv) and not force:
        typer.echo(f"Processed common words file already exists at {target_csv}. Use --force to process again.")
        return
        
    typer.echo("Processing common words data...")
    try:
        df = pd.read_excel(target_xlsx, sheet_name="1 lemmas")
        df = df[['lemma', 'freq']]
        df['freq'] = df['freq']/df['freq'].max()
        df.to_csv(target_csv, index=False)
        
        # Delete xlsx file after processing
        os.remove(target_xlsx)
        typer.echo(f"Deleted {target_xlsx}")
        
        typer.echo(f"Common words data processed and saved to {target_csv}")
    except Exception as e:
        typer.echo(f"Error processing common words data: {e}", err=True)
        raise typer.Exit(code=1)

@app.command()
def download_all(
    force: Annotated[bool, typer.Option(help="Force download even if files exist")] = False
):
    """Download and process all required data."""
    download_embeddings(force)
    common_words(force)
    typer.echo("All data preparation completed!")

if __name__ == "__main__":
    app()


