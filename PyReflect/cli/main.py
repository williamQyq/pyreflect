import yaml
from pathlib import Path
from typing import Annotated

import typer
from pyreflect.cli.index import process_data

app = typer.Typer(help="A CLI tool for neutron reflectivity data processing.")

# Command to initialize a configuration file
@app.command("init")
def init_settings(
    directory: Annotated[
        Path,
        typer.Option(
            help="The project root directory.",
            dir_okay=True,
            writable=True,
            resolve_path=True,
        ),
    ] = Path("."),
    force: bool = typer.Option(
        False, "--force", help="Force initialization even if settings.yml already exists."
    ),
):
    """Create a default settings.yml file."""
    config_path = directory / "settings.yml"

    if config_path.exists() and not force:
        typer.echo(f"Settings file already exists at {config_path}. Use --force to overwrite.")
        raise typer.Exit()

    # Default settings
    default_settings = {
        "mod_expt_file": "data/mod_expt.npy",
        "mod_sld_file": "data/mod_sld.npy",
        "mod_params_file": "data/mod_params.npy",

        "latent_dim":2
    }

    with open(config_path, "w") as f:
        yaml.dump(default_settings, f)

    typer.echo(f"Initialized settings file at {config_path}.")


# Command to run the data processing
@app.command("run")
def run_analysis(
    config: Annotated[
        Path,
        typer.Option(help="Path to the settings.yml file.", exists=True, readable=True),
    ] = Path("settings.yml")
):
    """Run data analysis using the specified settings."""
    # Load settings from the YAML file
    with open(config, "r") as f:
        settings = yaml.safe_load(f)

    # Extract file paths from the settings
    mod_expt_file = settings.get("mod_expt_file")
    mod_sld_file = settings.get("mod_sld_file")
    mod_params_file = settings.get("mod_params_file")

    if not (mod_expt_file and mod_sld_file and mod_params_file):
        typer.echo("Invalid settings file. Ensure all file paths are specified.")
        raise typer.Exit()

    # Run the main data processing function
    process_data(mod_expt_file, mod_sld_file, mod_params_file)

    # model = train_model()

    # model.predict()

if __name__ == "__main__":
    app()
