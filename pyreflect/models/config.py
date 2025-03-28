from dataclasses import dataclass, field
from pathlib import Path
import torch
import typer

from pyreflect.config.errors import *

DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
print(f'Selected device for model training: {DEVICE}')


def _resolve_file(root: str | Path, file_path: str | None):
    if file_path is None:
        msg = f"Config missing setting: {file_path}."
        raise ConfigMissingKeyError(msg)
    file_path = root / Path(file_path)

    return file_path

def _validate_file(file_path: Path)->Path:
    if not file_path.exists():
        msg = f"File not found: {file_path}"
        raise FileNotFoundError(msg)

    return file_path

@dataclass
class ChiPredTrainingParams:
    mod_expt_file: str | Path = None
    mod_sld_file: str | Path = None
    mod_params_file: str | Path = None
    batch_size: int = None
    latent_dim: int = None
    ae_epochs: int = None
    mlp_epochs: int = None
    _config: dict = field(default_factory=dict)

    def __post_init__(self):
        """
        Automatically extracts the required parameters from a nested YAML structure.
        """
        # Validate and extract nested fields
        if isinstance(self._config, dict) and "sld_predict_chi" in self._config:
            chi_section = self._config["sld_predict_chi"]

            try:
                root = self._config["root"]
                # Extract required parameters
                self.mod_expt_file = _resolve_file(root, chi_section["file"]["model_experimental_sld_profile"])
                self.mod_sld_file = _resolve_file(root, chi_section["file"]["model_sld_file"])
                self.mod_params_file = _resolve_file(root, chi_section["file"]["model_chi_params_file"])
                self.latent_dim = chi_section["models"]["latent_dim"]
                self.batch_size = chi_section["models"]["batch_size"]
                self.ae_epochs = chi_section["models"]["ae_epochs"]
                self.mlp_epochs = chi_section["models"]["mlp_epochs"]

                #Validation
                _validate_file(self.mod_expt_file)
                _validate_file(self.mod_sld_file)
                _validate_file(self.mod_params_file)

            except KeyError as e:
                raise ConfigMissingKeyError(e)

            # Store any extra parameters
            self.extra = {k: v for k, v in chi_section.items() if k not in ["file", "models"]}

        else:
            raise ConfigMissingKeyError("sld_predict_chi")


@dataclass
class NRSLDCurvesGeneratorParams:
    root: str | Path = Path(".")
    mod_nr_file: str | Path = None
    mod_sld_file: str | Path = None
    num_curves: int = None
    num_film_layers: int = None
    _config: dict = field(default_factory=dict)

    def __post_init__(self):
        """Extracts and validates required parameters from a nested YAML config."""
        if self._config and "nr_predict_sld" in self._config:
            nr_section = self._config["nr_predict_sld"]
            self.root = Path(self._config["root"])

            try:
                self.mod_nr_file = _resolve_file(self.root, nr_section["file"]["nr_curves_poly"])
                self.mod_sld_file = _resolve_file(self.root, nr_section["file"]["sld_curves_poly"])
                self.num_curves = nr_section["models"].get("num_curves", self.num_curves)
                self.num_film_layers = nr_section["models"].get("num_film_layers", self.num_film_layers)
            except KeyError as e:
                raise ConfigMissingKeyError(f"Missing key in NRSLDCurvesGeneratorParams: {e}")

        else:
            self.root = Path(self.root)
            # Direct input mode
            self.mod_nr_file = _resolve_file(self.root,self.mod_nr_file)
            self.mod_sld_file = _resolve_file(self.root,self.mod_sld_file)

        typer.echo(f"To be saved NR file:{self.mod_nr_file}")
        typer.echo(f"To be loaded SLD curves:{self.mod_sld_file}")


@dataclass
class NRSLDModelTrainerParams:
    """Handles parameters for training NR-SLD prediction models."""
    root:Path = Path(".")
    model_path: str|Path = None
    nr_file: str|Path = None
    sld_file: str|Path = None
    batch_size: int = None
    epochs: int = None
    layers: int = None
    # learning_rate: float = None
    _config: dict = field(default_factory=dict)

    def __post_init__(self):
        """Extracts and validates required parameters from a nested YAML config."""
        try:
            if self._config and "nr_predict_sld" in self._config:
                nr_section = self._config["nr_predict_sld"]

                root = Path(self._config["root"])
                self.root = root
                # path to save the generated data file and model
                self.model_path = _resolve_file(root,nr_section["models"]["model"])
                self.nr_file = _resolve_file(root, nr_section["file"]["nr_curves_poly"])
                self.sld_file = _resolve_file(root, nr_section["file"]["sld_curves_poly"])

                #file must exist
                _validate_file(self.nr_file)
                _validate_file(self.sld_file)

                # Model training parameters
                self.batch_size = nr_section["models"].get("batch_size", 32)
                self.epochs = nr_section["models"].get("epochs", 1)
                self.layers = nr_section["models"].get("layers", 12)

            else:
                self.root = Path(self.root)
                self.model_path = _resolve_file(self.root,self.model_path)
                self.nr_file = _resolve_file(self.root,self.nr_file)
                self.sld_file = _resolve_file(self.root,self.sld_file)

        except KeyError as e:
            raise ConfigMissingKeyError(f"Missing key in NRSLDModelTrainerParams: {e}")

from typing import List, Literal


@dataclass
class FilmLayer:
    sld: float
    isld: float = 0.0
    thickness: float = 0.0
    roughness: float = 0.0
    name: str = "layer"


@dataclass
class FilmModelDescription:
    layers: List[FilmLayer]
    scale: float = 1.0
    background: float = 0

@dataclass
class FilmLayerBound:
    i:int
    par:Literal["sld","thickness","roughness"]
    bounds:List[float]



