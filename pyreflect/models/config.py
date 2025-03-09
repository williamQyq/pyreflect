from dataclasses import dataclass, field
from pathlib import Path
import torch
from pyreflect.config.errors import *

DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

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
                self.mod_expt_file = self._resolve_file(root, chi_section["file"]["model_experimental_sld_profile"])
                self.mod_sld_file = self._resolve_file(root, chi_section["file"]["model_sld_file"])
                self.mod_params_file = self._resolve_file(root, chi_section["file"]["model_chi_params_file"])
                self.latent_dim = chi_section["models"]["latent_dim"]
                self.batch_size = chi_section["models"]["batch_size"]
                self.ae_epochs = chi_section["models"]["ae_epochs"]
                self.mlp_epochs = chi_section["models"]["mlp_epochs"]

            except KeyError as e:
                raise ConfigMissingKeyError(e)

            # Store any extra parameters
            self.extra = {k: v for k, v in chi_section.items() if k not in ["file", "models"]}

        else:
            raise ConfigMissingKeyError("sld_predict_chi")

    def _resolve_file(self, root: str | Path, file_path:str | Path):
        file_path = root / Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"Required file not found: {file_path}")

        return file_path

class NRSLDCurvesGeneratorParams:
    expt_nr_file: str | Path = None
    mod_nr_file: str | Path = None
    mod_sld_file: str | Path = None