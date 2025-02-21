from dataclasses import dataclass

@dataclass
class ChiPredTrainingParams:
    mod_expt_file:str
    mod_sld_file:str
    mod_params_file:str
    batch_size:int
    latent_dim:int
    ae_epochs: int
    mlp_epochs:int
