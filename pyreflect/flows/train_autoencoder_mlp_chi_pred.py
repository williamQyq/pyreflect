from pyreflect.input import SLDChiDataProcessor
from pyreflect.models.chi_pred_model_trainer import ModelTrainer
from pyreflect.models import mlp, autoencoder as ae
from pyreflect.models.config import ChiPredTrainingParams
import torch

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

def run_model_training(config: ChiPredTrainingParams):
    print(f"Processing data with:")
    print(f"  - mod_expt_file: {config.mod_expt_file}")
    print(f"  - mod_sld_file: {config.mod_sld_file}")
    print(f"  - mod_params_file: {config.mod_params_file}")
    print(f"  - batch_size: {config.batch_size}")

    # init processor
    data_processor = SLDChiDataProcessor(config.mod_expt_file, config.mod_sld_file, config.mod_params_file)
    # load data from file path
    data_processor.load_data()
    # remove flatten data and normalize
    data_processor.preprocess_data()

    #Fix Error for mat mult in VAE
    sld_arr_cut = data_processor.sld_arr
    sld_arr_cut_flat = sld_arr_cut.reshape(sld_arr_cut.shape[0], -1)

    # training, validation, testing arrays
    list_arrays = data_processor.split_arrays(sld_arr_cut_flat, data_processor.params_arr, size_split=0.7)
    tensor_arrays = data_processor.convert_tensors(list_arrays)
    #train val data tensors from data processing
    tr_data, val_data,tst_data, tr_load, val_load,tst_load = data_processor.get_dataloaders(*tensor_arrays,config.batch_size)

    dim_list = [('l' + str(i + 1)) for i in range(config.latent_dim)]

    ## Dimension of the input curves
    in_d1 = 2
    in_d2 = 72

    # Initialize Model Trainer
    trainer = ModelTrainer(
        autoencoder=ae.VariationalAutoencoder(144, config.latent_dim).to(device),
        mlp=mlp,
        data_processor=data_processor,
        batch_size=config.batch_size,
        ae_epochs=config.ae_epochs,
        mlp_epochs= config.mlp_epochs,
        loss_fn=torch.nn.MSELoss(),
        latent_dim=config.latent_dim,
        num_params=data_processor.num_params,
    )

    # Train Autoencoder
    trainer.train_autoencoder(tr_load, val_load)

    # Extract Latent Vectors & Store in DataFrame
    df_encoded_samples = trainer.extract_latent_vectors(tr_data)

    # Prepare MLP Datasets
    mlp_tr_data, mlp_val_data, mlp_tst_data, mlp_tr_load, mlp_val_load, mlp_tst_load = trainer.prepare_mlp_data(tr_load,
                                                                                                                val_load,                                                                                                   tst_load)
    # Train MLP
    trainer.train_mlp(mlp_tr_load, mlp_val_load)

    # Evaluate MLP on Training and Testing Data
    df_train_samples = trainer.evaluate_model(mlp_tr_data, trainer.percep)
    df_test_samples = trainer.evaluate_model(mlp_tst_data, trainer.percep)

    # Calculate Prediction Errors
    df_l2_err = trainer.calculate_error(df_test_samples)

    # Print Final Results
    print("\nFinal Mean Prediction Errors:")
    print(df_l2_err)

    return trainer.percep,trainer.autoencoder, data_processor

