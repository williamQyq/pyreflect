from pyreflect.data_processor import DataProcessor
from pyreflect.models.chi_pred_model_trainer import ModelTrainer
from pyreflect.models import mlp, autoencoder as ae
import torch

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

def run_model_training(
        mod_expt_file, mod_sld_file, mod_params_file, batch_size
):
    print(f"Processing data with:")
    print(f"  - mod_expt_file: {mod_expt_file}")
    print(f"  - mod_sld_file: {mod_sld_file}")
    print(f"  - mod_params_file: {mod_params_file}")
    print(f"  - batch_size: {batch_size}")

    data_processor = DataProcessor(mod_expt_file,mod_sld_file,mod_params_file)
    data_processor.load_data()
    data_processor.preprocess_data()

    list_arrays = data_processor.split_arrays(size_split=0.7)

    #train val data tensors from data processing
    tr_data, val_data,tst_data, tr_load, val_load,tst_load = data_processor.get_dataloaders(*list_arrays,batch_size)

    ## Dimension of the latent space
    latent_dim = 2
    dim_list = [('l' + str(i + 1)) for i in range(latent_dim)]
    ## Parameter size
    num_params = data_processor.params_arr.shape[1]
    data_processor.num_params = num_params
    ## Dimension of the input curves
    in_d1 = 2
    in_d2 = 72
    ## Number of epochs
    ae_epochs = 200
    mlp_epochs = 200
    ## Batch size
    batch_size = 16

    # Initialize Model Trainer
    trainer = ModelTrainer(
        autoencoder=ae.Autoencoder(144, latent_dim).to(device),
        mlp=mlp,
        data_processor=data_processor,
        batch_size=batch_size,
        ae_epochs=200,
        mlp_epochs=20,
        loss_fn=torch.nn.MSELoss(),
        latent_dim=latent_dim,
        num_params=num_params
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

