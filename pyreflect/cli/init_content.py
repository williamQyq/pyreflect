# Default settings
default_settings = {
    "mod_expt_file": "data/mod_expt.npy",
    "mod_sld_file": "data/mod_sld_fp49.npy",
    "mod_params_file": "data/mod_params_fp49.npy",

    # hyperparameter settings
    "latent_dim": 2,
    "batch_size": 16,
    "ae_epochs": 20,
    "mlp_epochs": 20,

}

INIT_YAML_CONTENT = f"""\
        # 🛠 Configuration file for NR-SLD-Chi Predictor
        # Modify these settings according to your project structure.
        
        ### ⚙️ SLD-Chi settings ###
        sld_predict_chi:
            file:
                model_experimental_sld_profile: {default_settings["mod_expt_file"]} # 📉 Experimental SLD profile data file (for Chi Prediction)
                model_sld_file: {default_settings["mod_sld_file"]} # 📉 SLD Profile file (input for training)
                model_chi_params_file: {default_settings["mod_params_file"]} # 📉 Chi Parameters file (input label for training)
            
            models:
                latent_dim: {default_settings["latent_dim"]}  # Dimension for latent space
                batch_size: {default_settings["batch_size"]}  # Batch size for training
                ae_epochs: {default_settings["ae_epochs"]}  # Autoencoder training epochs
                mlp_epochs: {default_settings["mlp_epochs"]}  # MLP training epochs

        ### ⚙️ NR predict SLD profile settings ###
        nr_predict_sld:
            file:
                experimental_nr_file: curves/experimental_nr_curves.npy # Experimental nr data for sld prediction
                nr_curves_poly: curves/nr_curves_poly.npy   #generated nr curves for training
                sld_curves_poly: curves/sld_curves_poly.npy #generated sld curves for training
                trained_nr_predict_sld_model: trained_nr_sld_model.pth # Trained CNN Model

            num_curves: 1000 #number of generated curves for training 
        """