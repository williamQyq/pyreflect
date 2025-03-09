from pathlib import Path

def run_sld_prediction(root:Path,config):
    ...
    # # Load NR and SLD data
    # nr_file = settings["nr_sld_curves_poly"]
    # sld_file = settings["sld_curves_poly"]
    # model_path = settings["nr_sld_model_file"]
    #
    # # Load experimental NR curves for inference
    # expt_nr_file = settings["expt_nr_file"]
    #
    # model = None
    #
    # # Check if a trained SLD model exists, else train one
    # if Path(model_path).exists():
    #     model = workflow.load_nr_sld_model(model_path)
    #     typer.echo("Loaded existing trained SLD model.")
    # else:
    #     typer.echo("No trained SLD model found. Training a new model...")
    #     generated_curves_folder = Path(settings["curves_folder"])
    #     nr_file, sld_file = workflow.generate_nr_sld_curves(10, curves_dir=generated_curves_folder)
    #
    #     # Save generated NR and SLD files to settings.yml
    #     settings["nr_sld_curves_poly"] = str(nr_file)
    #     settings["sld_curves_poly"] = str(sld_file)
    #
    #     # update yaml
    #     with open(config, "w") as f:
    #         yaml.safe_dump(settings, f)
    #
    #     model = workflow.train_nr_predict_sld_model(nr_file, sld_file, to_be_saved_model_path=model_path)
    #
    # if not model:
    #     typer.echo("Model not loaded.")
    #     raise typer.Exit()
    #
    # # Testing prediction***
    # print(f"Settings loaded: {settings}")
    # print(f"Using NR file: {nr_file}")
    # print(f"Using SLD file: {sld_file}")
    # print(f"Using Experimental NR file: {expt_nr_file}")
    # print(f"Using Model Path: {model_path}")
    #
    # expt_nr_file = settings["nr_sld_curves_poly"]
    # predicted_sld = workflow.predict_sld_from_nr(model, expt_nr_file)
    # print(predicted_sld)
    # typer.echo("SLD Prediction complete!")