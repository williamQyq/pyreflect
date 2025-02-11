from pyreflect.data_processor import DataProcessor

def process_data(expt_data_path, sld_data_path, chi_params_data_path):
    """
    Preprocess SLD profile data and chi_params data
    :param expt_data_path:
    :param sld_data_path:
    :param chi_params_data_path:
    :return:
    """
    data_processor = DataProcessor(expt_data_path,sld_data_path,chi_params_data_path)
    data_processor.load_data()
    sld_arr, params_arr = data_processor.preprocess_data()