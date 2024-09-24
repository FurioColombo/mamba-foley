from modules.utils.utilities import load_json_config

class Config:

    @staticmethod
    def get_config(config_file_path):
        config = load_json_config(config_file_path)
        return config
