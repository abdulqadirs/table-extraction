import logging
from copy import deepcopy

#logger = logging.getLogger('segmentation')

class Config:
    """
    Stores the content of config.ini using deepcopy(without changing the source)
    """
    __config = {}

    @staticmethod
    def get(name):
        """
        Returns the value of given name
        
        Args:
            name (str): key of specific value in dictionay(which stores config settings).
        
        Returns:
            The deep copy of the value from dictionary using the given key(name).
        
        Raises:
            KeyError: If the confi setting doesn't exist.
        """
        try:
            return deepcopy(Config.__config[name])
        except KeyError:
            #logger.warning("Config setting " + name + "not found.")
            return
    
    @staticmethod
    def set(name, value):
        """
        Sets the value of specific config setting
        Args:
            name (string): Name of config setting (key in dictionary).
            value (str/int/bool/float/Path): 
        """
        try:
            Config.__config[name] = deepcopy(value)
        except:
            Config.__config[name] = value