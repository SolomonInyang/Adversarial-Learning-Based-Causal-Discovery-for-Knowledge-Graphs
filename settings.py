import ast
import os
import warnings
import multiprocessing
import GPUtil

def message_warning(msg, *a, **kwargs):
    """Ignore everything except the message."""
    return str(msg) + '\n'


warnings.formatwarning = message_warning


class ConfigSettings(object):
    """Defining the class for the hardware/high level settings of the CDT.

    Attributes:
        NB_JOBS (int): Number of parallel jobs that can be executed on current
            hardware.
        GPU (int): The number of available GPUs ; defaults to `0`.
        default_device (str): Default device used for pytorch jobs.
        verbose (bool): Sets the verbosity of the toolbox algorithms.
        rpath (str): Path of the `Rscript` executable.
    """

    __slots__ = ("NJOBS",  # Number of parallel jobs runnable
                 "GPU",  # Number of GPUs Available
                 "default_device",  # Default device for gpu (pytorch 0.4)
                 "autoset_config",
                 "verbose",
                 "rpath")

    def __init__(self):
        """Define here the default values of the parameters."""
        super(ConfigSettings, self).__init__()
        self.NJOBS = 8
        self.GPU = 0
        self.autoset_config = True
        self.verbose = False
        self.default_device = 'cpu'
        self.rpath = 'Rscript'

        if self.autoset_config:
            self = autoset_settings(self)

        self.default_device = 'cuda:0' if self.GPU else 'cpu'

    def __setattr__(self, attr, value):
        """Set attribute override for GPU=True."""
        if attr == "GPU" and value and not self.GPU and self.default_device == 'cpu':
            self.default_device = 'cuda:0'
        super(ConfigSettings, self).__setattr__(attr, value)
        if attr == 'rpath' and not object.__getattribute__(self, 'rpath') is None:
            from R import RPackages
            RPackages.reset()

    def get_default(self, *args, **kwargs):
        """Get the default parameters as defined in the Settings instance.

        This function proceeds to seamlessly retrieve the argument to pass
        through, depending on either it was overidden or not: If no argument
        was overridden in a function of the toolbox, the default argument will
        be set to ``None``, and this function will retrieve the default
        parameters as defined by the ``cdt.SETTINGS`` 's attributes.

        It has two modes of processing:

        1. \**kwargs for retrieving a single argument: ``get_default(argument_name=value)``.
        2. \*args through a list of tuples of the shape ``('argument_name', value)`` to retrieve multiple values at once.
        """
        def retrieve_param(i):
            try:
                return self.__getattribute__(i)
            except AttributeError:
                if i == "device":
                    return self.default_device
                else:
                    return self.__getattribute__(i.upper())
        if len(args) == 0:
            if len(kwargs) == 1 and kwargs[list(kwargs.keys())[0]] is not None:
                return kwargs[list(kwargs.keys())[0]]
            elif len(kwargs) == 1:
                return retrieve_param(list(kwargs.keys())[0])
            else:
                raise TypeError("As dict is unordered, it is impossible to give"
                                "the parameters in the correct order.")
        else:
            out = []
            for i in args:
                if i[1] is None:
                    out.append(retrieve_param(i[0]))
                else:
                    out.append(i[1])
            return out


def autoset_settings(set_var):
    """Autoset GPU parameters using CUDA_VISIBLE_DEVICES variables.

    Return default config if variable not set.
    :param set_var: Variable to set. Must be of type ConfigSettings
    """
    try:
        devices = ast.literal_eval(os.environ["CUDA_VISIBLE_DEVICES"])
        if type(devices) != list and type(devices) != tuple:
            devices = [devices]
        if len(devices) != 0:
            set_var.GPU = len(devices)
            set_var.NJOBS = len(devices)
            warnings.warn("Detecting CUDA device(s) : {}".format(devices))

    except KeyError:
        try:
            set_var.GPU = len(GPUtil.getAvailable(order='first', limit=8,
                                                  maxLoad=0.5, maxMemory=0.5,
                                                  includeNan=False))

            if not set_var.GPU:
                warnings.warn("No GPU automatically detected. Setting SETTINGS.GPU to 0, " +
                              "and SETTINGS.NJOBS to cpu_count.")
                set_var.GPU = 0
                set_var.NJOBS = multiprocessing.cpu_count()
            else:
                set_var.NJOBS = set_var.GPU
                warnings.warn("Detecting {} CUDA device(s).".format(set_var.GPU))

        except (ValueError, FileNotFoundError):
            warnings.warn("No GPU automatically detected. Setting SETTINGS.GPU to 0, " +
                          "and SETTINGS.NJOBS to cpu_count.")
            set_var.GPU = 0
            set_var.NJOBS = multiprocessing.cpu_count()

    return set_var


SETTINGS = ConfigSettings()
