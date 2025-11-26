"""
Custom exceptions for GARMF-Mpox
"""


class GARMFException(Exception):
    """Base exception for GARMF-Mpox"""
    pass


class DataContractViolation(GARMFException):
    """Raised when data contract validation fails"""
    pass


class ConfigurationError(GARMFException):
    """Raised when configuration is invalid"""
    pass


class RunExecutionError(GARMFException):
    """Raised when run execution fails"""
    pass


class ArtefactNotFound(GARMFException):
    """Raised when artefact is not found"""
    pass


class DatasetNotFound(GARMFException):
    """Raised when dataset is not found"""
    pass


class ModelNotFound(GARMFException):
    """Raised when model is not found"""
    pass
