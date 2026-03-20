"""
Custom exceptions for the project.

All exceptions inherit from IdealFunctionError for easier catching.
"""


class IdealFunctionError(Exception):
    """
    Base exception class for the ideal function analyzer.

    All other custom exceptions inherit from this class, allowing
    callers to catch all project-specific errors with a single except.
    """
    pass


class DataLoadError(IdealFunctionError):
    """
    Raised when data loading fails.

    This occurs when CSV files are missing, empty, or malformed.
    """
    pass


class DataValidationError(IdealFunctionError):
    """
    Raised when data fails validation checks.

    This occurs when required columns are missing or data contains
    invalid values (e.g., non-finite numbers, mismatched x values).
    """
    pass


class DatabaseError(IdealFunctionError):
    """
    Raised when a database operation fails.

    This occurs during insert or update operations when SQLAlchemy
    encounters an error.
    """
    pass


class MappingError(IdealFunctionError):
    """
    Raised when test point mapping fails.

    This occurs when the results DataFrame is missing required columns
    or other mapping preconditions are not met.
    """
    pass
