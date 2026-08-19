"""Security boundary exceptions without credential details."""


class SecurityConfigurationError(RuntimeError):
    pass


class AuthenticationError(RuntimeError):
    pass


class AuthorizationError(PermissionError):
    def __init__(self, task_type: str):
        super().__init__(f"Permission denied for task type: {task_type}")
        self.task_type = task_type
        self.error_code = "PERMISSION_DENIED"


__all__ = ["AuthenticationError", "AuthorizationError", "SecurityConfigurationError"]
