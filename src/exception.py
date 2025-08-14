import sys

def get_error_detail(error):
    """
    Returns a formatted string with file name, line number, and error message.
    """
    _, _, detail = sys.exc_info()
    file_name = detail.tb_frame.f_code.co_filename
    error_message = "Error occurred in python script [{0}] line number [{1}] error message [{2}]".format(
        file_name,
        detail.tb_lineno,
        str(error)
    )
    return error_message


class CustomException(Exception):
    def __init__(self, error_message):
        super().__init__(error_message)
        self.error_message = get_error_detail(error_message)

    def __str__(self):
        return self.error_message