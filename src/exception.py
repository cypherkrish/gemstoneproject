import sys
from src.logger import logging

def error_message_detail(error, error_details: sys):
    _, _, exe_tb = error_details.exc_info()
    file_name = exe_tb.tb_frame.f_code.co_filename

    error_message = "Error occurred in python script name [{0}] line number [{1}] error message [{2}]".format(file_name,exe_tb.tb_lineno, str(error))

    return error_message

class CustomException(Exception):
    def __init__(self, error_message, error_deail: sys):
        super().__init__(error_message)
        self.error_message = error_message_detail(error_message, error_detail = error_deail)

    def __str__(self) -> str:
        return self.error_message


# Commenting the main code

'''

if __name__ == '__main__':
    logging.info("Logging has started")

    try:
        a = 1/0
    except Exception as e:
        logging.info("Error has occured")
        raise CustomException(e, sys)
    
'''