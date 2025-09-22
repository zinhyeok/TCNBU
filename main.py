from datetime import datetime
from test.logger_setup import setup_module_logger
import numpy as np

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
# 상위 로거 설정
logger = setup_module_logger("project", "project", timestamp)
logger.info("Main Experiment Started")

# from utilize.singlepoint_test import save_singlepoint_exp
from test.mutiplepoint_frequent_test import save_multipoint_frequent_GraphStep_exp


# save_singlepoint_exp(100, timestamp)
save_multipoint_frequent_GraphStep_exp(100, timestamp)

logger.info("All Experiments Completed")
