import logging, sys

sys_logger = logging.getLogger("banana_flow_sys")
sys_logger.setLevel(logging.INFO)
sys_logger.propagate = False

if not sys_logger.handlers:
    console_handler = logging.StreamHandler(sys.stdout)
    sys_logger.addHandler(console_handler)