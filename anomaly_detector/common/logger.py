#   Level | Level for Humans | Level Description                  
#  -------|------------------|------------------------------------ 
#   0     | DEBUG            | [Default] Print all messages       
#   1     | INFO             | Filter out INFO messages           
#   2     | WARNING          | Filter out INFO & WARNING messages 
#   3     | ERROR            | Filter out all messages  
import os
import logging
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # or any {'0', '1', '2'}

import tensorflow as tf
tf_logger = tf.get_logger()
tf_logger.setLevel(logging.ERROR)
for i in range(len(tf_logger.handlers)):
    tf_logger.removeHandler(tf_logger.handlers[i])

import sys

import tqdm
import colorlog

class TqdmLoggingHandler(logging.Handler):
    def __init__(self, level=logging.NOTSET):
        logging.Handler.__init__(self, level)

    def emit(self, record):
        try:
            msg = self.format(record)
            tqdm.tqdm.write(msg, file=sys.stdout)
            self.flush()
        except (KeyboardInterrupt, SystemExit):
            raise
        except:
            self.handleError(record)

class TqdmFile(object):
    file = None
    def __init__(self, file):
        self.file = file

    def write(self, x):
        # Avoid print() second call (useless \n)
        if len(x.rstrip()) > 0:
            tqdm.tqdm.write(x, file=self.file)

    def __eq__(self, other):
        return other is self.file

class Logger(object):
    def __init__(self):
        self.logger = logging.getLogger("root")
        self.logger.propagate = False
        
        if not self.logger.handlers:
            self.logger.setLevel(logging.INFO)
            handler = TqdmLoggingHandler()
            
            formatter = colorlog.ColoredFormatter(
                '%(log_color)s%(asctime)s %(levelname).1s: %(message)s',
                datefmt='%Y-%d-%d %H:%M:%S',
                log_colors={
                    'DEBUG': 'thin_cyan',
                    'INFO': 'thin_white',
                    'SUCCESS:': 'green',
                    'WARNING': 'yellow',
                    'ERROR': 'red',
                    'CRITICAL': 'red,bg_white'},)

            handler.setFormatter(formatter)
            self.logger.addHandler(handler)

    def debug(self, msg):
        self.logger.debug(msg)

    def info(self, msg):
        self.logger.info(msg)

    def warning(self, msg):
        self.logger.warning(msg)

    def error(self, msg):
        self.logger.error(msg)

# Redirect
# sys.stdout = TqdmFile(sys.stdout)

logger = Logger()

debug = logger.debug
info = logger.info
warning = logger.warning
error = logger.error

if __name__ == "__main__":
    import time
    import consts

    for i in tqdm.tqdm(range(100), desc="Progress", file=sys.stderr):
        if i == 5:
            logger.info("HALLO")
        if i == 10:
            logger.warning("HALLO")
        if i == 15:
            __import__("tensorflow")
            logger.error("HALLO")
        if i == 20:
            __import__("tensorflow_hub")
            logger.debug("HALLO")
        if i == 25:
            print("PUP")
        if i == 30:
            tqdm.tqdm.write("HALLALA")
        if i == 40:
            for x in tqdm.tqdm(range(30), desc="Subprogress", file=sys.stderr):
                time.sleep(0.02)
        if i == 45:
            from feature_extractor import FeatureExtractorEfficientNetB0
            extractor = FeatureExtractorEfficientNetB0()
        if i == 50:
            extractor.extract_files(consts.EXTRACT_FILES_TEST, output_file="/tmp/test.h5")
        time.sleep(0.1)