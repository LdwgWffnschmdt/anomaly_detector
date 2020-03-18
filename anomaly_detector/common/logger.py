if "LOGGER_LOADED" not in globals():
    LOGGER_LOADED = True

    import logging
    import sys

    import tqdm
    import colorlog

    #   Level | Level for Humans | Level Description                  
    #  -------|------------------|------------------------------------ 
    #   0     | DEBUG            | [Default] Print all messages       
    #   1     | INFO             | Filter out INFO messages           
    #   2     | WARNING          | Filter out INFO & WARNING messages 
    #   3     | ERROR            | Filter out all messages  
    import os
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # or any {'0', '1', '2'}

    import tensorflow as tf

    # Configure logging
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

    logger = tf.get_logger()
    logger.removeHandler(logger.handlers[0])
    logger.setLevel(logging.INFO)

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

    logger.addHandler(handler)

    # Redirect

    class DummyFile(object):
        file = None
        def __init__(self, file):
            self.file = file

        def write(self, x):
            # Avoid print() second call (useless \n)
            if len(x.rstrip()) > 0:
                tqdm.tqdm.write(x, file=self.file)

        def __eq__(self, other):
            return other is self.file

    sys.stdout = DummyFile(sys.stdout)

    debug = logger.debug
    info = logger.info
    warning = logger.warning
    error = logger.error
    exception = logger.exception
    critical = logger.critical
    fatal = critical
    log = logger.log

    if __name__ == "__main__":
        import time

        for i in tqdm.tqdm(range(100), desc="Progress", file=sys.stderr):
            if i == 5:
                logger.info("HALLO")
            if i == 10:
                logger.warning("HALLO")
            if i == 15:
                logger.error("HALLO")
            if i == 20:
                logger.debug("HALLO")
            if i == 25:
                print("PUP")
            if i == 30:
                tqdm.tqdm.write("HALLALA")
            if i == 45:
                from feature_extractor import FeatureExtractorC3D
                FeatureExtractorC3D()
            if i == 40:
                for x in tqdm.tqdm(range(30), desc="Subprogress", file=sys.stderr):
                    time.sleep(0.05)
            time.sleep(0.1)