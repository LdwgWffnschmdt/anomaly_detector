import os
import logging
from datetime import datetime
import inspect
import timeit

import xlsxwriter
import tensorflow as tf

import feature_extractor
import common.utils as utils

# Only for tests
if __name__ == "__main__":
    BATCH_SIZE = 32
    
    # Create a workbook and add a worksheet for meta data.
    workbook = xlsxwriter.Workbook(os.path.join(os.path.dirname(os.path.realpath(__file__)), datetime.now().strftime("%Y_%m_%d_%H_%M_benchmark.xlsx")))
    heading_format = workbook.add_format({'bold': True, 'font_color': '#0071bc', "font_size": 20})
    subheading_format = workbook.add_format({'bold': True, "font_size": 12})

    #Write metadata
    worksheet_meta = workbook.add_worksheet("Meta")
    worksheet_meta.set_column(0, 0, 25)
    worksheet_meta.set_column(1, 1, 40)

    row = 0
    def add_meta(n, s="", format=None):
        global row
        worksheet_meta.write(row, 0, n, format)
        worksheet_meta.write(row, 1, s, format)
        row += 1

    add_meta("Feature extractor benchmark", format=heading_format)

    add_meta("Start", datetime.now().strftime("%d.%m.%Y, %H:%M:%S"))

    computer_info = utils.getComputerInfo()

    for key, value in computer_info.items():
        add_meta(key, value)

    # # Get CPU info
    # cpu = cpuinfo.get_cpu_info()

    # add_meta("Python version", cpu["python_version"])

    # add_meta("CPU", format=subheading_format)
    # add_meta("Description", cpu["brand"])
    # add_meta("Clock speed (advertised)", cpu["hz_advertised"])
    # add_meta("Clock speed (actual)", cpu["hz_actual"])
    # add_meta("Architecture", cpu["arch"])

    # # Get GPU info
    # add_meta("GPU", format=subheading_format)
    # gpus_tf = tf.config.experimental.list_physical_devices("GPU")
    
    # add_meta("Number of GPUs (tf)", len(gpus_tf))

    # gpus = GPUtil.getGPUs()
    # gpus_available = GPUtil.getAvailability(gpus)
    # for i, gpu in enumerate(gpus):
    #     add_meta("GPU:%i" % gpu.id, gpu.name)
    #     add_meta("GPU:%i (driver)" % gpu.id, gpu.driver)
    #     add_meta("GPU:%i (memory total)" % gpu.id, gpu.memoryTotal)
    #     add_meta("GPU:%i (memory free)" % gpu.id, gpu.memoryFree)
    #     add_meta("GPU:%i (available?)" % gpu.id, gpus_available[i])

    # # Get RAM info
    # add_meta("RAM", format=subheading_format)
    # mem = virtual_memory()

    # add_meta("RAM (total)", mem.total)
    # add_meta("RAM (available)", mem.available)

    # Get all the available feature extractor names
    extractor_names = inspect.getmembers(feature_extractor, inspect.isclass)

    # Get an instance of each class
    module = __import__("feature_extractor")
    
    try:
        for extractor_name in extractor_names:
            worksheet = workbook.add_worksheet(extractor_name[0].replace("FeatureExtractor", ""))
            worksheet.set_column(0, 20, 20)

            col = 0
            def add_to_excel(n, times):
                global col
                worksheet.write(0, col, n, subheading_format)
                for i, t in enumerate(times):
                    worksheet.write_number(i + 1, col, t)
                col += 1

            def log(s, t):
                """Log duration t with info string s"""
                logging.info("%-40s (%s): %s" % (extractor_name[0], s, str(t)))
                add_to_excel(s, t)

            _class = getattr(module, extractor_name[0])

            # Test extractor initialization
            log("Initialization", timeit.repeat(lambda: _class(), number=1, repeat=5))

            # Load a test dataset
            dataset = utils.load_dataset("/home/ludwig/ros/src/ROS-kate_bag/bags/TFRecord/autonomous_realsense.tfrecord")

            # Test batch extraction
            extractor = _class()
            batch = list(dataset.take(BATCH_SIZE).batch(BATCH_SIZE).as_numpy_iterator())[0]
            log("Extract batch", timeit.repeat(lambda: extractor.extract_batch(batch[0]), number=1, repeat=10))

            # Test single image extraction
            extractor = _class()
            single = list(dataset.take(1).as_numpy_iterator())[0] # Get a single entry
            log("Extract single", timeit.repeat(lambda: extractor.extract(single[0]), number=1, repeat=10))
    finally:
        workbook.close()