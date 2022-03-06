import psutil

def get_HWresource_percent():
    # gives a single float value
    cpu_percent=psutil.cpu_percent()
    cpu_times=psutil.cpu_times()
    # gives an object with many fields
    psutil.virtual_memory()
    # you can convert that object to a dictionary
    dict(psutil.virtual_memory()._asdict())
    # you can have the percentage of used RAM
    memory_percent=psutil.virtual_memory().percent
    return cpu_percent,memory_percent