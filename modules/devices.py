import sounddevice as sd

def list_input_devices():
    devs = sd.query_devices()
    input_devs = []
    for i, d in enumerate(devs):
        if d['max_input_channels'] > 0:
            input_devs.append({'index': i, 'name': d['name'], 'max_input_channels': d['max_input_channels']})
    return input_devs
