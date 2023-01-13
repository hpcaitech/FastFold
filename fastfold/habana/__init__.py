ENABLE_HABANA = False
ENABLE_HMP = False

def enable_habana():
    global ENABLE_HABANA
    ENABLE_HABANA = True
    global ENABLE_LAZY_MODE
    ENABLE_LAZY_MODE = True
    import habana_frameworks.torch.core

def is_habana():
    global ENABLE_HABANA
    return ENABLE_HABANA

def enable_hmp():
    global ENABLE_HMP
    ENABLE_HMP = True

def is_hmp():
    global ENABLE_HMP
    return ENABLE_HMP