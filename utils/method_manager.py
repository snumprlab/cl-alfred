from methods.er_baseline import ER
from methods.ewc import EWCpp
from methods.mir import MIR
from methods.clib import CLIB
from methods.der import DER
from methods.xder import XDER
from methods.cama_nodc import CAMA_NODC
from methods.cama import CAMA


def select_method(args, n_classes, model):
    kwargs = vars(args)

    methods = {
        'er': ER,
        'ewc++': EWCpp,
        'mir': MIR,
        'clib': CLIB,
        'der': DER,
        'xder': XDER,
        'cama_nodc': CAMA_NODC,
        'cama': CAMA,
    }

    if args.mode in methods:
        method = methods[args.mode](n_classes, model, **kwargs)
    else:
        raise NotImplementedError(f"Choose the args.mode in {list(methods.keys())}")

    return method
