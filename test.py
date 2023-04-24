from src.datasets import DFDC,FFPP
from accelerate import Accelerator
x = FFPP(FFPP.get_default_config(),50,10,lambda x: x, Accelerator(),split="test",pack=True)
x[44]
# DFDC.get_default_config().merge_from_other_cfg(FFPP.get_default_config())
