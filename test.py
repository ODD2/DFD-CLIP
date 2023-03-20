import evaluate
import torch

mse_calc = evaluate.load("mse","multilist")

print(mse_calc.compute(references=[[1,1],[2,2],[3,3]],predictions=[[0,0],[0,0],[0,0]]))
# mse_calc.add_batch(references=torch.randn((1,10)).tolist(),predictions=torch.randn((1,10)).tolist())
# mse_calc.compute(references=torch.randn((1,180)),predictions=torch.randn((1,180)))