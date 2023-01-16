import pickle
import time

import habana_frameworks.torch.core as htcore
import torch

import fastfold.habana as habana
from fastfold.config import model_config
from fastfold.habana.distributed import init_dist
from fastfold.habana.fastnn.ops import set_chunk_size
from fastfold.habana.inject_habana import inject_habana
from fastfold.model.hub import AlphaFold

def main():
    habana.enable_habana()

    init_dist()
    batch = pickle.load(open('./test_batch.pkl', 'rb'))

    model_name = "model_1"
    device = torch.device("hpu")

    config = model_config(model_name)
    config.globals.inplace = False
    config.globals.chunk_size = 512
    # habana.enable_hmp()
    model = AlphaFold(config)
    model = inject_habana(model)
    model = model.eval()
    model = model.to(device=device)

    if config.globals.chunk_size is not None:
        set_chunk_size(model.globals.chunk_size + 1)

    if habana.is_hmp():
        from habana_frameworks.torch.hpex import hmp
        hmp.convert(opt_level='O1',
                    bf16_file_path='./habana/ops_bf16.txt',
                    fp32_file_path='./habana/ops_fp32.txt',
                    isVerbose=False)
        print("========= AMP ENABLED!!")

    with torch.no_grad():
        batch = {k: torch.as_tensor(v).to(device=device) for k, v in batch.items()}

        for _ in range(5):
            t = time.perf_counter()
            out = model(batch)
            htcore.mark_step()
            htcore.hpu.default_stream().synchronize()
            print(f"Inference time: {time.perf_counter() - t}")


if __name__ == '__main__':
    main()
