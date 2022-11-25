import pickle

import torch
import habana_frameworks.torch.core as htcore

from fastfold.config import model_config
from fastfold.model.hub import AlphaFold

from fastfold.utils.inject_habana import inject_habana

def main():
    batch = pickle.load(open('./test_batch.pkl', 'rb'))

    model_name = "model_1"
    device = torch.device("hpu")

    config = model_config(model_name)
    config.globals.inplace = False
    model = AlphaFold(config)
    model = inject_habana(model)
    model = model.eval()
    model = model.to(device=device)

    with torch.no_grad():
        batch = {k: torch.as_tensor(v).to(device=device) for k, v in batch.items()}
        out = model(batch)

        htcore.mark_step()
        print(out)


if __name__ == '__main__':
    main()

