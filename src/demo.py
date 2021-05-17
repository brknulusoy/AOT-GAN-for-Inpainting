import importlib
import numpy as np
import matplotlib.pyplot as plt

import torch
from torchvision.transforms import ToTensor

from utils.option import args
from metric import metric as module_metric
from data.dataset import TerrainDataset


def demo(args):
    dataset = TerrainDataset(
        args.dir_train,
        dataset_type="train",
        randomize=True,
        random_state=343,
        idx_offset=args.idx_offset,
        block_variance=1,
    )

    # Model and version
    net = importlib.import_module("model." + args.model)
    model = net.InpaintGenerator(args)
    model.load_state_dict(torch.load(args.pre_train, map_location="cpu"))
    model.eval()

    for target, mask, fn, _ in dataset:
        print(f"[**] inpainting {fn} ... ")
        target = target.unsqueeze(0)
        mask = mask.unsqueeze(0)

        with torch.no_grad():
            masked_tensor = (target * (1 - mask).float()) + mask
            pred_tensor = model(masked_tensor, mask)
            comp_tensor = pred_tensor * mask + target * (1 - mask)

            # metrics prepare for image assesments
            metrics = {
                met: getattr(module_metric, met) for met in ["mae", "psnr", "ssim"]
            }
            evaluation_scores = {key: 0 for key, val in metrics.items()}
            for key, val in metrics.items():
                evaluation_scores[key] = val(
                    target.numpy().reshape((256, 256)),
                    comp_tensor.numpy().reshape((256, 256)),
                    num_worker=1,
                )

            fig, axs = plt.subplots(2, 2, figsize=(12, 12), sharex=True, sharey=True)

            target = target.reshape((256, 256))
            mask = mask.reshape((256, 256))
            masked = np.copy(target.numpy())
            masked[mask == 1] = np.nan
            comp_tensor = comp_tensor.reshape((256, 256))
            
            target = target.numpy()
            levels = np.arange(target.min(), target.max(), (target.max() - target.min()) / 100)
            axs[0, 0].contourf(target, levels=levels, cmap="terrain")
            axs[0, 1].contourf(mask.numpy(), cmap="gray")
            axs[1, 0].contourf(masked, levels=levels, cmap="terrain")
            axs[1, 1].contourf(comp_tensor.numpy(), levels=levels, cmap="terrain")

            fig.tight_layout(rect=[0, 0.03, 1, 0.95])
            fig.suptitle(
                " ".join(
                    [
                        "{}: {:6f}".format(key, val)
                        for key, val in evaluation_scores.items()
                    ],
                ),
                fontsize=22,
            )
            plt.show()


if __name__ == "__main__":
    demo(args)
