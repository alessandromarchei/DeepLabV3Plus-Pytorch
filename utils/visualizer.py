import wandb
import numpy as np


class WandbVisualizer:
    """
    WandB-based visualizer, API-compatible with the old Visdom Visualizer.

    Supports:
      - scalars
      - images (optionally with masks)
      - tables (dicts)
    """

    def __init__(
        self,
        project: str,
        exp_name: str,
        config: dict = None,
        id: str = None,
        dir: str = None,
        mode: str = "online",  # "online", "offline", "disabled"
    ):
        """
        Args:
            project (str): wandb project name
            exp_name (str): run name
            config (dict): experiment config (args)
            id (str): optional prefix for all logged keys
            dir (str): directory for wandb files
            mode (str): wandb mode
        """
        self.id = id
        self.run = wandb.init(
            project=project,
            name=exp_name,
            config=config,
            dir=dir,
            mode=mode,
        )

    def _prefix(self, name: str) -> str:
        if self.id is not None:
            return f"[{self.id}] {name}"
        return name

    # ------------------------------------------------------------------
    # Scalars
    # ------------------------------------------------------------------
    def vis_scalar(self, name, x, y, opts=None):
        """
        Equivalent of visdom line plot (append).
        In wandb: scalar vs step.
        """
        name = self._prefix(name)

        if not isinstance(x, (list, tuple)):
            x = [x]
        if not isinstance(y, (list, tuple)):
            y = [y]

        for xi, yi in zip(x, y):
            wandb.log({name: yi}, step=xi)

    # ------------------------------------------------------------------
    # Images
    # ------------------------------------------------------------------
    def vis_image(
        self,
        name,
        img,
        step=None,
        masks=None,
        caption=None,
    ):
        """
        Log an image.
        Optionally supports segmentation masks.

        Args:
            img: numpy array (H,W,C) or (C,H,W)
            masks: dict like:
                {
                  "prediction": {"mask_data": pred, "class_labels": labels},
                  "ground_truth": {"mask_data": gt, "class_labels": labels}
                }
        """
        name = self._prefix(name)

        if img.ndim == 3 and img.shape[0] in (1, 3):
            img = np.transpose(img, (1, 2, 0))

        wandb_img = wandb.Image(
            img,
            masks=masks,
            caption=caption,
        )

        if step is None:
            wandb.log({name: wandb_img})
        else:
            wandb.log({name: wandb_img}, step=step)

    # ------------------------------------------------------------------
    # Tables
    # ------------------------------------------------------------------
    def vis_table(self, name, tbl: dict, step=None):
        """
        Log a dict as a table.
        """
        name = self._prefix(name)

        table = wandb.Table(columns=["key", "value"])
        for k, v in tbl.items():
            table.add_data(k, str(v))

        if step is None:
            wandb.log({name: table})
        else:
            wandb.log({name: table}, step=step)

    # ------------------------------------------------------------------
    # Finish
    # ------------------------------------------------------------------
    def finish(self):
        wandb.finish()
