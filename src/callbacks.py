from catalyst.dl.utils import UtilsFactory
import numpy as np
import torch.nn.functional as F
from catalyst.dl.callbacks import Callback
from sklearn.metrics import f1_score, precision_score, recall_score, fbeta_score


class MixupLossCallback(Callback):
    def __init__(
        self,
        input_key: str = "targets",
        output_key: str = "logits",
        prefix: str = "loss",
        criterion_key: str = None,
        loss_key: str = None,
        multiplier: float = 1.0
    ):
        self.input_key = input_key
        self.output_key = output_key
        self.prefix = prefix
        self.criterion_key = criterion_key
        self.loss_key = loss_key
        self.multiplier = multiplier

    def _add_loss_to_state(self, state, loss):
        if self.loss_key is None:
            if state.loss is not None:
                if isinstance(state.loss, list):
                    state.loss.append(loss)
                else:
                    state.loss = [state.loss, loss]
            else:
                state.loss = loss
        else:
            if state.loss is not None:
                assert isinstance(state.loss, dict)
                state.loss[self.loss_key] = loss
            else:
                state.loss = {self.loss_key: loss}

    def _compute_loss(self, state, criterion):

        if state.loader_name == 'train':
            lam = state.input['lam']
            targets_a = state.input['targets_a']
            targets_b = state.input['targets_b']

            loss = lam * criterion(
                state.output[self.output_key],
                targets_a
            ) + (1 - lam) * criterion(
                state.output[self.output_key],
                targets_b
            )
        else:
            loss = criterion(
                state.output[self.output_key],
                state.input[self.input_key]
            )
        return loss

    def on_stage_start(self, state):
        assert state.criterion is not None

    def on_batch_end(self, state):
        criterion = state.get_key(
            key="criterion", inner_key=self.criterion_key
        )

        loss = self._compute_loss(state, criterion) * self.multiplier

        state.metrics.add_batch_value(metrics_dict={
            self.prefix: loss.item(),
        })

        self._add_loss_to_state(state, loss)


class IterCheckpointCallback(Callback):
    """
    Checkpoint callback to save/restore your model/criterion/optimizer/metrics.
    """

    def __init__(
        self, n_iters : int = 2500, save_n_best: int = 3, resume: str = None, resume_dir: str = None
    ):
        """
        :param save_n_best: number of best checkpoint to keep
        :param resume: path to checkpoint to load and initialize runner state
        """
        self.save_n_best = save_n_best
        self.resume = resume
        self.resume_dir = resume_dir
        self.top_best_metrics = []
        self.n_iters = n_iters
        self.count = 0

        self._keys_from_state = ["resume", "resume_dir"]

    def save_checkpoint(
        self,
        logdir,
        checkpoint,
        is_best,
    ):
        suffix = f"{checkpoint['stage']}.iter.{self.count}"
        filepath = UtilsFactory.save_checkpoint(
            logdir=f"{logdir}/checkpoints/",
            checkpoint=checkpoint,
            suffix=suffix,
            is_best=is_best,
            is_last=True
        )
        print(f"\nSaved checkpoint at {filepath}")

    def pack_checkpoint(self, **kwargs):
        return UtilsFactory.pack_checkpoint(**kwargs)

    def on_batch_end(self, state):
        self.count += 1
        if self.count % self.n_iters == 0:
            checkpoint = self.pack_checkpoint(
                model=state.model,
                criterion=state.criterion,
                optimizer=state.optimizer,
                scheduler=state.scheduler,
                epoch_metrics=None,
                valid_metrics=None,
                stage=state.stage,
                epoch=state.epoch
            )
            self.save_checkpoint(
                logdir=state.logdir,
                checkpoint=checkpoint,
                is_best=True,
            )

    def on_stage_start(self, state):
        self.count = 0
        for key in self._keys_from_state:
            value = getattr(state, key, None)
            if value is not None:
                setattr(self, key, value)

        if self.resume_dir is not None:
            self.resume = str(self.resume_dir) + "/" + str(self.resume)


class F1Callback(Callback):
    """
    F1 metric callback.
    """

    def __init__(self,
                 input_key: str = "targets",
                 output_key: str = "logits"):
        """
        :param input_key: input key to use for precision calculation;
            specifies our `y_true`.
        :param output_key: output key to use for precision calculation;
            specifies our `y_pred`.
        """
        super().__init__()
        self.input_key = input_key
        self.output_key = output_key

    def on_loader_start(self, state):
        self.outputs = []
        self.labels = []

    def on_batch_end(self, state):
        output = state.output[self.output_key].detach().cpu().numpy()
        label = state.input[self.input_key].detach().cpu().numpy()

        output = np.argmax(output, axis=1)

        self.outputs.append(output)
        self.labels.append(label)

    def on_loader_end(self, state):
        import warnings
        warnings.filterwarnings("ignore")

        self.outputs = np.concatenate(self.outputs, axis=0)
        self.labels = np.concatenate(self.labels, axis=0)

        f1 = f1_score(y_true=self.labels, y_pred=self.outputs, average='macro')
        precision = precision_score(y_true=self.labels, y_pred=self.outputs, average='macro')
        recall = recall_score(y_true=self.labels, y_pred=self.outputs, average='macro')

        state.metrics.epoch_values[state.loader_name]['p'] = precision
        state.metrics.epoch_values[state.loader_name]['r'] = recall
        state.metrics.epoch_values[state.loader_name]['f1'] = f1


class FbetaCallback(Callback):
    """
    Fbeta metric callback.
    """

    def __init__(self,
                 th: float = 0.3,
                 beta: int = 2,
                 input_key: str = "targets",
                 output_key: str = "logits"):
        """
        :param input_key: input key to use for precision calculation;
            specifies our `y_true`.
        :param output_key: output key to use for precision calculation;
            specifies our `y_pred`.
        """
        super().__init__()
        self.input_key = input_key
        self.output_key = output_key
        self.th = th
        self.beta = beta

    def on_loader_start(self, state):
        self.outputs = []
        self.labels = []

    def on_batch_end(self, state):
        output = F.sigmoid(state.output[self.output_key])
        output = output.detach().cpu().numpy()
        label = state.input[self.input_key].detach().cpu().numpy()

        self.outputs.append(output)
        self.labels.append(label)

    def on_loader_end(self, state):
        import warnings
        warnings.filterwarnings("ignore")

        self.outputs = np.concatenate(self.outputs, axis=0)
        self.labels = np.concatenate(self.labels, axis=0)

        fscore = fbeta_score(y_pred=self.outputs > self.th, y_true=self.labels, beta=self.beta, average="samples")
        state.metrics.epoch_values[state.loader_name]['fbeta'] = fscore
