import torch
import numpy as np
import torch.nn.functional as F
from catalyst.dl.callbacks import Callback
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score, fbeta_score
from sklearn.metrics.classification import precision_recall_fscore_support


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