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
        
        
class FbetaDeepSupervisionCallback(Callback):
    """
    F1 metric callback.
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

        output = state.output[self.output_key]
        output = [F.sigmoid(o) for o in output]
        output = sum(output) / len(output)
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
        

class LossDeepSupervisionCallback(Callback):
    def __init__(self, input_key: str = "targets", output_key: str = "logits"):
        self.input_key = input_key
        self.output_key = output_key

    def on_stage_start(self, state):
        assert state.criterion is not None

    def on_batch_end(self, state):
        weights = [
            0.001,
            0.005,
            0.01,
            0.02,
            0.02,
            0.1,
            1.0
        ]

        outputs = state.output[self.output_key]
        loss = 0
        for i, output in enumerate(outputs):
            loss += weights[i] * state.criterion(
                output, state.input[self.input_key]
            )

        state.loss = loss


class LossTwoHeadCallback(Callback):
    def __init__(self,
                 sigmoid_input_key: str = "sigmoid_label",
                 sigmoid_output_key: str = "sigmoid_logits",
                 softmax_input_key: str = "softmax_label",
                 softmax_output_key: str = "softmax_logits",
                 ):
        super(LossTwoHeadCallback, self).__init__()
        self.sigmoid_input_key = sigmoid_input_key
        self.sigmoid_output_key = sigmoid_output_key
        self.softmax_input_key = softmax_input_key
        self.softmax_output_key = softmax_output_key

    def on_stage_start(self, state):
        assert state.criterion is not None

    def on_batch_end(self, state):

        sigmoid_output = state.output[self.sigmoid_output_key]
        sigmoid_input = state.input[self.sigmoid_input_key]
        softmax_output = state.output[self.softmax_output_key]
        softmax_input = state.input[self.softmax_input_key]


        state.loss = state.criterion(
            sigmoid_output, softmax_output, sigmoid_input, softmax_input
        )


class FbetaTwoHeadsCallback(Callback):
    """
    Fbeta metric callback.
    """

    def __init__(self,
                 th: float = 0.3,
                 beta: int = 2,
                 sigmoid_input_key: str = "sigmoid_label",
                 sigmoid_output_key: str = "sigmoid_logits",
                 softmax_input_key: str = "softmax_label",
                 softmax_output_key: str = "softmax_logits",
                 ):
        """
        :param input_key: input key to use for precision calculation;
            specifies our `y_true`.
        :param output_key: output key to use for precision calculation;
            specifies our `y_pred`.
        """
        super().__init__()
        self.sigmoid_input_key = sigmoid_input_key
        self.sigmoid_output_key = sigmoid_output_key
        self.softmax_input_key = softmax_input_key
        self.softmax_output_key = softmax_output_key
        self.th = th
        self.beta = beta

    def on_loader_start(self, state):
        self.sigmoid_outputs = []
        self.sigmoid_labels = []

        self.softmax_outputs = []
        self.softmax_labels = []

    def on_batch_end(self, state):
        sigmoid_output = F.sigmoid(state.output[self.sigmoid_output_key])
        sigmoid_output = sigmoid_output.detach().cpu().numpy()
        sigmoid_label = state.input[self.sigmoid_input_key].detach().cpu().numpy()
        self.sigmoid_outputs.append(sigmoid_output)
        self.sigmoid_labels.append(sigmoid_label)

        softmax_output = F.softmax(state.output[self.softmax_output_key])
        softmax_output = softmax_output.detach().cpu().numpy()
        softmax_label = state.input[self.softmax_input_key].detach().cpu().numpy()
        self.softmax_outputs.append(softmax_output)
        self.softmax_labels.append(softmax_label)

    def on_loader_end(self, state):
        import warnings
        warnings.filterwarnings("ignore")

        self.sigmoid_outputs = np.concatenate(self.sigmoid_outputs, axis=0)
        self.sigmoid_labels = np.concatenate(self.sigmoid_labels, axis=0)

        sigmoid_fscore = fbeta_score(
            y_pred=self.sigmoid_outputs > self.th,
            y_true=self.sigmoid_labels,
            beta=self.beta,
            average="samples"
        )
        state.metrics.epoch_values[state.loader_name]['sigmoid_fbeta'] = sigmoid_fscore

        self.softmax_outputs = np.concatenate(self.softmax_outputs, axis=0)
        self.softmax_labels = np.concatenate(self.softmax_labels, axis=0)

        softmax_fscore = fbeta_score(
            y_pred=self.softmax_outputs > self.th,
            y_true=self.softmax_labels,
            beta=self.beta,
            average="samples"
        )
        state.metrics.epoch_values[state.loader_name]['softmax_fbeta'] = softmax_fscore

        state.metrics.epoch_values[state.loader_name]['avg_fbeta'] = (softmax_fscore + sigmoid_fscore) / 2
