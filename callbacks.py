import torch
import pandas as pd
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
                 culture_input_key: str = "culture_labels",
                 culture_output_key: str = "culture_logits",
                 tag_input_key: str = "tag_labels",
                 tag_output_key: str = "tag_logits",
                 ):
        super(LossTwoHeadCallback, self).__init__()
        self.culture_input_key = culture_input_key
        self.culture_output_key = culture_output_key
        self.tag_input_key = tag_input_key
        self.tag_output_key = tag_output_key

    def on_stage_start(self, state):
        assert state.criterion is not None

    def on_batch_end(self, state):

        culture_logits = state.output[self.culture_output_key]
        culture_labels = state.input[self.culture_input_key]
        tag_logits = state.output[self.tag_output_key]
        tag_labels = state.input[self.tag_input_key]

        state.loss = state.criterion(
            culture_logits, tag_logits, culture_labels, tag_labels
        )


class FbetaTwoHeadsCallback(Callback):
    """
    Fbeta metric callback.
    """

    def __init__(self,
                 label_file: str="./data/labels.csv",
                 th: float = 0.3,
                 beta: int = 2,
                 culture_input_key: str = "culture_labels",
                 culture_output_key: str = "culture_logits",
                 tag_input_key: str = "tag_labels",
                 tag_output_key: str = "tag_logits",
                 ):
        """
        :param input_key: input key to use for precision calculation;
            specifies our `y_true`.
        :param output_key: output key to use for precision calculation;
            specifies our `y_pred`.
        """
        super().__init__()
        self.culture_input_key = culture_input_key
        self.culture_output_key = culture_output_key
        self.tag_input_key = tag_input_key
        self.tag_output_key = tag_output_key
        self.th = th
        self.beta = beta
        label_df = pd.read_csv(label_file)
        self.attribute_names = label_df['attribute_name'].values
        self.NUM_CLASSES = 1103

        self.culture_class = np.load('./data/culture_class.npy')
        self.tag_class = np.load('./data/tag_class.npy')

    def on_loader_start(self, state):
        self.culture_outputs = []
        self.culture_labels = []

        self.tag_outputs = []
        self.tag_labels = []

    def merge_culture_tag(self, cultures, tags):
        merge_arr = np.zeros((cultures.shape[0], self.NUM_CLASSES))

        for i, culture in enumerate(cultures):
            cls_idx = np.where(culture == 1)[0]
            for idx in cls_idx:
                cls = self.culture_class[idx]
                merge_idx = np.where(self.attribute_names == cls)[0]
                merge_arr[i][merge_idx] = 1

        for i, tag in enumerate(tags):
            cls_idx = np.where(tag == 1)[0]
            for idx in cls_idx:
                cls = self.tag_class[idx]
                merge_idx = np.where(self.attribute_names == cls)[0]
                merge_arr[i][merge_idx] = 1

        return merge_arr

    def on_batch_end(self, state):
        culture_output = F.sigmoid(state.output[self.culture_output_key])
        culture_output = culture_output.detach().cpu().numpy()
        culture_label = state.input[self.culture_input_key].detach().cpu().numpy()
        self.culture_outputs.append(culture_output)
        self.culture_labels.append(culture_label)

        tag_output = F.sigmoid(state.output[self.tag_output_key])
        tag_output = tag_output.detach().cpu().numpy()
        tag_label = state.input[self.tag_input_key].detach().cpu().numpy()
        self.tag_outputs.append(tag_output)
        self.tag_labels.append(tag_label)

    def on_loader_end(self, state):
        import warnings
        warnings.filterwarnings("ignore")

        self.culture_outputs = np.concatenate(self.culture_outputs, axis=0)
        self.culture_labels = np.concatenate(self.culture_labels, axis=0)

        culture_score = fbeta_score(
            y_pred=self.culture_outputs > self.th,
            y_true=self.culture_labels,
            beta=self.beta,
            average="samples"
        )
        state.metrics.epoch_values[state.loader_name]['culture_score'] = culture_score

        self.tag_outputs = np.concatenate(self.tag_outputs, axis=0)
        self.tag_labels = np.concatenate(self.tag_labels, axis=0)

        tag_fscore = fbeta_score(
            y_pred=self.tag_outputs > self.th,
            y_true=self.tag_labels,
            beta=self.beta,
            average="samples"
        )
        state.metrics.epoch_values[state.loader_name]['tag_score'] = tag_fscore

        merge_predict = self.merge_culture_tag(self.culture_outputs > self.th, self.tag_outputs > self.th)
        merge_gt = self.merge_culture_tag(self.culture_labels, self.tag_labels)

        merge_score = fbeta_score(
            y_pred=merge_predict,
            y_true=merge_gt,
            beta=self.beta,
            average="samples"
        )
        state.metrics.epoch_values[state.loader_name]['merge_score'] = merge_score
