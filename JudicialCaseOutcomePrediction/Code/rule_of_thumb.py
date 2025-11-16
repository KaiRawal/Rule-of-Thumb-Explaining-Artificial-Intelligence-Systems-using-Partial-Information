import torch

import numpy as np
from rot_class import RoT_text


class RuleOfThumb():

    def __init__(self, y_outputs, x_inputs, epochs=500, batch_size=5000, learning_rate=0.05, dropout_rate=0.5, l1_penalty=0.0, write_file=None) -> None:
        y_preds = y_outputs.flatten()
        self._explainer_model = RoT_text(2, (x_inputs.shape[1],x_inputs.shape[2]), dropout_rate=dropout_rate, l1_penalty=l1_penalty)
        xx = torch.from_numpy(x_inputs)
        yy = torch.from_numpy(y_preds)
        if write_file is None:
            write_file = f"./DATA/TOKEN_EXPS_{epochs}_{batch_size}_{learning_rate}/hyperparams.txt"
        self._explainer_model.fit(xx.to(torch.float), yy, epochs=epochs, batch_size=batch_size, lr=learning_rate, write_file=write_file)

    def fit_existing_model(self, y_outputs, x_inputs, epochs=500, batch_size=5000, learning_rate=0.05, append_file=None) -> None:
        y_preds = y_outputs.flatten()
        xx = torch.from_numpy(x_inputs)
        yy = torch.from_numpy(y_preds)
        if append_file is None:
            append_file = f"./DATA/TOKEN_EXPS_{epochs}_{batch_size}_{learning_rate}/hyperparams.txt"
        return self._explainer_model.continue_fit(xx.to(torch.float), yy, epochs=epochs, batch_size=batch_size, lr=learning_rate, append_file=append_file)

    def get_explanation(self, x_numpy) -> np.array:
        """
        Returns a random standard normal vector of shape x.shape.

        Args:
            x (torch.Tensor, [N x 1 x d] for tabular instance; [N x m x n x d] for image instance): feature tensor
            label (torch.Tensor, [N x ...]): labels to explain
        Returns:
            exp (torch.Tensor, [N x 1 x d] for tabular instance; [N x m x n x d] for image instance: instance level explanation):
        """

        x = torch.from_numpy(x_numpy).to(torch.float)
        imp = self._explainer_model.importance(x).detach().numpy()[:, :, 1]
        # print(f'{imp.shape=}')

        #imp = imp[:, 1, :, :].sum(axis=2)


        return imp
