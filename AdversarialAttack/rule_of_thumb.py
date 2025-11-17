import torch

import numpy as np
from rot_class import RoT


class RuleOfThumb():

    def __init__(self, y_outputs, x_inputs, epochs=500, batch_size=5000, learning_rate=0.05, dropout_rate=0.5) -> None:
        y_preds = y_outputs.flatten()
        self._explainer_model = RoT(2, (x_inputs.shape[1],), dropout_rate=dropout_rate)
        xx = torch.from_numpy(x_inputs)
        yy = torch.from_numpy(y_preds)
        self._explainer_model.fit(xx.to(torch.float), yy, epochs=epochs, batch_size=batch_size, lr=learning_rate)

    def get_explanation(self, x_numpy) -> np.array:
        """
        Returns a random standard normal vector of shape x.shape.
        
        Args:
            x (torch.Tensor, [N x 1 x d] for tabular instance; [N x m x n x d] for image instance): feature tensor
            label (torch.Tensor, [N x ...]): labels to explain
        Returns:
            exp (torch.Tensor, [N x 1 x d] for tabular instance; [N x m x n x d] for image instance: instance level explanation):
        """

        x = torch.from_numpy(x_numpy)
        imp=self._explainer_model.importance(x).detach().numpy()
        imp=np.abs(imp).sum(1)
        imp=imp.reshape(imp.shape[0],-1)
        return imp # [(_feat, _imp) for _feat, _imp in zip(self._feature_names, imp)]

