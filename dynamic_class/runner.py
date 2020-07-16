from typing import Mapping

import torch
from catalyst import dl


class CustomRunner(dl.Runner):
    def __init__(self, *, prototypes: torch.Tensor = None, **kwargs):
        super().__init__(self)
        self.parked_callbacks = []
        self.proto_encodings = []
        self.proto_targets = []

        if prototypes is not None:
            assert isinstance(
                prototypes, torch.Tensor
            ), "Expect prototypes as torch.Tensor"
        self.prototypes = prototypes

    def _run_event(self, event: str):
        super()._run_event(event)
        if event == "on_loader_start" and self.state.loader_name == "prototypes":
            self.proto_encodings = []
            self.proto_targets = []
            self.parked_callbacks = []
            key = 0
            item = self.state.callbacks.pop(key)
            self.parked_callbacks.append((key, item))
        if event == "on_loader_end" and self.state.loader_name == "prototypes":
            for k, v in self.parked_callbacks:
                self.state.callbacks[k] = v
            self.state.callbacks.move_to_end(0, last=False)

            encodings = torch.cat(self.proto_encodings, dim=0)
            labels = torch.cat(self.proto_targets, dim=0)
            self.prototypes = self.model.compute_prototypes(encodings, labels)

    def _handle_batch(self, batch: Mapping[str, torch.Tensor]):
        if self.state.loader_name == "train":
            for k, v in self.state.input.items():
                self.state.input[k] = v.squeeze()

            outputs = self.model.forward(
                query=batch["query"].squeeze(),
                support=batch["support"].squeeze(),
                support_label=batch["support_label"].squeeze(),
            )
            self.state.output = outputs

        if self.state.loader_name == "prototypes":
            encodings = self.model.encode(text=batch["inputs"])
            self.proto_encodings.append(encodings)
            self.proto_targets.append(batch["targets"])

        if self.state.loader_name == "valid":
            self.state.input["query_label"] = batch["targets"]

            outputs = self.model.forward(
                query=batch["inputs"], prototypes=self.prototypes
            )
            self.state.output = outputs

            val_metric = (
                (outputs["logits"].argmax(dim=1) == batch["targets"])
                .float()
                .mean()
                .item()
            )
            batch_metrics = {"acc": val_metric}
            self.state.batch_metrics.update(**batch_metrics)

    def predict_batch(self, batch: Mapping[str, torch.Tensor]):
        batch = self._batch2device(batch, self.device)
        if self.prototypes is None:
            embeddings = self.model.encode(batch["inputs"])
            return {"embeddings": embeddings}
        return self.model.forward(
            query=batch["inputs"],
            prototypes=self._batch2device(self.prototypes, self.device),
        )
