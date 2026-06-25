import torch


def pixelwise_weighted_binary_crossentropy2(weight_scal, with_logits=True, eps=1e-6):
    print(f"Using weight_scaling = {weight_scal}")
    def pixelwise_crossentropy(y_pred, y_true):
        if with_logits:
            y_pred = torch.sigmoid(y_pred)
        weight = (y_true * weight_scal) + 1
        
        y_pred = torch.clip(y_pred, min=eps, max=1.0 - eps)
        y_pred = torch.log(y_pred / (1.0 - y_pred))

        zeros = torch.zeros_like(y_pred, dtype=y_pred.dtype)
        cond = y_pred >= zeros
        relu_logits = torch.where(cond, y_pred, zeros)
        neg_abs_logits = torch.where(cond, -y_pred, y_pred)

        entropy = relu_logits - y_pred * y_true + torch.log1p(torch.exp(neg_abs_logits))
        loss = torch.mean(weight * entropy)
        return loss

    return pixelwise_crossentropy
