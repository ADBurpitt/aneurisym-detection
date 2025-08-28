import torch
import numpy as np
from src import slice_bag as sb
from src.utils import extract_targets, unpack_sample_from_batch
from src.metrics import weighted_auc_14


def forward_one(sample, model, device, criterion, target_cols, *, k, stride, resize):
    bag, _ = sb.make_slice_bag(sample, k=k, stride=stride, resize=resize)  # [N,K,H,W]
    bag = bag.to(device)
    logits_inst, logits_bag = model(bag)  # [N,C], [1,C]
    y = extract_targets(sample, target_cols).to(device)  # [C]
    loss = criterion(logits_bag.squeeze(0), y)
    return loss, logits_bag.detach().cpu().squeeze(0), y.detach().cpu()


@torch.no_grad()
def eval_loader_full(loader, model, device, target_cols, *, k, stride, resize):
    model.eval()
    logits, targets = [], []
    for batch in loader:
        sample = unpack_sample_from_batch(batch)
        bag, _ = sb.make_slice_bag(sample, k=k, stride=stride, resize=resize)
        bag = bag.to(device)
        _, logit_bag = model(bag)
        y = extract_targets(sample, target_cols)
        logits.append(torch.sigmoid(logit_bag.squeeze(0)).cpu().numpy())
        targets.append(y.cpu().numpy())
    y_pred = np.stack(logits, 0)
    y_true = np.stack(targets, 0)
    return weighted_auc_14(target_cols, y_true, y_pred)


@torch.no_grad()
def eval_val_union(
    dl_list, model, device, target_cols, *, k, stride, resize, verbose=True
):
    model.eval()
    pred_map, targ_map = {}, {}
    for loader in dl_list:
        if loader is None:
            continue
        for batch in loader:
            sample = unpack_sample_from_batch(batch)
            uid = sample["uid"]
            bag, _ = sb.make_slice_bag(sample, k=k, stride=stride, resize=resize)
            bag = bag.to(device)
            _, logit_bag = model(bag)
            probs = torch.sigmoid(logit_bag.squeeze(0)).cpu().numpy()
            y = extract_targets(sample, target_cols).cpu().numpy()
            pred_map.setdefault(uid, []).append(probs)
            targ_map[uid] = y
    if not pred_map:
        return float("nan")
    uids = sorted(pred_map.keys())
    y_pred = np.stack([np.mean(pred_map[u], 0) for u in uids], 0)
    y_true = np.stack([targ_map[u] for u in uids], 0)
    return weighted_auc_14(target_cols, y_true, y_pred)


def run_epoch(
    loader_or_pair,
    model,
    device,
    optimizer,
    scaler,
    train,
    target_cols,
    *,
    criterion,
    k,
    stride,
    resize,
):
    model.train(train)
    from src.utils import alt_iter

    if isinstance(loader_or_pair, (tuple, list)) and len(loader_or_pair) == 2:
        iterator = alt_iter(loader_or_pair[0], loader_or_pair[1])
        num_steps = len(loader_or_pair[0]) + len(loader_or_pair[1])

        def pick(x):
            return x[0]
    else:
        iterator = loader_or_pair
        num_steps = len(loader_or_pair)

        def pick(x):
            return x

    running = 0.0
    preds_ap, targs_ap = [], []
    ap_idx = target_cols.index("Aneurysm Present")

    for item in iterator:
        batch = pick(item)
        sample = unpack_sample_from_batch(batch)

        if train:
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=(device.type == "cuda"),
            ):
                loss, logits_bag_cpu, y_cpu = forward_one(
                    sample,
                    model,
                    device,
                    criterion,
                    target_cols,
                    k=k,
                    stride=stride,
                    resize=resize,
                )
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            with (
                torch.no_grad(),
                torch.autocast(
                    device_type=device.type,
                    dtype=torch.float16,
                    enabled=(device.type == "cuda"),
                ),
            ):
                loss, logits_bag_cpu, y_cpu = forward_one(
                    sample,
                    model,
                    device,
                    criterion,
                    target_cols,
                    k=k,
                    stride=stride,
                    resize=resize,
                )

        running += float(loss.item())
        preds_ap.append(torch.sigmoid(torch.as_tensor(logits_bag_cpu[ap_idx])).item())
        targs_ap.append(float(y_cpu[ap_idx].item()))

    from sklearn.metrics import roc_auc_score

    avg_loss = running / max(1, num_steps)
    auc_ap = None
    try:
        auc_ap = roc_auc_score(targs_ap, preds_ap) if len(set(targs_ap)) > 1 else None
    except Exception:
        auc_ap = None
    return avg_loss, auc_ap
