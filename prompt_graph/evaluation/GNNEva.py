import torchmetrics
import torch
from tqdm import tqdm


def GNNNodeEva(loader, gnn, answering,embedding, num_class, device):
    gnn.eval()
    answering.eval()
    embedding.eval()

    accuracy = torchmetrics.classification.Accuracy(task="multiclass", num_classes=num_class).to(device)
    macro_f1 = torchmetrics.classification.F1Score(task="multiclass", num_classes=num_class, average="macro").to(device)
    auroc = torchmetrics.classification.AUROC(task="multiclass", num_classes=num_class).to(device)
    auprc = torchmetrics.classification.AveragePrecision(task="multiclass", num_classes=num_class).to(device)

    accuracy.reset()
    macro_f1.reset()
    auroc.reset()
    auprc.reset()

    with torch.no_grad():
        for batch_id, batch in enumerate(loader):
            batch = batch.to(device)
            
            batch.x = embedding(batch.x)
            
            graph_emb = gnn(batch.x, batch.edge_index, batch.batch)
            pre = answering(graph_emb)
            pred = pre.argmax(dim=1)

            acc = accuracy(pred, batch.y)
            ma_f1 = macro_f1(pred, batch.y)
            roc = auroc(pre, batch.y)
            prc = auprc(pre, batch.y)
            if len(loader) > 20:
                print("Batch {}/{} Acc: {:.4f} | Macro-F1: {:.4f}| AUROC: {:.4f}| AUPRC: {:.4f}".format(batch_id,len(loader), acc.item(), ma_f1.item(),roc.item(), prc.item()))

    acc = accuracy.compute()
    ma_f1 = macro_f1.compute()
    roc = auroc.compute()
    prc = auprc.compute()

    return acc.item(), ma_f1.item(), roc.item(), prc.item()


def GNNGraphEva(loader, gnn, answering,embedding,num_class, device):
    gnn.eval()
    accuracy = torchmetrics.classification.Accuracy(task="multiclass", num_classes=num_class).to(device)
    macro_f1 = torchmetrics.classification.F1Score(task="multiclass", num_classes=num_class, average="macro").to(device)
    auroc = torchmetrics.classification.AUROC(task="multiclass", num_classes=num_class).to(device)
    auprc = torchmetrics.classification.AveragePrecision(task="multiclass", num_classes=num_class).to(device)

    accuracy.reset()
    macro_f1.reset()
    auroc.reset()
    auprc.reset()
    if answering:
        answering.eval()
    with torch.no_grad(): 
        for batch_id, batch in enumerate(loader): 
            batch = batch.to(device) 
            
            embedding_x = embedding(batch.x)
            
            out = gnn(embedding_x, batch.edge_index, batch.batch)
            if answering:
                out = answering(out)  
            pred = out.argmax(dim=1)  
            acc = accuracy(pred, batch.y)
            ma_f1 = macro_f1(pred, batch.y)
            roc = auroc(out, batch.y)
            prc = auprc(out, batch.y)
            if len(loader) > 20:
                print("Batch {}/{} Acc: {:.4f} | Macro-F1: {:.4f}| AUROC: {:.4f}| AUPRC: {:.4f}".format(batch_id,len(loader), acc.item(), ma_f1.item(),roc.item(), prc.item()))

    acc = accuracy.compute()
    ma_f1 = macro_f1.compute()
    roc = auroc.compute()
    prc = auprc.compute()
       
    return acc.item(), ma_f1.item(), roc.item(), prc.item()