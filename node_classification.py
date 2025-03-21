import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.datasets import Planetoid
import torch_geometric.transforms as T
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns

# Define GNN Model for Node Classification
class GCN_node(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_classes, dropout=0.5):
        super(GCN_node, self).__init__()
        self.conv1 = nn.Linear(input_dim, hidden_dim)
        self.conv2 = nn.Linear(hidden_dim, num_classes)
        self.dropout = dropout

    def forward(self, x, edge_index):
        A_hat = torch.eye(x.size(0), device=x.device)
        A_hat[edge_index[0], edge_index[1]] = 1
        D_hat = torch.diag_embed(1.0 / torch.sqrt(A_hat.sum(dim=1) + 1e-5))
        A_norm = D_hat @ A_hat @ D_hat
        x = F.relu(A_norm @ self.conv1(x))
        x = F.dropout(x, p=self.dropout, training=self.training)
        return self.conv2(x)

# Training Function
def train(model, data, optimizer, criterion):
    model.train()
    optimizer.zero_grad()
    out = model(data.x, data.edge_index)
    loss = criterion(out[data.train_mask], data.y[data.train_mask])
    loss.backward()
    optimizer.step()
    return loss.item()

# Evaluation Function
def evaluate(model, data):
    model.eval()
    with torch.no_grad():
        logits = model(data.x, data.edge_index)
        pred = logits.argmax(dim=1)
        acc = (pred[data.test_mask] == data.y[data.test_mask]).sum().item() / data.test_mask.sum().item()
    return acc

# Extract Embeddings for Visualization
def extract_embeddings(model, data):
    model.eval()
    with torch.no_grad():
        embeddings = model(data.x, data.edge_index)
    return embeddings.cpu().numpy()

# Visualize t-SNE Embeddings
def visualize_tsne(embeddings, labels):
    tsne = TSNE(n_components=2, perplexity=30, random_state=42)
    node_embeddings_2D = tsne.fit_transform(embeddings)
    plt.figure(figsize=(10, 8))
    sns.scatterplot(x=node_embeddings_2D[:, 0], y=node_embeddings_2D[:, 1], hue=labels, palette="bright", s=30, alpha=0.8)
    plt.title("t-SNE Visualization of Node Embeddings")
    plt.xlabel("t-SNE Component 1")
    plt.ylabel("t-SNE Component 2")
    plt.legend(title="Classes")
    plt.show()