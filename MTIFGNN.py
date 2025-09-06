import torch
import torch.nn.functional as F
import torch.nn as nn
import torch_geometric as tg


class PEWE(nn.Module):
    def __init__(self, non_image_dim=2, embedding_dim=256, dropout=0.1):
        super(PEWE, self).__init__()
        hidden = 128
        self.parser = nn.Sequential(
            nn.Linear(non_image_dim, hidden, bias=True),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(hidden),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden, bias=True),
        )

        self.cos = nn.CosineSimilarity(dim=1, eps=1e-8)
        self.non_image_dim = non_image_dim
        self.embedding_dim = embedding_dim
        self.model_init()

    def forward(self, non_image_x, ess):
        # ess.shape = [871, 256]
        x1_non_image = non_image_x[:, 0:self.non_image_dim]  # shape [9149, 128]
        x1_es = ess[:, 0:self.embedding_dim]
        x2_non_image = non_image_x[:, self.non_image_dim:]  # shape [9149, 128]
        x2_es = ess[:, self.embedding_dim:]
        x1_non_image = self.parser(x1_non_image)
        x2_non_image = self.parser(x2_non_image)
        h1 = torch.cat((x1_non_image, x1_es), dim=1)
        h2 = torch.cat((x2_non_image, x2_es), dim=1)
        p = (self.cos(h1, h2) + 1) * 0.5
        return p

    def model_init(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                m.weight.requires_grad = True
                if m.bias is not None:
                    m.bias.data.zero_()
                    m.bias.requires_grad = True

class MTIF(torch.nn.Module):
    def __init__(self, config, input_dim=2000, embedding_dim=512, gcn_input_dim=2000, gcn_hidden_filters=16, gcn_layers=4,
                 dropout=0.2, num_classes=2, non_image_dim=2, edge_dropout=0.5):
        super(MTIF, self).__init__()
        self.config = config
        K = 3  # K-nearest neighbors
        self.dropout = dropout  # for non-image feature transformation
        self.edge_dropout = edge_dropout  # for population gcn
        self.gcn_layers = gcn_layers  #
        self.embedding_dim = embedding_dim
        self.relu = nn.ReLU(inplace=True)
        bias = False  # for ChebConv
        # disentangling modules
        self.EI = nn.Sequential(
            nn.Linear(input_dim, embedding_dim * 2),
            nn.Linear(embedding_dim * 2, embedding_dim)
        )
        self.ES = nn.Linear(in_features=input_dim, out_features=embedding_dim)

        # gcn
        self.gconv = nn.ModuleList()
        hidden = [gcn_hidden_filters for _ in range(gcn_layers)]
        for i in range(gcn_layers):
            in_channels = gcn_input_dim if i == 0 else hidden[i - 1]
            self.gconv.append(tg.nn.ChebConv(in_channels, hidden[i], K, normalization='sym', bias=bias))

        # 定义 1D 卷积层，输入通道为 num_features，输出通道可以自定义
        self.conv1d_layers = nn.ModuleList([
            nn.Conv1d(in_channels=16, out_channels=16, kernel_size=3, padding=1)
            for _ in range(gcn_layers)
        ])

        self.label_clf = nn.Sequential(
            nn.Linear(16, 256),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(256),
            nn.Dropout(self.dropout),
            nn.Linear(256, num_classes))

        # edge_net for constructing population graph
        self.edge_net = PEWE(non_image_dim=non_image_dim, embedding_dim=embedding_dim, dropout=self.dropout)
        self.model_init()

    def forward(self, image_features, edge_index, non_image_features, edge_index1, edge_weight1, enforce_edropout=False):
        if not self.config.device:
            device = torch.device("cuda:0")
        else:
            device = torch.device("cuda:1")
        if self.edge_dropout > 0:
            if enforce_edropout or self.training:
                one_mask = torch.ones(
                    [non_image_features.shape[0], 1])  # non-image-features.shape = torch.Size([18845, 6])
                drop_mask = F.dropout(one_mask, self.edge_dropout, True)
                bool_mask = torch.squeeze(drop_mask.type(torch.bool))
                edge_index = edge_index[:, bool_mask]  # dropout掉一部分，torch.Size([2, 9159])
                non_image_features = non_image_features[bool_mask]  # torch.Size([9159, 6])

        ei = self.EI(image_features)  # torch.Size([871, 256])
        es = ei.detach_()

        flatten_id = 0
        ES_M = torch.zeros((edge_index.shape[1], 2 * self.embedding_dim))

        for i in range(edge_index.shape[1]):
            # print(edge_index[:, i])
            source = edge_index[0, i]
            target = edge_index[1, i]
            ES_M[flatten_id] = torch.cat((es[source], es[target]), dim=0)
            flatten_id += 1

        ES_M = ES_M.to(device)
        edge_weight = torch.squeeze(self.edge_net(non_image_features, ES_M))

        ei = F.dropout(ei, self.dropout, self.training)  # torch.Size([871, 2000])
        h = self.relu(self.gconv[0](ei, edge_index, edge_weight))
        graph = self.relu(self.gconv[0](ei, edge_index1, edge_weight1))

        h = h.unsqueeze(0).permute(0, 2, 1)
        graph = graph.unsqueeze(0).permute(0, 2, 1)
        h = self.conv1d_layers[0](h)
        graph = self.conv1d_layers[0](graph)

        # 应用最大池化，kernel_size=1 保持长度不变
        max_pooled = F.max_pool1d(h, kernel_size=1)
        max_pooled_1 = F.max_pool1d(graph, kernel_size=1)
        # 应用最大池化，kernel_size=1 保持长度不变
        avg_pooled = F.avg_pool1d(h, kernel_size=1)
        avg_pooled_1 = F.avg_pool1d(graph, kernel_size=1)

        # 将平均池化和最大池化的结果合并
        combined = max_pooled + avg_pooled  # (1, 16, 871)
        combined_1 = max_pooled_1 + avg_pooled_1

        h = combined.squeeze(0).permute(1, 0)
        h0 = h
        h1 = h
        graph = combined_1.squeeze(0).permute(1, 0)
        graph0 = graph
        graph1 = graph

        # PGC操作
        for i in range(1, self.gcn_layers):
            h1 = F.dropout(h1, self.dropout, self.training)
            h = self.relu(self.gconv[i](h1, edge_index, edge_weight))
            graph1 = F.dropout(graph1, self.dropout, self.training)
            graph = self.relu(self.gconv[i](graph1, edge_index1, edge_weight1))

            # GCN 层输出，准备进入 Conv1d
            h = h.unsqueeze(0).permute(0, 2, 1)  # (1, num_features, num_nodes)
            h = self.conv1d_layers[i](h)
            graph = graph.unsqueeze(0).permute(0, 2, 1)  # (1, num_features, num_nodes)
            graph = self.conv1d_layers[i](graph)

            # 应用最大池化，kernel_size=1 保持长度不变
            max_pooled = F.max_pool1d(h, kernel_size=1)
            max_pooled_1 = F.max_pool1d(graph, kernel_size=1)
            # 应用最大池化，kernel_size=1 保持长度不变
            avg_pooled = F.avg_pool1d(h, kernel_size=1)
            avg_pooled_1 = F.avg_pool1d(graph, kernel_size=1)

            combined = max_pooled + avg_pooled
            combined_1 = max_pooled_1 + avg_pooled_1

            h = combined.squeeze(0).permute(1, 0)  # 恢复为 (num_nodes, num_features)
            graph = combined_1.squeeze(0).permute(1, 0)  # 恢复为 (num_nodes, num_features)

            # 残差
            h1 = h1 + h
            graph1 = graph1 + graph

            jk = h


        label_logits = self.label_clf(jk)  # 871 x 2

        # return
        return label_logits

    def model_init(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                torch.nn.init.kaiming_normal_(m.weight)
                m.weight.requires_grad = True
                if m.bias is not None:
                    m.bias.data.zero_()
                    m.bias.requires_grad = True
