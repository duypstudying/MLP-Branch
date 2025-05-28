import torch
import torch.nn.functional as F
import torch_geometric
import numpy as np


class PreNormException(Exception):
    pass


sing网络)
class BipartiteGraphConvolution(torch_geometric.nn.MessagePassing):
    def __init__(self):class PreNormLayer(torch.nn.Module):  # 归一化
    def __init__(self, n_units, shift=True, scale=True, name=None):
        super().__init__()
        assert shift or scale
        self.register_buffer('shift', torch.zeros(n_units) if shift else None)
        self.register_buffer('scale', torch.ones(n_units) if scale else None)
        self.n_units = n_units
        self.waiting_updates = False
        self.received_updates = False

    def forward(self, input_):
        if self.waiting_updates:
            self.update_stats(input_)
            self.received_updates = True
            raise PreNormException

        if self.shift is not None:
            input_ = input_ + self.shift

        if self.scale is not None:
            input_ = input_ * self.scale

        return input_

    def start_updates(self):
        self.avg = 0
        self.var = 0
        self.m2 = 0
        self.count = 0
        self.waiting_updates = True
        self.received_updates = False

    def update_stats(self, input_):
        """
        Online mean and variance estimation. See: Chan et al. (1979) Updating
        Formulae and a Pairwise Algorithm for Computing Sample Variances.
        https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Online_algorithm
        """
        assert self.n_units == 1 or input_.shape[
            -1] == self.n_units, f"Expected input dimension of size {self.n_units}, got {input_.shape[-1]}."

        input_ = input_.reshape(-1, self.n_units)
        sample_avg = input_.mean(dim=0)
        sample_var = (input_ - sample_avg).pow(2).mean(dim=0)
        sample_count = np.prod(input_.size()) / self.n_units

        delta = sample_avg - self.avg

        self.m2 = self.var * self.count + sample_var * sample_count + delta ** 2 * self.count * sample_count / (
                self.count + sample_count)

        self.count += sample_count
        self.avg += delta * sample_count / self.count
        self.var = self.m2 / self.count if self.count > 0 else 1

    def stop_updates(self):
        """
        Ends pre-training for that layer, and fixes the layers's parameters.
        """
        assert self.count > 0
        if self.shift is not None:
            self.shift = -self.avg

        if self.scale is not None:
            self.var[self.var < 1e-8] = 1
            self.scale = 1 / torch.sqrt(self.var)

        del self.avg, self.var, self.m2, self.count
        self.waiting_updates = False
        self.trainable = False


# 二部图卷积网络，继承了 MPNN(Message Pas
        super().__init__('add')  # 聚合模式为相加
        emb_size = 64

        self.feature_module_left = torch.nn.Sequential(  # 将node1经过一层LP，得到的形状不变(n, 64)
            torch.nn.Linear(emb_size, emb_size)
        )
        self.feature_module_edge = torch.nn.Sequential(  # 将边结点经过一层LP，形状由(e, 1)变为(e, 64)
            torch.nn.Linear(1, emb_size, bias=False)
        )
        self.feature_module_right = torch.nn.Sequential(  # 将node2经过一层LP，得到的形状不变(n, 64)
            torch.nn.Linear(emb_size, emb_size, bias=False)
        )
        self.feature_module_final = torch.nn.Sequential(
            PreNormLayer(1, shift=False),
            torch.nn.ReLU(),
            torch.nn.Linear(emb_size, emb_size)
        )

        self.post_conv_module = torch.nn.Sequential(
            PreNormLayer(1, shift=False)
        )

        # output_layers
        self.output_module = torch.nn.Sequential(
            # 将形状为 (e, 64)的两个矩阵拼接到一起，得到一个(e, 128)的矩阵
            # 最后通过两层MLP，转为(e, 64)的矩阵
            torch.nn.Linear(2 * emb_size, emb_size),
            torch.nn.ReLU(),
            torch.nn.Linear(emb_size, emb_size),
        )

    def forward(self, left_features, edge_indices, edge_features, right_features):
        # 这是信息传递过程的主要驱动函数。它负责调用message、aggregate和update函数，并将结果传递给下一层
        output = self.propagate(edge_indices, size=(left_features.shape[0], right_features.shape[0]),
                                node_features=(left_features, right_features), edge_features=edge_features)
        # 最后update将(e, 64)的矩阵顶点相同的聚合获得(n, 64)的矩阵
        # 两个(n, 64)的结点拼接在一起，得到(n, 128)的矩阵，通过输出层得到(n, 64)的结果
        return self.output_module(torch.cat([self.post_conv_module(output), right_features], dim=-1))

    def message(self, node_features_i, node_features_j, edge_features):
        # 通过边索引，将获得两个结点与边的矩阵(e ,64)
        output = self.feature_module_final(self.feature_module_left(node_features_i)
                                           + self.feature_module_edge(edge_features)
                                           + self.feature_module_right(node_features_j))
        # 将三个矩阵相加聚合
        return output


class BaseModel(torch.nn.Module):
    """
    Our base model class, which implements pre-training methods.
    """

    def pre_train_init(self):
        for module in self.modules():
            if isinstance(module, PreNormLayer):
                module.start_updates()

    def pre_train_next(self):
        for module in self.modules():
            if isinstance(module, PreNormLayer) and module.waiting_updates and module.received_updates:
                module.stop_updates()
                return module
        return None

    def pre_train(self, *args, **kwargs):
        try:
            with torch.no_grad():
                self.forward(*args, **kwargs)
            return False
        except PreNormException:
            return True


class GNNPolicy(BaseModel):
    def __init__(self):
        super().__init__()
        emb_size = 64  # 嵌入后结点的特征个数
        cons_nfeats = 5  # 约束条件的特征个数
        edge_nfeats = 1  # 边特征个数
        var_nfeats = 19  # 变量的特征个数

        # CONSTRAINT EMBEDDING
        # 约束条件的嵌入，将原 (m，5)的矩阵经过两层全连接层嵌入成为 (m ,64)的矩阵，全连接层的激活函数为 ReLU
        self.cons_embedding = torch.nn.Sequential(
            PreNormLayer(cons_nfeats),  # 归一化
            torch.nn.Linear(cons_nfeats, emb_size),  # MLP1
            torch.nn.ReLU(),
            torch.nn.Linear(emb_size, emb_size),  # MLP2
            torch.nn.ReLU(),
        )

        # EDGE EMBEDDING
        # 将边的特征进行嵌入归一化，矩阵形状保持不变 ，始终为(e, 1)
        self.edge_embedding = torch.nn.Sequential(
            PreNormLayer(edge_nfeats),  # 归一化
        )

        # VARIABLE EMBEDDING
        # 变量特征的嵌入，将原 (n，19)的矩阵经过两层全连接层嵌入成为 (n ,64)的矩阵，全连接层的激活函数为 ReLU
        self.var_embedding = torch.nn.Sequential(
            PreNormLayer(var_nfeats),  # 归一化
            torch.nn.Linear(var_nfeats, emb_size),  # MLP1
            torch.nn.ReLU(),
            torch.nn.Linear(emb_size, emb_size),  # MLP2
            torch.nn.ReLU(),
        )

        self.conv_v_to_c = BipartiteGraphConvolution()  # 变量向约束条件的卷积
        self.conv_c_to_v = BipartiteGraphConvolution()  # 约束条件向变量的卷积

        # softmax层 ，最后得到关于决策变量分支的概率分布
        self.output_module = torch.nn.Sequential(
            torch.nn.Linear(emb_size, emb_size),
            torch.nn.ReLU(),
            torch.nn.Linear(emb_size, 1, bias=False),
        )

    def forward(self, constraint_features, edge_indices, edge_features, variable_features):
        # 后续卷积中需要首先将边特征卷积入变量或约束条件特征中，且图为无向图
        # 所以需要翻转边索引，由变量向约束条件翻成约束条件向变量，方便后续约束条件向变量的卷积
        reversed_edge_indices = torch.stack([edge_indices[1], edge_indices[0]], dim=0)

        constraint_features = self.cons_embedding(constraint_features)  # 约束条件特征嵌入
        edge_features = self.edge_embedding(edge_features)  # 边特征嵌入
        variable_features = self.var_embedding(variable_features)  # 变量特征嵌入

        # 约束条件向变量的卷积，最后得到约束条件特征的卷积后的矩阵形状为(m, 64)
        constraint_features = self.conv_v_to_c(variable_features, reversed_edge_indices, edge_features,constraint_features)
        # 变量向约束条件的卷积，最后得到变量特征的卷积后的矩阵形状为(n, 64)
        variable_features = self.conv_c_to_v(constraint_features, edge_indices, edge_features, variable_features)

        # softmax
        output = self.output_module(variable_features).squeeze(-1)
        return output
