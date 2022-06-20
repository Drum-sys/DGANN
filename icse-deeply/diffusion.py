def forward(self, inp, adj):
        """
        inp: input_fea [N, in_features]  in_features表示节点的输入特征向量元素个数
        adj: 图的邻接矩阵 维度[N, N] 非零即一，数据结构基本知识
        """
        h = torch.mm(inp, self.W)   # [N, out_features]
        N = h.size()[0]    # N 图的节点数
        
        for i in range steps:
            
            a_input = torch.cat([h.repeat(1, N).view(N*N, -1), h.repeat(N, 1)], dim=1).view(N, -1, 2*self.out_features)
            # [N, N, 2*out_features]
            e = self.leakyrelu(torch.matmul(a_input, self.a).squeeze(2))
            # [N, N, 1] => [N, N] 图注意力的相关系数（未归一化）
            
            zero_vec = -1e12 * torch.ones_like(e)    # 将没有连接的边置为负无穷
            attention = torch.where(adj>0, e, zero_vec)   # [N, N]
            # 表示如果邻接矩阵元素大于0时，则两个节点有连接，该位置的注意力系数保留，
            # 否则需要mask并置为非常小的值，原因是softmax的时候这个最小值会不考虑。
            attention = F.softmax(attention, dim=1)    # softmax形状保持不变 [N, N]，得到归一化的注意力权重！
            adj = adj * adj
            
        attention = F.dropout(attention, self.dropout, training=self.training)   # dropout，防止过拟合
        h_prime = torch.matmul(attention, h)  # [N, N].[N, out_features] => [N, out_features]
        # 得到由周围节点通过注意力权重进行更新的表示
        if self.concat:
            return F.elu(h_prime)
        else:
            return h_prime