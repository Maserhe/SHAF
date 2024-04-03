import torch
from torch import nn
import torch.nn.functional as F


class SelfAttention(nn.Module):
    def __init__(self, hidden_size, num_heads):
        super(SelfAttention, self).__init__()

        self.multihead_attention = nn.MultiheadAttention(hidden_size, num_heads)
        self.fc = nn.Linear(hidden_size, hidden_size)
        self.dropout = nn.Dropout(0.1)

    def forward(self, query, key, value):
        attended, _ = self.multihead_attention(query, key, value)

        attended = self.fc(attended)
        # attended = self.dropout(F.relu(attended))

        attended = attended + query  # 残差连接
        return attended


class SelfAttentionFusion(nn.Module):
    def __init__(self, text_shape, image_shape, hidden_size, num_heads, num_layers, output_dim, use_ln):
        super(SelfAttentionFusion, self).__init__()

        self.text_embedding = nn.Linear(text_shape[-1], hidden_size)
        self.image_embedding = nn.Linear(image_shape[-1], hidden_size)

        self.attention_layers = nn.ModuleList([
            SelfAttention(hidden_size, num_heads)
            for _ in range(num_layers)
        ])

        self.use_ln = use_ln
        if self.use_ln:
            self.ln = nn.LayerNorm(hidden_size)

        self.fc = nn.Linear(hidden_size, output_dim * 2)

    def forward(self, text_features, image_features):
        text_embedding = self.text_embedding(text_features)
        image_embedding = self.image_embedding(image_features)

        text_embedding = text_embedding.permute(1, 0, 2)  # 调整形状为(seq_len, batch_size, hidden_size)
        image_embedding = image_embedding.permute(1, 0, 2)  # 调整形状为(seq_len, batch_size, hidden_size)

        query = text_embedding
        key = value = image_embedding

        for layer in self.attention_layers:
            query = layer(query, key, value)

        if query.size(1) > 1:
            fused_features = torch.mean(query, dim=0)  # 平均池化，除非批量大小为1

        else:
            fused_features = query[0]  # 批量大小为1时，直接使用查询结果

        if self.use_ln:
            fused_features = self.ln(fused_features)

        # print(fused_features.shape)
        fused_features = self.fc(fused_features)

        return fused_features  # 添加批量维度


class DualAttention(nn.Module):
    def __init__(self, input_dim1, input_dim2, output_dim, num_heads=1):
        super(DualAttention, self).__init__()

        # Linear layers for mapping to the specified dimension
        self.linear1 = nn.Linear(input_dim1, output_dim)
        self.linear2 = nn.Linear(input_dim2, output_dim)

        # Multi-Head Self-Attention
        self.attention1 = nn.MultiheadAttention(output_dim, num_heads)
        self.attention2 = nn.MultiheadAttention(output_dim, num_heads)

    def forward(self, vec1, vec2):
        # Map vec1 and vec2 to the specified output dimension
        vec1_mapped = self.linear1(vec1)
        vec2_mapped = self.linear2(vec2)

        # Transpose to fit Multi-Head Attention input shape (seq_len, batch_size, embed_dim)
        vec1_mapped = vec1_mapped.permute(1, 0, 2)
        vec2_mapped = vec2_mapped.permute(1, 0, 2)

        # Compute the first attention using vec1 as keys and queries and vec2 as values
        attn_output1, _ = self.attention1(vec1_mapped, vec2_mapped, vec2_mapped)

        # Compute the second attention using vec2 as keys and queries and vec1 as values
        attn_output2, _ = self.attention2(vec2_mapped, vec1_mapped, vec1_mapped)

        # Transpose the attention outputs back to (batch_size, seq_len, embed_dim)
        attn_output1 = attn_output1.permute(1, 0, 2)
        attn_output2 = attn_output2.permute(1, 0, 2)

        # Add the results of the two attentions
        final_output = attn_output1 + attn_output2

        return final_output.flatten(start_dim=1)


class Combiner(nn.Module):
    """
    Combiner module which once trained fuses textual and visual information
    """

    def __init__(self, clip_feature_dim: int, projection_dim: int, hidden_dim: int):
        """
        :param clip_feature_dim: CLIP input feature dimension
        :param projection_dim: projection dimension
        :param hidden_dim: hidden dimension
        """
        super(Combiner, self).__init__()
        self.text_projection_layer = nn.Linear(clip_feature_dim, projection_dim)
        self.image_projection_layer = nn.Linear(clip_feature_dim, projection_dim)
        self.late_projection_layer = nn.Linear(5808  , projection_dim)

        self.late_clip_layer = nn.Linear(projection_dim, clip_feature_dim)

        self.dropout1 = nn.Dropout(0.5)
        self.dropout2 = nn.Dropout(0.5)
        self.dropout3 = nn.Dropout(0.1)
        self.dropout4 = nn.Dropout(0.1)
        self.dropout5 = nn.Dropout(0.5)

        self.combiner_layer = nn.Linear(projection_dim * 3, hidden_dim)
        self.output_layer = nn.Linear(hidden_dim, clip_feature_dim)

        self.dynamic_scalar = nn.Sequential(nn.Linear(projection_dim * 3, hidden_dim), nn.ReLU(), nn.Dropout(0.5), nn.Linear(hidden_dim, 3), nn.Softmax(dim=-1))

        # self.dynamic_scalar = nn.Sequential(nn.Linear(projection_dim * 3, hidden_dim), nn.ReLU(), nn.Dropout(0.5), nn.Linear(hidden_dim, 1), nn.Sigmoid())

        self.atten = DualAttention(768, 512, 484, num_heads=1) # 12
        # self.late_fusion_layer = nn.Sequential(nn.Linear(12288, projection_dim * 2), nn.ReLU(), nn.Linear(projection_dim * 2, clip_feature_dim))


        # self.global_fusion = SelfAttentionFusion(text_shape=(77, 512), image_shape=(50, 768), hidden_size=512, output_dim=projection_dim, num_heads=4, num_layers=3)
        # self.global_fusion = SelfAttentionFusion(text_shape=(8, 512), image_shape=(50, 768), hidden_size=512, output_dim=projection_dim, num_heads=8, num_layers=4, use_ln=False)

        self.logit_scale = 100


    def forward(self, image_features: torch.tensor, text_features: torch.tensor,
                target_features: torch.tensor, global_image_features: torch.tensor, global_text_features: torch.tensor) -> torch.tensor:
        """
        Takes as input a triplet: image_features, text_features and target_features and outputs the logits which are
        the normalized dot product between the predicted features and the target_features.
        The logits are also multiplied by logit_scale parameter
        :param image_features: CLIP reference image features
        :param text_features: CLIP relative caption features
        :param target_features: CLIP target image features
        :return: scaled logits
        """

        predicted_features, global_features = self.combine_features(image_features, text_features, global_image_features, global_text_features)
        target_features = F.normalize(target_features, dim=-1)

        logits = self.logit_scale * predicted_features @ target_features.T
        return logits

    def combine_features(self, image_features: torch.tensor, text_features: torch.tensor, global_image_features: torch.tensor, global_text_features: torch.tensor) -> torch.tensor:
        """
        Combine the reference image features and the caption features. It outputs the predicted features
        :param image_features: CLIP reference image features
        :param text_features: CLIP relative caption features
        :return: predicted features
        """
        text_projected_features = self.dropout1(F.relu(self.text_projection_layer(text_features)))
        image_projected_features = self.dropout2(F.relu(self.image_projection_layer(image_features)))
        #
        # # 12288
        global_features = self.atten(global_image_features, global_text_features)
        #
        # # projection_dim
        late_projected_features = self.dropout4(F.relu(self.late_projection_layer(global_features)))

        # clip_feature_dim
        late_features = self.late_clip_layer(late_projected_features)


        raw_combined_features = torch.cat((text_projected_features, image_projected_features, late_projected_features), -1)
        #
        # # print(raw_combined_features.shape)
        #
        combined_features = self.dropout5(F.relu(self.combiner_layer(raw_combined_features)))
        #
        dynamic_scalar = self.dynamic_scalar(raw_combined_features)
        #
        # # print(dynamic_scalar[0])
        output = self.output_layer(combined_features) + dynamic_scalar[:, 0:1] * text_features + dynamic_scalar[:, 1:2] * image_features + dynamic_scalar[:, 2:] * late_features

        # filename = "F:\\FashionCLIP_Retrieval4cir\\test.csv"
        #
        # # 追加写入内容到文件
        # with open(filename, "a") as file:
        #     file.write(str(dynamic_scalar[:, 0:1].mean().item()) + "," + str(dynamic_scalar[:, 1:2].mean().item()) + "," + str(dynamic_scalar[:, 2:].mean().item()) + "\n")
        # file.close()

        # output = self.output_layer(combined_features) + dynamic_scalar * text_features + (1 - dynamic_scalar) * image_features

        return F.normalize(output, dim=-1), F.normalize(global_features, dim=-1)
