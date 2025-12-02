# @Time   : 2020/6/25
# @Author : Shanlei Mu
# @Email  : slmu@ruc.edu.cn

# UPDATE:
# @Time   : 2022/7/16, 2020/8/6, 2020/8/25, 2023/4/24
# @Author : Zhen Tian, Shanlei Mu, Yupeng Hou, Chenglong Ma
# @Email  : chenyuwuxinn@gmail.com, slmu@ruc.edu.cn, houyupeng@ruc.edu.cn, chenglong.m@outlook.com

"""
recbole.model.abstract_recommender
##################################
"""

from logging import getLogger

import numpy as np
import torch
import torch.nn as nn

from recbole.model.layers import FMEmbedding, FMFirstOrderLinear, FLEmbedding
from recbole.utils import ModelType, InputType, FeatureSource, FeatureType, set_color


class AbstractRecommender(nn.Module):
    r"""Base class for all models"""

    def __init__(self):
        self.logger = getLogger()
        super(AbstractRecommender, self).__init__()

    def calculate_loss(self, interaction):
        r"""Calculate the training loss for a batch data.

        Args:
            interaction (Interaction): Interaction class of the batch.

        Returns:
            torch.Tensor: Training loss, shape: []
        """
        raise NotImplementedError

    def predict(self, interaction):
        r"""Predict the scores between users and items.

        Args:
            interaction (Interaction): Interaction class of the batch.

        Returns:
            torch.Tensor: Predicted scores for given users and items, shape: [batch_size]
        """
        raise NotImplementedError

    def full_sort_predict(self, interaction):
        r"""full sort prediction function.
        Given users, calculate the scores between users and all candidate items.

        Args:
            interaction (Interaction): Interaction class of the batch.

        Returns:
            torch.Tensor: Predicted scores for given users and all candidate items,
            shape: [n_batch_users * n_candidate_items]
        """
        raise NotImplementedError

    def other_parameter(self):
        if hasattr(self, "other_parameter_name"):
            return {key: getattr(self, key) for key in self.other_parameter_name}
        return dict()

    def load_other_parameter(self, para):
        if para is None:
            return
        for key, value in para.items():
            setattr(self, key, value)

    def __str__(self):
        """
        Model prints with number of trainable parameters
        """
        model_parameters = filter(lambda p: p.requires_grad, self.parameters())
        params = sum([np.prod(p.size()) for p in model_parameters])
        return (
            super().__str__()
            + set_color("\nTrainable parameters", "blue")
            + f": {params}"
        )


class GeneralRecommender(AbstractRecommender):
    """This is a abstract general recommender. All the general model should implement this class.
    The base general recommender class provide the basic dataset and parameters information.
    """

    type = ModelType.GENERAL

    def __init__(self, config, dataset):
        super(GeneralRecommender, self).__init__()

        # load dataset info
        self.USER_ID = config["USER_ID_FIELD"]
        self.ITEM_ID = config["ITEM_ID_FIELD"]
        self.NEG_ITEM_ID = config["NEG_PREFIX"] + self.ITEM_ID
        self.n_users = dataset.num(self.USER_ID)
        self.n_items = dataset.num(self.ITEM_ID)

        # load parameters info
        self.device = config["device"]


class AutoEncoderMixin(object):
    """This is a common part of auto-encoders. All the auto-encoder models should inherit this class,
    including CDAE, MacridVAE, MultiDAE, MultiVAE, RaCT and RecVAE.
    The base AutoEncoderMixin class provides basic dataset information and rating matrix function.
    """

    def build_histroy_items(self, dataset):
        self.history_item_id, self.history_item_value, _ = dataset.history_item_matrix()
        self.history_item_id = self.history_item_id.to(self.device)
        self.history_item_value = self.history_item_value.to(self.device)

    def get_rating_matrix(self, user):
        r"""Get a batch of user's feature with the user's id and history interaction matrix.

        Args:
            user (torch.LongTensor): The input tensor that contains user's id, shape: [batch_size, ]

        Returns:
            torch.FloatTensor: The user's feature of a batch of user, shape: [batch_size, n_items]
        """
        # Following lines construct tensor of shape [B,n_items] using the tensor of shape [B,H]
        col_indices = self.history_item_id[user].flatten()
        row_indices = torch.arange(user.shape[0]).repeat_interleave(
            self.history_item_id.shape[1], dim=0
        )
        rating_matrix = torch.zeros(1, device=self.device).repeat(
            user.shape[0], self.n_items
        )
        rating_matrix.index_put_(
            (row_indices, col_indices), self.history_item_value[user].flatten()
        )
        return rating_matrix


class SequentialRecommender(AbstractRecommender):
    """
    This is a abstract sequential recommender. All the sequential model should implement This class.
    """

    type = ModelType.SEQUENTIAL

    def __init__(self, config, dataset):
        super(SequentialRecommender, self).__init__()

        # load dataset info
        self.USER_ID = config["USER_ID_FIELD"]
        self.ITEM_ID = config["ITEM_ID_FIELD"]
        self.ITEM_SEQ = self.ITEM_ID + config["LIST_SUFFIX"]
        self.ITEM_SEQ_LEN = config["ITEM_LIST_LENGTH_FIELD"]
        self.POS_ITEM_ID = self.ITEM_ID
        self.NEG_ITEM_ID = config["NEG_PREFIX"] + self.ITEM_ID
        self.max_seq_length = config["MAX_ITEM_LIST_LENGTH"]
        self.n_items = dataset.num(self.ITEM_ID)

        # load parameters info
        self.device = config["device"]

    def gather_indexes(self, output, gather_index):
        """Gathers the vectors at the specific positions over a minibatch"""
        gather_index = gather_index.view(-1, 1, 1).expand(-1, -1, output.shape[-1])
        output_tensor = output.gather(dim=1, index=gather_index)
        return output_tensor.squeeze(1)

    def get_attention_mask(self, item_seq, bidirectional=False):
        """Generate left-to-right uni-directional or bidirectional attention mask for multi-head attention."""
        attention_mask = item_seq != 0
        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)  # torch.bool
        if not bidirectional:
            extended_attention_mask = torch.tril(
                extended_attention_mask.expand((-1, -1, item_seq.size(-1), -1))
            )
        extended_attention_mask = torch.where(extended_attention_mask, 0.0, -10000.0)
        return extended_attention_mask


class KnowledgeRecommender(AbstractRecommender):
    """This is a abstract knowledge-based recommender. All the knowledge-based model should implement this class.
    The base knowledge-based recommender class provide the basic dataset and parameters information.
    """

    type = ModelType.KNOWLEDGE

    def __init__(self, config, dataset):
        super(KnowledgeRecommender, self).__init__()

        # load dataset info
        self.USER_ID = config["USER_ID_FIELD"]
        self.ITEM_ID = config["ITEM_ID_FIELD"]
        self.NEG_ITEM_ID = config["NEG_PREFIX"] + self.ITEM_ID
        self.ENTITY_ID = config["ENTITY_ID_FIELD"]
        self.RELATION_ID = config["RELATION_ID_FIELD"]
        self.HEAD_ENTITY_ID = config["HEAD_ENTITY_ID_FIELD"]
        self.TAIL_ENTITY_ID = config["TAIL_ENTITY_ID_FIELD"]
        self.NEG_TAIL_ENTITY_ID = config["NEG_PREFIX"] + self.TAIL_ENTITY_ID
        self.n_users = dataset.num(self.USER_ID)
        self.n_items = dataset.num(self.ITEM_ID)
        self.n_entities = dataset.num(self.ENTITY_ID)
        self.n_relations = dataset.num(self.RELATION_ID)

        # load parameters info
        self.device = config["device"]


class ContextRecommender(AbstractRecommender):
    """This is a abstract context-aware recommender. All the context-aware model should implement this class.
    The base context-aware recommender class provide the basic embedding function of feature fields which also
    contains a first-order part of feature fields.
    """

    type = ModelType.CONTEXT
    input_type = InputType.POINTWISE

    def __init__(self, config, dataset):
        super(ContextRecommender, self).__init__()

        self.field_names = dataset.fields(
            source=[
                FeatureSource.INTERACTION,
                FeatureSource.USER,
                FeatureSource.USER_ID,
                FeatureSource.ITEM,
                FeatureSource.ITEM_ID,
            ]
        )
        self.LABEL = config["LABEL_FIELD"]
        self.embedding_size = config["embedding_size"]
        self.device = config["device"]
        self.double_tower = config["double_tower"]
        self.numerical_features = config["numerical_features"]
        if self.double_tower is None:
            self.double_tower = False
        self.token_field_names = []
        self.token_field_dims = []
        self.float_field_names = []
        self.float_field_dims = []
        self.token_seq_field_names = []
        self.token_seq_field_dims = []
        self.float_seq_field_names = []
        self.float_seq_field_dims = []
        self.num_feature_field = 0

        if self.double_tower:
            self.user_field_names = dataset.fields(
                source=[FeatureSource.USER, FeatureSource.USER_ID]
            )
            self.item_field_names = dataset.fields(
                source=[FeatureSource.ITEM, FeatureSource.ITEM_ID]
            )
            self.field_names = self.user_field_names + self.item_field_names
            self.user_token_field_num = 0
            self.user_float_field_num = 0
            self.user_token_seq_field_num = 0
            for field_name in self.user_field_names:
                if dataset.field2type[field_name] == FeatureType.TOKEN:
                    self.user_token_field_num += 1
                elif dataset.field2type[field_name] == FeatureType.TOKEN_SEQ:
                    self.user_token_seq_field_num += 1
                else:
                    self.user_float_field_num += 1
            self.item_token_field_num = 0
            self.item_float_field_num = 0
            self.item_token_seq_field_num = 0
            for field_name in self.item_field_names:
                if dataset.field2type[field_name] == FeatureType.TOKEN:
                    self.item_token_field_num += 1
                elif dataset.field2type[field_name] == FeatureType.TOKEN_SEQ:
                    self.item_token_seq_field_num += 1
                else:
                    self.item_float_field_num += 1

        for field_name in self.field_names:
            if field_name == self.LABEL:
                continue
            if dataset.field2type[field_name] == FeatureType.TOKEN:
                self.token_field_names.append(field_name)
                self.token_field_dims.append(dataset.num(field_name))
            elif dataset.field2type[field_name] == FeatureType.TOKEN_SEQ:
                self.token_seq_field_names.append(field_name)
                self.token_seq_field_dims.append(dataset.num(field_name))
            elif (
                dataset.field2type[field_name] == FeatureType.FLOAT
                and field_name in self.numerical_features
            ):
                self.float_field_names.append(field_name)
                self.float_field_dims.append(dataset.num(field_name))
            elif (
                dataset.field2type[field_name] == FeatureType.FLOAT_SEQ
                and field_name in self.numerical_features
            ):
                self.float_seq_field_names.append(field_name)
                self.float_seq_field_dims.append(dataset.num(field_name))
            else:
                continue

            self.num_feature_field += 1
        if len(self.token_field_dims) > 0:
            self.token_field_offsets = np.array(
                (0, *np.cumsum(self.token_field_dims)[:-1]), dtype=np.int64
            )
            self.token_embedding_table = FMEmbedding(
                self.token_field_dims, self.token_field_offsets, self.embedding_size
            )
        if len(self.float_field_dims) > 0:
            self.float_field_offsets = np.array(
                (0, *np.cumsum(self.float_field_dims)[:-1]), dtype=np.int64
            )
            self.float_embedding_table = FLEmbedding(
                self.float_field_dims, self.float_field_offsets, self.embedding_size
            )
        if len(self.token_seq_field_dims) > 0:
            self.token_seq_embedding_table = nn.ModuleList()
            for token_seq_field_dim in self.token_seq_field_dims:
                self.token_seq_embedding_table.append(
                    nn.Embedding(token_seq_field_dim, self.embedding_size)
                )
        if len(self.float_seq_field_dims) > 0:
            self.float_seq_embedding_table = nn.ModuleList()
            for float_seq_field_dim in self.float_seq_field_dims:
                self.float_seq_embedding_table.append(
                    nn.Embedding(float_seq_field_dim, self.embedding_size)
                )

        self.first_order_linear = FMFirstOrderLinear(config, dataset)

    def embed_float_fields(self, float_fields):
        """Embed the float feature columns

        Args:
            float_fields (torch.FloatTensor): The input dense tensor. shape of [batch_size, num_float_field]

        Returns:
            torch.FloatTensor: The result embedding tensor of float columns.
        """
        # input Tensor shape : [batch_size, num_float_field]
        if float_fields is None:
            return None
        # [batch_size, num_float_field, embed_dim]
        float_embedding = self.float_embedding_table(float_fields)

        return float_embedding

    def embed_float_seq_fields(self, float_seq_fields, mode="mean"):
        """Embed the float feature columns

        Args:
            float_seq_fields (torch.LongTensor): The input tensor. shape of [batch_size, seq_len]
            mode (str): How to aggregate the embedding of feature in this field. default=mean

        Returns:
            torch.FloatTensor: The result embedding tensor of token sequence columns.
        """
        # input is a list of Tensor shape of [batch_size, seq_len, 2]
        fields_result = []
        for i, float_seq_field in enumerate(float_seq_fields):
            embedding_table = self.float_seq_embedding_table[i]
            base, index = torch.split(float_seq_field, [1, 1], dim=-1)
            index = index.squeeze(-1)
            mask = index != 0  # [batch_size, seq_len]
            mask = mask.float()
            value_cnt = torch.sum(mask, dim=1, keepdim=True)  # [batch_size, 1]

            float_seq_embedding = base * embedding_table(
                index.long()
            )  # [batch_size, seq_len, embed_dim]

            mask = mask.unsqueeze(2).expand_as(
                float_seq_embedding
            )  # [batch_size, seq_len, embed_dim]
            if mode == "max":
                masked_float_seq_embedding = (
                    float_seq_embedding - (1 - mask) * 1e9
                )  # [batch_size, seq_len, embed_dim]
                result = torch.max(
                    masked_float_seq_embedding, dim=1, keepdim=True
                )  # [batch_size, 1, embed_dim]
            elif mode == "sum":
                masked_float_seq_embedding = float_seq_embedding * mask.float()
                result = torch.sum(
                    masked_float_seq_embedding, dim=1, keepdim=True
                )  # [batch_size, 1, embed_dim]
            else:
                masked_float_seq_embedding = float_seq_embedding * mask.float()
                result = torch.sum(
                    masked_float_seq_embedding, dim=1
                )  # [batch_size, embed_dim]
                eps = torch.FloatTensor([1e-8]).to(self.device)
                result = torch.div(result, value_cnt + eps)  # [batch_size, embed_dim]
                result = result.unsqueeze(1)  # [batch_size, 1, embed_dim]
            fields_result.append(result)
        if len(fields_result) == 0:
            return None
        else:
            return torch.cat(
                fields_result, dim=1
            )  # [batch_size, num_token_seq_field, embed_dim]

    def embed_token_fields(self, token_fields):
        """Embed the token feature columns

        Args:
            token_fields (torch.LongTensor): The input tensor. shape of [batch_size, num_token_field]

        Returns:
            torch.FloatTensor: The result embedding tensor of token columns.
        """
        # input Tensor shape : [batch_size, num_token_field]
        if token_fields is None:
            return None
        # [batch_size, num_token_field, embed_dim]
        token_embedding = self.token_embedding_table(token_fields)

        return token_embedding

    def embed_token_seq_fields(self, token_seq_fields, mode="mean"):
        """Embed the token feature columns

        Args:
            token_seq_fields (torch.LongTensor): The input tensor. shape of [batch_size, seq_len]
            mode (str): How to aggregate the embedding of feature in this field. default=mean

        Returns:
            torch.FloatTensor: The result embedding tensor of token sequence columns.
        """
        # input is a list of Tensor shape of [batch_size, seq_len]
        fields_result = []
        for i, token_seq_field in enumerate(token_seq_fields):
            embedding_table = self.token_seq_embedding_table[i]
            mask = token_seq_field != 0  # [batch_size, seq_len]
            mask = mask.float()
            value_cnt = torch.sum(mask, dim=1, keepdim=True)  # [batch_size, 1]

            token_seq_embedding = embedding_table(
                token_seq_field
            )  # [batch_size, seq_len, embed_dim]

            mask = mask.unsqueeze(2).expand_as(
                token_seq_embedding
            )  # [batch_size, seq_len, embed_dim]
            if mode == "max":
                masked_token_seq_embedding = (
                    token_seq_embedding - (1 - mask) * 1e9
                )  # [batch_size, seq_len, embed_dim]
                result = torch.max(
                    masked_token_seq_embedding, dim=1, keepdim=True
                )  # [batch_size, 1, embed_dim]
            elif mode == "sum":
                masked_token_seq_embedding = token_seq_embedding * mask.float()
                result = torch.sum(
                    masked_token_seq_embedding, dim=1, keepdim=True
                )  # [batch_size, 1, embed_dim]
            else:
                masked_token_seq_embedding = token_seq_embedding * mask.float()
                result = torch.sum(
                    masked_token_seq_embedding, dim=1
                )  # [batch_size, embed_dim]
                eps = torch.FloatTensor([1e-8]).to(self.device)
                result = torch.div(result, value_cnt + eps)  # [batch_size, embed_dim]
                result = result.unsqueeze(1)  # [batch_size, 1, embed_dim]
            fields_result.append(result)
        if len(fields_result) == 0:
            return None
        else:
            return torch.cat(
                fields_result, dim=1
            )  # [batch_size, num_token_seq_field, embed_dim]

    def double_tower_embed_input_fields(self, interaction):
        """Embed the whole feature columns in a double tower way.

        Args:
            interaction (Interaction): The input data collection.

        Returns:
            torch.FloatTensor: The embedding tensor of token sequence columns in the first part.
            torch.FloatTensor: The embedding tensor of float sequence columns in the first part.
            torch.FloatTensor: The embedding tensor of token sequence columns in the second part.
            torch.FloatTensor: The embedding tensor of float sequence columns in the second part.

        """
        if not self.double_tower:
            raise RuntimeError(
                "Please check your model hyper parameters and set 'double tower' as True"
            )
        sparse_embedding, dense_embedding = self.embed_input_fields(interaction)
        if dense_embedding is not None:
            first_dense_embedding, second_dense_embedding = torch.split(
                dense_embedding,
                [self.user_float_field_num, self.item_float_field_num],
                dim=1,
            )
        else:
            first_dense_embedding, second_dense_embedding = None, None

        if sparse_embedding is not None:
            sizes = [
                self.user_token_seq_field_num,
                self.item_token_seq_field_num,
                self.user_token_field_num,
                self.item_token_field_num,
            ]
            (
                first_token_seq_embedding,
                second_token_seq_embedding,
                first_token_embedding,
                second_token_embedding,
            ) = torch.split(sparse_embedding, sizes, dim=1)
            first_sparse_embedding = torch.cat(
                [first_token_seq_embedding, first_token_embedding], dim=1
            )
            second_sparse_embedding = torch.cat(
                [second_token_seq_embedding, second_token_embedding], dim=1
            )
        else:
            first_sparse_embedding, second_sparse_embedding = None, None

        return (
            first_sparse_embedding,
            first_dense_embedding,
            second_sparse_embedding,
            second_dense_embedding,
        )

    def concat_embed_input_fields(self, interaction):
        sparse_embedding, dense_embedding = self.embed_input_fields(interaction)
        all_embeddings = []
        if sparse_embedding is not None:
            all_embeddings.append(sparse_embedding)
        if dense_embedding is not None and len(dense_embedding.shape) == 3:
            all_embeddings.append(dense_embedding)
        return torch.cat(all_embeddings, dim=1)  # [batch_size, num_field, embed_dim]

    def embed_input_fields(self, interaction):
        """Embed the whole feature columns.

        Args:
            interaction (Interaction): The input data collection.

        Returns:
            torch.FloatTensor: The embedding tensor of token sequence columns.
            torch.FloatTensor: The embedding tensor of float sequence columns.
        """
        float_fields = []
        for field_name in self.float_field_names:
            if len(interaction[field_name].shape) == 3:
                float_fields.append(interaction[field_name])
            else:
                float_fields.append(interaction[field_name].unsqueeze(1))
        if len(float_fields) > 0:
            float_fields = torch.cat(
                float_fields, dim=1
            )  # [batch_size, num_float_field, 2]
        else:
            float_fields = None
        # [batch_size, num_float_field] or [batch_size, num_float_field, embed_dim] or None
        float_fields_embedding = self.embed_float_fields(float_fields)

        float_seq_fields = []
        for field_name in self.float_seq_field_names:
            float_seq_fields.append(interaction[field_name])

        float_seq_fields_embedding = self.embed_float_seq_fields(float_seq_fields)

        if float_fields_embedding is None:
            dense_embedding = float_seq_fields_embedding
        else:
            if float_seq_fields_embedding is None:
                dense_embedding = float_fields_embedding
            else:
                dense_embedding = torch.cat(
                    [float_seq_fields_embedding, float_fields_embedding], dim=1
                )

        token_fields = []
        for field_name in self.token_field_names:
            token_fields.append(interaction[field_name].unsqueeze(1))
        if len(token_fields) > 0:
            token_fields = torch.cat(
                token_fields, dim=1
            )  # [batch_size, num_token_field, 2]
        else:
            token_fields = None
        # [batch_size, num_token_field, embed_dim] or None
        token_fields_embedding = self.embed_token_fields(token_fields)

        token_seq_fields = []
        for field_name in self.token_seq_field_names:
            token_seq_fields.append(interaction[field_name])
        # [batch_size, num_token_seq_field, embed_dim] or None
        token_seq_fields_embedding = self.embed_token_seq_fields(token_seq_fields)

        if token_fields_embedding is None:
            sparse_embedding = token_seq_fields_embedding
        else:
            if token_seq_fields_embedding is None:
                sparse_embedding = token_fields_embedding
            else:
                sparse_embedding = torch.cat(
                    [token_seq_fields_embedding, token_fields_embedding], dim=1
                )

        # sparse_embedding shape: [batch_size, num_token_seq_field+num_token_field, embed_dim] or None
        # dense_embedding shape: [batch_size, num_float_field, 2] or [batch_size, num_float_field, embed_dim] or None
        return sparse_embedding, dense_embedding


class ContextRecommenderWithEmbeddings(ContextRecommender):
    """This is a context-aware recommender that supports pre-computed embeddings.

    This class extends ContextRecommender to allow passing pre-computed embedding vectors
    (e.g., from text encoders) directly as input features without re-embedding them.

    All existing context recommender models (EulerNet, KD_DAGFM, DeepFM, etc.) can inherit
    from this class instead of ContextRecommender to gain support for pre-computed embeddings.

    To use pre-computed embeddings:
    1. Specify field names in config['precomputed_embedding_fields']
    2. Pass embeddings with shape [batch_size, embedding_dim] or [batch_size, num_fields, embedding_dim]
    3. Add embeddings to interaction: interaction['field_name'] = embeddings_tensor
    """

    def __init__(self, config, dataset):
        super(ContextRecommenderWithEmbeddings, self).__init__(config, dataset)

        # Get list of fields that contain pre-computed embeddings (e.g., from text encoders)
        self.precomputed_embedding_fields = config.get('precomputed_embedding_fields', [])

        # For double tower mode, separate user and item pre-computed embeddings
        if self.double_tower:
            self.precomputed_embedding_fields_user = config.get('precomputed_embedding_fields_user', [])
            self.precomputed_embedding_fields_item = config.get('precomputed_embedding_fields_item', [])
            # If not specified separately, treat all as item embeddings (common case)
            if not self.precomputed_embedding_fields_user and not self.precomputed_embedding_fields_item:
                self.precomputed_embedding_fields_item = self.precomputed_embedding_fields.copy()
                self.precomputed_embedding_fields_user = []
        else:
            self.precomputed_embedding_fields_user = []
            self.precomputed_embedding_fields_item = []

    def embed_input_fields(self, interaction):
        """Embed the whole feature columns, handling pre-computed embeddings.

        This method overrides the parent class to support pre-computed embeddings.
        If a field is in precomputed_embedding_fields and has the right shape,
        it will be used directly without re-embedding.

        Args:
            interaction (Interaction): The input data collection.

        Returns:
            torch.FloatTensor: The embedding tensor of token sequence columns.
            torch.FloatTensor: The embedding tensor of float sequence columns (including pre-computed).
        """
        # Handle pre-computed embeddings first
        precomputed_embeddings = []

        for field_name in self.precomputed_embedding_fields:
            if field_name in interaction:
                embedding = interaction[field_name]
                # Check if it's already an embedding vector
                if len(embedding.shape) == 2:
                    # Shape: [batch_size, embedding_dim]
                    # Add dimension to match expected format: [batch_size, 1, embedding_dim]
                    embedding = embedding.unsqueeze(1)
                    precomputed_embeddings.append(embedding)
                elif len(embedding.shape) == 3:
                    # Shape: [batch_size, num_fields, embedding_dim]
                    precomputed_embeddings.append(embedding)
                else:
                    self.logger.warning(
                        f"Field {field_name} has unexpected shape {embedding.shape}. "
                        f"Expected [batch_size, embedding_dim] or [batch_size, num_fields, embedding_dim]. "
                        f"Skipping pre-computed embedding for this field."
                    )

        # Get regular embeddings from parent class
        # Temporarily filter out pre-computed fields to avoid double processing
        original_float_field_names = self.float_field_names.copy()
        self.float_field_names = [
            f for f in self.float_field_names
            if f not in self.precomputed_embedding_fields
        ]

        # Get regular embeddings
        sparse_embedding, dense_embedding = super().embed_input_fields(interaction)

        # Restore original float_field_names
        self.float_field_names = original_float_field_names

        # Combine pre-computed embeddings with regular dense embeddings
        if len(precomputed_embeddings) > 0:
            precomputed_tensor = torch.cat(precomputed_embeddings, dim=1)  # [batch_size, num_precomputed_fields, embed_dim]

            if dense_embedding is None:
                dense_embedding = precomputed_tensor
            else:
                # Ensure dense_embedding has the right shape
                # It should be [batch_size, num_float_field, embed_dim] from parent class
                if len(dense_embedding.shape) == 2:
                    # This shouldn't happen, but handle edge case
                    # If shape is [batch_size, num_float_field, 2], we need to embed it
                    if dense_embedding.shape[-1] == 2:
                        dense_embedding = self.embed_float_fields(dense_embedding)
                    else:
                        # Unexpected shape, add dimension
                        dense_embedding = dense_embedding.unsqueeze(1)

                # Concatenate: [batch_size, num_regular_fields + num_precomputed_fields, embed_dim]
                dense_embedding = torch.cat([dense_embedding, precomputed_tensor], dim=1)

        return sparse_embedding, dense_embedding

    def double_tower_embed_input_fields(self, interaction):
        """Embed the whole feature columns in a double tower way, handling pre-computed embeddings.

        This method overrides the parent class to properly handle pre-computed embeddings
        in double tower mode by separating user and item embeddings.

        Args:
            interaction (Interaction): The input data collection.

        Returns:
            torch.FloatTensor: The embedding tensor of token sequence columns in the first part.
            torch.FloatTensor: The embedding tensor of float sequence columns in the first part.
            torch.FloatTensor: The embedding tensor of token sequence columns in the second part.
            torch.FloatTensor: The embedding tensor of float sequence columns in the second part.
        """
        if not self.double_tower:
            raise RuntimeError(
                "Please check your model hyper parameters and set 'double tower' as True"
            )

        # Handle pre-computed embeddings separately for user and item
        precomputed_user_embeddings = []
        precomputed_item_embeddings = []

        for field_name in self.precomputed_embedding_fields_user:
            if field_name in interaction:
                embedding = interaction[field_name]
                if len(embedding.shape) == 2:
                    embedding = embedding.unsqueeze(1)
                precomputed_user_embeddings.append(embedding)

        for field_name in self.precomputed_embedding_fields_item:
            if field_name in interaction:
                embedding = interaction[field_name]
                if len(embedding.shape) == 2:
                    embedding = embedding.unsqueeze(1)
                precomputed_item_embeddings.append(embedding)

        # Temporarily filter out pre-computed fields to avoid double processing
        original_float_field_names = self.float_field_names.copy()
        all_precomputed_fields = self.precomputed_embedding_fields_user + self.precomputed_embedding_fields_item
        self.float_field_names = [
            f for f in self.float_field_names
            if f not in all_precomputed_fields
        ]

        # Get regular embeddings from parent class
        sparse_embedding, dense_embedding = super().embed_input_fields(interaction)

        # Restore original float_field_names
        self.float_field_names = original_float_field_names

        # Combine pre-computed embeddings with regular dense embeddings
        # For double tower, we need to insert user embeddings before item embeddings
        if len(precomputed_user_embeddings) > 0 or len(precomputed_item_embeddings) > 0:
            user_dense_parts = []
            item_dense_parts = []

            # Split regular dense embedding into user and item parts
            if dense_embedding is not None:
                if len(dense_embedding.shape) == 2 and dense_embedding.shape[-1] == 2:
                    dense_embedding = self.embed_float_fields(dense_embedding)
                elif len(dense_embedding.shape) == 2:
                    dense_embedding = dense_embedding.unsqueeze(1)

                # Split regular embeddings
                if dense_embedding.shape[1] > 0:
                    regular_user, regular_item = torch.split(
                        dense_embedding,
                        [self.user_float_field_num, self.item_float_field_num],
                        dim=1,
                    )
                    if regular_user.shape[1] > 0:
                        user_dense_parts.append(regular_user)
                    if regular_item.shape[1] > 0:
                        item_dense_parts.append(regular_item)

            # Add pre-computed user embeddings
            if len(precomputed_user_embeddings) > 0:
                user_dense_parts.append(torch.cat(precomputed_user_embeddings, dim=1))

            # Add pre-computed item embeddings
            if len(precomputed_item_embeddings) > 0:
                item_dense_parts.append(torch.cat(precomputed_item_embeddings, dim=1))

            # Combine parts
            if len(user_dense_parts) > 0:
                first_dense_embedding = torch.cat(user_dense_parts, dim=1)
            else:
                first_dense_embedding = None

            if len(item_dense_parts) > 0:
                second_dense_embedding = torch.cat(item_dense_parts, dim=1)
            else:
                second_dense_embedding = None
        else:
            # No pre-computed embeddings, use parent's split
            if dense_embedding is not None:
                if len(dense_embedding.shape) == 2 and dense_embedding.shape[-1] == 2:
                    dense_embedding = self.embed_float_fields(dense_embedding)
                elif len(dense_embedding.shape) == 2:
                    dense_embedding = dense_embedding.unsqueeze(1)

                first_dense_embedding, second_dense_embedding = torch.split(
                    dense_embedding,
                    [self.user_float_field_num, self.item_float_field_num],
                    dim=1,
                )
            else:
                first_dense_embedding, second_dense_embedding = None, None

        # Handle sparse embeddings (same as parent)
        if sparse_embedding is not None:
            sizes = [
                self.user_token_seq_field_num,
                self.item_token_seq_field_num,
                self.user_token_field_num,
                self.item_token_field_num,
            ]
            (
                first_token_seq_embedding,
                second_token_seq_embedding,
                first_token_embedding,
                second_token_embedding,
            ) = torch.split(sparse_embedding, sizes, dim=1)
            first_sparse_embedding = torch.cat(
                [first_token_seq_embedding, first_token_embedding], dim=1
            )
            second_sparse_embedding = torch.cat(
                [second_token_seq_embedding, second_token_embedding], dim=1
            )
        else:
            first_sparse_embedding, second_sparse_embedding = None, None

        return (
            first_sparse_embedding,
            first_dense_embedding,
            second_sparse_embedding,
            second_dense_embedding,
        )
