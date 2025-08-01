import torch
import torch.nn as nn


class WeightedImputer(nn.Module):
    """
    v_{p,t} = Σ_m  w_m  ·  mean_{i∈N_p,t^m} u_{i,t}
    One scalar weight per metadata type (author, venue, reference, …).
    """
    def __init__(self, meta_types, hidden_dim): # meta_types: list of metadata types (e.g. ['author', 'venue'])
        super().__init__()
        self.hidden_dim = hidden_dim
        self.w = nn.ParameterDict({
            m: nn.Parameter(torch.tensor(1.0)) for m in meta_types
        })
        self.w['self'] = nn.Parameter(torch.tensor(1.0))
        self.author_attention = nn.Sequential(
            nn.Linear(2* hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Softmax(dim=0)
        )
        
    @staticmethod
    def collect_neighbours(data, paper_id: int, device):
        """
        Collect neighbour indices *in the publication year* snapshot.

        Returns a dict
            { 'author': LongTensor,        # authors of the paper (in publication order)
              'venue' : LongTensor,        # venue of the paper
              'paper' : LongTensor }       # references (= cited papers)
        """
        neighbours = {}

        # 1) authors --------------------------------------------------------
        # Use the stored author order information if available
        if hasattr(data['paper'], 'author_order') and paper_id in data['paper'].author_order:
            # Get authors in their original publication order
            author_order_list = data['paper'].author_order[paper_id]
            # Sort by order (second element of tuple) and extract author indices
            sorted_authors = sorted(author_order_list, key=lambda x: x[1])
            author_indices = torch.tensor([author_idx for author_idx, _ in sorted_authors], 
                                        dtype=torch.long, device=device)
            neighbours['author'] = author_indices
        else:
            # Fallback to edge index method (original behavior)
            src, dst = data['author', 'writes', 'paper'].edge_index.to(device)
            mask = (dst == paper_id).nonzero(as_tuple=False).view(-1)
            if mask.numel():
                neighbours['author'] = src.index_select(0, mask)

        # 2) venue ----------------------------------------------------------
        src, dst = data['paper', 'published_in', 'venue'].edge_index.to(device)
        mask = (src == paper_id).nonzero(as_tuple=False).view(-1)
        if mask.numel():
            neighbours['venue'] = dst.index_select(0, mask)

        # 3) references (citations) ----------------------------------------
        src, dst = data['paper', 'cites', 'paper'].edge_index.to(device)
        mask = (src == paper_id).nonzero(as_tuple=False).view(-1)
        if mask.numel():
            neighbours['paper'] = dst.index_select(0, mask)
        return neighbours

    def aggregate_authors_with_attention(self, author_embeddings):
        """
        Apply attention mechanism to aggregate author embeddings.
        
        Args:
            author_embeddings: Tensor [num_authors, hidden_dim]
            
        Returns:
            Tensor [hidden_dim] - attention-weighted author representation
        """
        if author_embeddings.size(0) == 1:
            # Single author, no need for attention
            return author_embeddings.squeeze(0)
        
        # Compute attention weights for each author
        attention_weights = self.author_attention(author_embeddings) # [num_authors, hidden_dim]
        
        # Apply attention weights and sum
        weighted_authors = (author_embeddings * attention_weights).sum(dim=0)  # [hidden_dim]
        
        return weighted_authors

    def forward(
        self,
        paper_id: int | None,
        year_idx: int,
        snapshots,
        embeddings,
        predefined_neigh: dict[str, torch.Tensor] | None = None,
        topic_vec=None,
        drop_social: bool = False,
        mean_social: torch.Tensor = None,
        drop_authors: bool = False,
        drop_author_topic: bool = False,
        mean_author_topic: torch.Tensor = None,
        mean_author_social: torch.Tensor = None
    ):
        """
        Args
        ----
        paper_id          : index of paper *in its publication year* snapshot.
                            Ignored when `predefined_neigh` is given.
        year_idx          : index of the snapshot we want to impute for
                            (t, t-2, …).
        snapshots         : list[HeteroData]
        embeddings        : list[dict] – output of the encoder for every year
        predefined_neigh  : optional neighbour dict produced by
                            `collect_neighbours`.  Needed because the paper
                            itself is not present in earlier graphs.
        topic_vec         : topic embedding for the paper
        drop_social       : if True, use mean_social for the social part of author embedding
        mean_social       : mean social embedding to use for ablation
        drop_authors      : if True, use mean_author_topic and mean_author_social for all authors
        drop_author_topic : if True, use mean_author_topic but keep original author_social
        mean_author_topic : mean author topic embedding to use for ablation
        mean_author_social: mean author social embedding to use for ablation

        Returns
        -------
        Tensor [2*hidden_dim] – imputed embedding v_{p, year_idx}
        """
        # data['paper'].y_year can get the publication year of the paper, we only need the papers published in year t.
        data = snapshots[year_idx]
        embs = embeddings[year_idx]
        device = embs['paper'].device

        # decide which neighbour set to use -------------------------------
        if predefined_neigh is not None:
            neighbours = predefined_neigh
        else:
            raise ValueError(
                "predefined_neigh must be provided for imputation in earlier years"
            )

        # ----- nothing to aggregate --------------------------------------
        if not neighbours:
            return torch.zeros(embs['paper'].size(-1) * 2, device=device)  # output is 2*hidden_dim

        # ----- weighted average  -----------------------------------------
        parts = []
        weights = []
        for ntype, ids in neighbours.items():
            # some neighbour ids may not exist yet in an earlier snapshot
            ids = ids[ids < embs[ntype].size(0)]
            if ids.numel() == 0:
                continue
            if ntype == 'author':
                if drop_authors and mean_author_topic is not None:
                    # Use mean embeddings for all authors
                    author_topic = mean_author_topic.expand(ids.size(0), -1)
                    if mean_author_social is not None:
                        author_social = mean_author_social.expand(ids.size(0), -1)
                    elif 'author_social' in embs:
                        # Fallback: use zeros if social embeddings exist but mean wasn't computed
                        author_social = torch.zeros(ids.size(0), embs['author_social'].size(-1), device=embs['author_social'].device)
                    else:
                        # No social embeddings in dataset
                        author_social = torch.zeros(ids.size(0), mean_author_topic.size(-1), device=mean_author_topic.device)
                # This condition was causing full model to use mean embeddings - removed to fix train/test mismatch
                elif drop_author_topic and mean_author_topic is not None:
                    # Use mean topic but keep original social
                    author_topic = mean_author_topic.expand(ids.size(0), -1)
                    if drop_social and mean_social is not None:
                        # Also drop social if both flags are set
                        author_social = mean_social.expand(ids.size(0), -1)
                    else:
                        if 'author_social' in embs:
                            author_social = embs['author_social'][ids]   # [num_authors, hidden_dim]
                        else:
                            # No social embeddings in dataset
                            author_social = torch.zeros(ids.size(0), mean_author_topic.size(-1), device=mean_author_topic.device)
                else:
                    author_topic = embs['author'][ids]           # [num_authors, hidden_dim]
                    if drop_social and mean_social is not None:
                        # Use mean_social for all authors
                        author_social = mean_social.expand(author_topic.size(0), -1)
                    else:
                        if 'author_social' in embs:
                            author_social = embs['author_social'][ids]   # [num_authors, hidden_dim]
                        else:
                            # No social embeddings in dataset  
                            author_social = torch.zeros(author_topic.size(0), author_topic.size(-1), device=author_topic.device)
                author_embeddings = torch.cat([author_topic, author_social], dim=1)  # [num_authors, 2*hidden_dim]
                # Use attention mechanism for authors
                aggregated_authors = self.aggregate_authors_with_attention(author_embeddings)
                parts.append(aggregated_authors)
                weights.append(self.w[ntype])
            else:
                # For other types, expand to match output dim
                emb = embs[ntype][ids].mean(dim=0)
                emb = torch.cat([emb, torch.zeros_like(emb)], dim=0)  # [2*hidden_dim]
                parts.append(emb)
                weights.append(self.w[ntype])

        # --- add the paper's own embedding --------------------------------
        if topic_vec is not None:
            # Expand topic_vec to 2*hidden_dim by concatenating zeros for social part
            topic_vec_expanded = torch.cat([topic_vec, torch.zeros_like(topic_vec)], dim=0)
            parts.append(topic_vec_expanded)
            weights.append(self.w['self'])
        
        if len(parts) == 0:
            return torch.zeros(embs['paper'].size(-1) * 2, device=device)

        # Normalize weights
        weights = torch.stack(weights)
        weights = torch.softmax(weights, dim=0)  # Normalize weights to sum to 1
        
        # Apply normalized weights
        weighted_parts = [w * p for w, p in zip(weights, parts)]
        return torch.stack(weighted_parts, dim=0).sum(dim=0)