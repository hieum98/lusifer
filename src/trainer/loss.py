import einops
import torch
import torch.nn as nn
from pytorch_metric_learning import losses, miners, distances


class AllGather(torch.autograd.Function):
    """
    all_gather with gradient back-propagation
    """

    @staticmethod
    def forward(ctx, tensor_list, tensor, group, async_op):
        torch.distributed.all_gather(
            tensor_list, tensor, group=group, async_op=async_op
        )
        return tuple(tensor_list)

    @staticmethod
    def backward(ctx, *grad_list):
        grad_list = list(grad_list)
        rank = torch.distributed.get_rank()

        dist_ops = [
            torch.distributed.reduce(grad_list[i], i, async_op=True)
            for i in range(torch.distributed.get_world_size())
        ]

        for op in dist_ops:
            op.wait()

        return None, grad_list[rank], None, None


all_gather_with_grad = AllGather.apply

def mismatched_sizes_all_gather(
    tensor: torch.Tensor, group=None, async_op=False, mismatched_axis=0
):
    # all_gather doesn't support tensor lists where the first dimension is mismatched. This does.
    assert torch.distributed.is_initialized(), "torch.distributed not initialized"
    world_size = torch.distributed.get_world_size()
    # let's get the sizes for everyone
    mismatched_sizes = torch.tensor(
        [tensor.shape[mismatched_axis]], dtype=torch.int64, device="cuda"
    )
    sizes = [torch.zeros_like(mismatched_sizes) for _ in range(world_size)]
    torch.distributed.all_gather(
        sizes, mismatched_sizes, group=group, async_op=async_op
    )
    sizes = torch.cat(sizes).cpu().tolist()
    # now pad to the max dim-0 size
    max_size = max(sizes)
    padded = torch.zeros(
        (
            *tensor.shape[:mismatched_axis],
            max_size,
            *tensor.shape[mismatched_axis + 1 :],
        ),
        device=tensor.device,
        dtype=tensor.dtype,
    )
    # selects the place where we're adding information
    padded_to_fill = padded.narrow(mismatched_axis, 0, tensor.shape[mismatched_axis])
    padded_to_fill[...] = tensor
    # gather the padded tensors
    tensor_list = [
        torch.zeros(padded.shape, device=padded.device, dtype=padded.dtype)
        for _ in range(world_size)
    ]
    all_gather_with_grad(tensor_list, padded, group, async_op)
    # trim off the padding
    for rank in range(world_size):
        # checks that the rest is 0
        assert (
            not tensor_list[rank]
            .narrow(
                mismatched_axis,
                sizes[rank],
                padded.shape[mismatched_axis] - sizes[rank],
            )
            .count_nonzero()
            .is_nonzero()
        ), "This would remove non-padding information"
        tensor_list[rank] = tensor_list[rank].narrow(mismatched_axis, 0, sizes[rank])
    return tensor_list


class ContrastiveLoss:
    def __init__(
            self,
            loss_type: str = 'NTXentLoss', # 'NTXentLoss' or 'SupConLoss'
            temperature: float = 0.05,
            normalize: bool = True,
            use_miner: bool = False,
            cross_batch_loss: bool = True,
            ) -> None:
        self.cross_batch_loss = cross_batch_loss

        distance = distances.CosineSimilarity() if normalize else distances.LpDistance(normalize_embeddings=False)
        if loss_type == 'NTXentLoss':
            self.loss_fn = losses.NTXentLoss(temperature=temperature, distance=distance)
        elif loss_type == 'SupConLoss':
            self.loss_fn = losses.SupConLoss(temperature=temperature, base_distance=distance)
        else:
            raise ValueError(f"Unsupported loss type: {loss_type}")

        if use_miner:
            self.miner = miners.PairMarginMiner(
                pos_margin = 1.0 if normalize else 0.0, # under 1.0 for cosine similarity or over 0.0 for LpDistance will be considered as hard positive 
                neg_margin = 0.5, # over 0.5 for cosine similarity or under 0.5 for LpDistance will be considered as hard negative.
                distance=distance
                )
        else:
            self.miner = None
    
    def __call__(
            self, 
            q_embeds: torch.Tensor, # (batch_size, embed_dim) 
            q_labels: torch.Tensor, # (batch_size,)
            pos_embeds: torch.Tensor, # (batch_size, num_pos, embed_dim)
            neg_embeds: torch.Tensor, # (batch_size, num_neg, embed_dim)
            ):
        pos_labels = einops.repeat(q_labels, 'b -> b n', n=pos_embeds.size(1))
        pos_labels = einops.rearrange(pos_labels, 'b n -> (b n)')

        pos_embeds = einops.rearrange(pos_embeds, 'b n d -> (b n) d') # (batch_size * num_pos, embed_dim)
        neg_embeds = einops.rearrange(neg_embeds, 'b n d -> (b n) d') # (batch_size * num_neg, embed_dim)
        if torch.distributed.is_initialized() and self.cross_batch_loss:
            full_q_labels = mismatched_sizes_all_gather(q_labels)
            full_pos_labels = mismatched_sizes_all_gather(pos_labels)
            full_q_embeds = mismatched_sizes_all_gather(q_embeds) 
            full_pos_embeds = mismatched_sizes_all_gather(pos_embeds)
            full_neg_embeds = mismatched_sizes_all_gather(neg_embeds)

            full_q_labels = torch.cat(full_q_labels, dim=0) # (world_size * batch_size,)
            full_pos_labels = torch.cat(full_pos_labels, dim=0) # (num_all_pos,)
            full_q_embeds = torch.cat(full_q_embeds, dim=0) # (world_size * batch_size, embed_dim)
            full_pos_embeds = torch.cat(full_pos_embeds, dim=0) # (num_all_pos, embed_dim)
            full_neg_embeds = torch.cat(full_neg_embeds, dim=0) # (num_all_neg, embed_dim)
            max_idx = torch.max(full_q_labels)
            full_neg_labels = torch.arange(full_neg_embeds.size(0), device=full_neg_embeds.device) + max_idx + 1 # (num_all_neg,)
            
            full_labels = torch.cat([full_q_labels, full_pos_labels, full_neg_labels], dim=0)
            full_embeds = torch.cat([full_q_embeds, full_pos_embeds, full_neg_embeds], dim=0)
        else:
            max_idx = torch.max(q_labels)
            neg_labels = torch.arange(neg_embeds.size(0), device=neg_embeds.device) + max_idx + 1 # (num_neg,)
            
            full_labels = torch.cat([q_labels, pos_labels, neg_labels], dim=0)
            full_embeds = torch.cat([q_embeds, pos_embeds, neg_embeds], dim=0)
        
        if self.miner is not None:
            hard_pairs = self.miner(full_embeds, full_labels)
            loss = self.loss_fn(full_embeds, full_labels, hard_pairs)
        else:
            loss = self.loss_fn(full_embeds, full_labels)
        
        return loss


