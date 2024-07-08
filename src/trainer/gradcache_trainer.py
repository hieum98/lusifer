from collections import UserDict
import pathlib
from contextlib import nullcontext
import time
from typing import Any, Callable, Dict, List, Optional
import typing
import einops
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.checkpoint import get_device_states, set_device_states
import lightning as L

from src.models.lusifer import Lusifer
from src.trainer.loss import ContrastiveLoss
from src.trainer.utils import split_input


class RandContext:
    def __init__(self, *tensors):
        self.fwd_cpu_state = torch.get_rng_state()
        self.fwd_gpu_devices, self.fwd_gpu_states = get_device_states(*tensors)

    def __enter__(self):
        self._fork = torch.random.fork_rng(devices=self.fwd_gpu_devices, enabled=True)
        self._fork.__enter__()
        torch.set_rng_state(self.fwd_cpu_state)
        set_device_states(self.fwd_gpu_devices, self.fwd_gpu_states)

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._fork.__exit__(exc_type, exc_val, exc_tb)
        self._fork = None


class GradCacheTrainer:
    def __init__(
            self,
            fabric: L.Fabric,
            loss_type: str = 'NTXentLoss',
            temperature: float = 0.05,
            normalize: bool = True,
            use_miner: bool = False,
            cross_batch_loss: bool = True,
            chunk_size: Optional[int] = 1,
            ) -> None:
        self.fabric = fabric
        self.chunk_size = chunk_size
        self.cross_batch_loss = cross_batch_loss

        self.loss_fn = ContrastiveLoss(
            loss_type=loss_type,
            temperature=temperature,
            normalize=normalize,
            use_miner=use_miner,
            cross_batch_loss=cross_batch_loss,
        )

    def get_input_tensors(self, model_input) -> List[torch.Tensor]:
        """
        Recursively go through model input and grab all tensors, which are then used to record current device random
        states. This method will do its best to parse types of Tensor, tuple, list, dict and UserDict. Other types will
        be ignored unless self._get_input_tensors_strict is set to True, in which case an exception will be raised.
        :param model_input: input to model
        :return: all torch tensors in model_input
        """
        if isinstance(model_input, torch.Tensor):
            return [model_input]
        elif isinstance(model_input, (list, tuple)):
            return sum((self.get_input_tensors(x) for x in model_input), [])
        elif isinstance(model_input, (dict, UserDict)):
            return sum((self.get_input_tensors(x) for x in model_input.values()), [])
        else:
            return []
    
    def forward_no_grad(
            self, 
            model: Lusifer, 
            model_inputs: Dict[str, torch.Tensor],
            ):
        """
        Forward pass through the model, but without gradients. This is useful for caching forward passes for gradient
        accumulation.
        :param model: model to forward pass through
        :param model_inputs: inputs to the model
        :return: query_projections, pos_projections, neg_projections, rnd_state
        """
        with torch.no_grad():
            rnd_state = RandContext(*self.get_input_tensors(model_inputs))
            
            query_input_ids = model_inputs['query_input_ids'] # (batch_size, seq_len)
            query_attention_mask = model_inputs['query_attention_mask']
            query_prompt_length = model_inputs['query_prompt_length'] # (batch_size,)

            pos_input_ids = model_inputs['pos_input_ids'] # (batch_size, num_pos, seq_len)
            pos_attention_mask = model_inputs['pos_attention_mask'] 
            pos_prompt_length = model_inputs['pos_prompt_length'] # (batch_size, num_pos)

            neg_input_ids = model_inputs['neg_input_ids'] # (batch_size, num_neg, seq_len)
            neg_attention_mask = model_inputs['neg_attention_mask']
            neg_prompt_length = model_inputs['neg_prompt_length'] # (batch_size, num_neg)

            B, P, _ = pos_input_ids.size()
            B, N, _ = neg_input_ids.size()

            query_projections = model(
                input_ids=query_input_ids,
                attention_mask=query_attention_mask,
                prompt_length=query_prompt_length,
            )['projection'] # (batch_size, embed_dim)

            pos_projections = model(
                input_ids=einops.rearrange(pos_input_ids, 'b n l -> (b n) l'),
                attention_mask=einops.rearrange(pos_attention_mask, 'b n l -> (b n) l'),
                prompt_length=einops.rearrange(pos_prompt_length, 'b n -> (b n)'),
            )['projection'] # (batch_size * num_pos, embed_dim)
            pos_projections = einops.rearrange(pos_projections, '(b n) d -> b n d', b=B, n=P)

            neg_projections = model(
                input_ids=einops.rearrange(neg_input_ids, 'b n l -> (b n) l'),
                attention_mask=einops.rearrange(neg_attention_mask, 'b n l -> (b n) l'),
                prompt_length=einops.rearrange(neg_prompt_length, 'b n -> (b n)'),
            )['projection'] # (batch_size * num_neg, embed_dim)
            neg_projections = einops.rearrange(neg_projections, '(b n) d -> b n d', b=B, n=N)
        
        return query_projections, pos_projections, neg_projections, rnd_state
    
    def compute_cons_loss_from_reps(
            self,
            query_projections: torch.Tensor, # (batch_size, embed_dim)
            pos_projections: torch.Tensor, # (batch_size, num_pos, embed_dim)
            neg_projections: torch.Tensor, # (batch_size, num_neg, embed_dim)
            query_labels: torch.Tensor, # (batch_size,)
            ) -> torch.Tensor:
        """
        Compute contrastive loss from representations.
        :param query_projections: query projections
        :param pos_projections: positive projections
        :param neg_projections: negative projections
        :param query_labels: query labels
        :return: contrastive loss value
        """
        con_loss = self.loss_fn(
            q_embeds=query_projections,
            q_labels=query_labels,
            pos_embeds=pos_projections,
            neg_embeds=neg_projections,
        )
        return con_loss
    
    @typing.no_type_check
    def build_cache(
            self,
            query_projections: torch.Tensor, # (batch_size, embed_dim)
            pos_projections: torch.Tensor, # (batch_size, num_pos, embed_dim)
            neg_projections: torch.Tensor, # (batch_size, num_neg, embed_dim)
            query_labels: torch.Tensor, # (batch_size,)
            ):
        """
        Build cache for gradient computation.
        :param query_projections: query projections
        :param pos_projections: positive projections
        :param neg_projections: negative projections
        :param query_labels: query labels
        :return: cache: gradient cache, con_loss: contrastive loss value
        """
        query_projections = query_projections.detach().requires_grad_(True)
        pos_projections = pos_projections.detach().requires_grad_(True)
        neg_projections = neg_projections.detach().requires_grad_(True)

        with nullcontext():
            with self.fabric.autocast():
                con_loss = self.compute_cons_loss_from_reps(
                    query_projections=query_projections,
                    pos_projections=pos_projections,
                    neg_projections=neg_projections,
                    query_labels=query_labels,
                )
        self.fabric.backward(con_loss)
        query_cache = query_projections.grad
        pos_cache = pos_projections.grad
        neg_cache = neg_projections.grad

        return query_cache, pos_cache, neg_cache, con_loss.detach()
    
    def forward_backward(
            self,
            model: Lusifer,
            model_inputs: Dict[str, torch.Tensor],
            stage: RandContext, 
            query_cache: torch.Tensor, # (batch_size, embed_dim)
            pos_cache: torch.Tensor, # (batch_size, num_pos, embed_dim)
            neg_cache: torch.Tensor, # (batch_size, num_neg, embed_dim)
            ):
        """
        Forward and backward pass through the model.
        :param model: model to forward pass through
        :param model_inputs: inputs to the model
        :param stage: random states
        :param query_cache: query gradient cache
        :param pos_cache: positive gradient cache
        :param neg_cache: negative gradient cache
        """
        with stage:
            query_input_ids = model_inputs['query_input_ids']
            query_attention_mask = model_inputs['query_attention_mask']
            query_prompt_length = model_inputs['query_prompt_length']

            pos_input_ids = model_inputs['pos_input_ids']
            pos_attention_mask = model_inputs['pos_attention_mask']
            pos_prompt_length = model_inputs['pos_prompt_length']

            neg_input_ids = model_inputs['neg_input_ids']
            neg_attention_mask = model_inputs['neg_attention_mask']
            neg_prompt_length = model_inputs['neg_prompt_length']

            B, P, _ = pos_input_ids.size()
            B, N, _ = neg_input_ids.size()

            # Forward pass
            query_projections = model(
                input_ids=query_input_ids,
                attention_mask=query_attention_mask,
                prompt_length=query_prompt_length,
            )['projection']

            pos_projections = model(
                input_ids=einops.rearrange(pos_input_ids, 'b n l -> (b n) l'),
                attention_mask=einops.rearrange(pos_attention_mask, 'b n l -> (b n) l'),
                prompt_length=einops.rearrange(pos_prompt_length, 'b n -> (b n)'),
            )['projection'] # (batch_size * num_pos, embed_dim)
            pos_cache = einops.rearrange(pos_cache, 'b n d -> (b n) d', b=B, n=P)

            neg_projections = model(
                input_ids=einops.rearrange(neg_input_ids, 'b n l -> (b n) l'),
                attention_mask=einops.rearrange(neg_attention_mask, 'b n l -> (b n) l'),
                prompt_length=einops.rearrange(neg_prompt_length, 'b n -> (b n)'),
            )['projection'] # (batch_size * num_neg, embed_dim)
            neg_cache = einops.rearrange(neg_cache, 'b n d -> (b n) d', b=B, n=N)

            surrougate = torch.dot(query_projections.flatten(), query_cache.flatten()) \
                + torch.dot(pos_projections.flatten(), pos_cache.flatten()) \
                + torch.dot(neg_projections.flatten(), neg_cache.flatten())
            
            # Backward pass
            self.fabric.backward(surrougate)

    def train_step(
            self,
            model: Lusifer,
            batch: Dict[str, torch.Tensor],
            ) -> torch.Tensor:
        """
        Train step for gradient cache training that includes forward, backward pass.
        :param model: model to train
        :param batch: batch of inputs
        :return: loss value
        """
        # Split input into chunks
        splitted_inputs = split_input(batch, self.chunk_size)

        # Forward pass for each chunk
        rnd_stage = []
        all_query_projections = []
        all_pos_projections = []
        all_neg_projections = []
        for chunk in splitted_inputs:
            query_projections, pos_projections, neg_projections, rnd_state = self.forward_no_grad(model, chunk)
            all_query_projections.append(query_projections)
            all_pos_projections.append(pos_projections)
            all_neg_projections.append(neg_projections)
            rnd_stage.append(rnd_state)
        all_query_projections = torch.cat(all_query_projections, dim=0)
        all_pos_projections = torch.cat(all_pos_projections, dim=0)
        all_neg_projections = torch.cat(all_neg_projections, dim=0)

        # Build cache for representations from all chunks
        labels = batch['query_labels']
        query_cache, pos_cache, neg_cache, con_loss = self.build_cache(
            query_projections=all_query_projections,
            pos_projections=all_pos_projections,
            neg_projections=all_neg_projections,
            query_labels=labels,
        )
        query_cache = query_cache.split(self.chunk_size, dim=0)
        pos_cache = pos_cache.split(self.chunk_size, dim=0)
        neg_cache = neg_cache.split(self.chunk_size, dim=0)

        # Forward and backward pass for each chunk
        accumulated_flags = [True for _ in range(len(splitted_inputs)-1)] + [False]
        for chunk, qc, pc, nc, stage, flag in zip(splitted_inputs, query_cache, pos_cache, neg_cache, rnd_stage, accumulated_flags):
            with self.fabric.no_backward_sync(model, enabled=flag):
                self.forward_backward(
                    model=model,
                    model_inputs=chunk,
                    stage=stage,
                    query_cache=qc,
                    pos_cache=pc,
                    neg_cache=nc,
                )
        return con_loss
    
    def fit_epoch(
            self,
            model: Lusifer,
            train_loader: DataLoader,
            stage: Dict[str, Any],
            lr_max_steps: int = 1000,
            grad_norm_clip: float = None,
            log_interval: int = 1,
            checkpoint_iterval: Optional[int] = 10000,
            checkpoint_dir: Optional[str] = './checkpoints/',
            checkpoint_filter: Optional[Callable] = None,
            eval_batch_size: Optional[int] = 32,
            ):
        """
        Fit epoch for gradient cache training.
        """
        optimizer: torch.optim.Optimizer = stage["optimizer"]
        scheduler : torch.optim.lr_scheduler.LambdaLR = stage.get("scheduler", None)
        current_step = stage.get("toal_iter_num", 0) # checkpoint iteration number
        epoch_num = stage.get("epoch_num", 0) # checkpoint epoch number
        self.fabric.print(f"Starting epoch {epoch_num} with {len(train_loader)} iterations")
        model.train()

        for batch_idx, batch in enumerate(train_loader):
            if current_step > batch_idx:
                continue
            if current_step > lr_max_steps:
                break

            current_step = current_step + 1
            if current_step == 1:
                size_info = {k: v.size() for k, v in batch.items() if isinstance(v, torch.Tensor)}
                self.fabric.print("First batch data: {}".format(size_info))

            iter_t0 = time.perf_counter()  
            con_loss = self.train_step(model=model, batch=batch)
            if self.cross_batch_loss:
                con_loss = con_loss / self.fabric.world_size

            if grad_norm_clip is not None:
                self.fabric.clip_gradients(model, optimizer, max_norm=grad_norm_clip)
            
            optimizer.step()
            optimizer.zero_grad()
            if scheduler is not None:
                scheduler.step()

            # Log metrics
            if current_step % log_interval == 0:
                t1 = time.perf_counter()

                metrics = {
                    'con_loss': con_loss.item(),
                    'iter_time': t1 - iter_t0,
                    'epoch': epoch_num,
                    # 'iter_num': batch_idx,
                    'lr': scheduler.get_last_lr()[0] if scheduler is not None else optimizer.param_groups[0]['lr'],
                }
                self.fabric.log_dict(metrics, step=current_step)
                self.fabric.print(
                    f"Epoch {epoch_num} | Iter {batch_idx} |"
                    f" ConLoss: {metrics['con_loss']:.4f} |"
                    f" LR: {metrics['lr']} |"
                    f" Iter time: {metrics['iter_time']:.4f}s |"
                )
            
            # Save checkpoint and evaluate
            if current_step % checkpoint_iterval == 0:
                checkpoint_path = pathlib.Path(checkpoint_dir) / f"checkpoint_step-{current_step}_epoch-{epoch_num}.ckpt"
                checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
                stage = {
                    "model": model,
                    "optimizer": optimizer,
                    "scheduler": scheduler,
                    "iter_num": current_step,
                    "epoch_num": epoch_num if batch_idx < len(train_loader)-1 else epoch_num + 1,
                }
                if checkpoint_filter is not None:
                    self.fabric.save(checkpoint_path, stage, filter={'model': checkpoint_filter})
                else:
                    self.fabric.save(checkpoint_path, stage)
                self.fabric.print(f"Checkpoint saved at {checkpoint_path}")
                self.fabric.barrier()
        
                # Restore model from checkpoint
                torch.cuda.empty_cache()
                self.fabric.load(checkpoint_path, stage, strict=False)
                model = stage.pop("model")
                optimizer = stage.pop("optimizer")
                scheduler = stage.pop("scheduler")

                # Evaluate model
                if self.fabric.global_rank == 0:
                    self.fabric.print("Evaluating model")
                    model_hprams = model.hprams
                    eval_model = Lusifer(**model_hprams)
                    self.fabric.print(f"Loading model from {checkpoint_path}")
                    stage_dict = torch.load(checkpoint_path)
                    eval_model.load_state_dict(stage_dict['model'], strict=False)
                    eval_model.eval()
                    eval_model = eval_model.to(0)
                    # Eval logic here
                    self.fabric.print("Model evaluation finished")
                
                self.fabric.barrier()
        return checkpoint_path