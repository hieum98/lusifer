import pathlib
import shutil
import time
from typing import Any, Callable, Dict, Optional
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import lightning as L

from lusifer.models.lusifer import Lusifer, WrappedLusifer
from lusifer.eval.eval import eval_mteb, eval_multilingual
from lusifer.trainer.utils import clear_unused_gpu_mem

class AlignmentTrainer:
    def __init__(
        self,
        fabric: L.Fabric,
        num_accumulation_steps: int = 1
    ):
        self.fabric = fabric
        self.num_accumulation_steps = num_accumulation_steps

        self.best_overall_metric = 0.0
        self.best_en = 0.0
        self.best_multi = 0.0

    def train_step(
        self,
        batch: Dict[str, torch.Tensor],
        model: Lusifer
    ):
        universal_learner_input_ids = batch['universal_learner_input_ids'] # (bs, input_len)
        universal_learner_attention_mask = batch['universal_learner_attention_mask'] 
        lm_input_ids = batch['lm_input_ids'] # (bs, output_len)
        lm_attention_mask = batch['lm_attention_mask']
        lm_labels = batch['lm_labels'] # (bs, output_len)

        # forward pass
        loss = model(
            input_ids=universal_learner_input_ids,
            attention_mask=universal_learner_attention_mask,
            llm_input_ids=lm_input_ids,
            llm_attention_mask=lm_attention_mask,
            lm_labels=lm_labels,
            is_encoding=False
        )['loss']
        return loss

    def fit_epoch(
        self,
        model: Lusifer,
        train_loader: DataLoader,
        state: Dict[str, Any],
        lr_max_steps: int = 1000,
        grad_norm_clip: float = None,
        log_interval: int = 1,
        checkpoint_iterval: Optional[int] = 10000,
        checkpoint_dir: Optional[str] = './checkpoints/',
        checkpoint_filter: Optional[Callable] = None,
        model_revision: Optional[str] = 'v0.1',
        eval_batch_size: Optional[int] = 32,
    ):
        optimizer: torch.optim.Optimizer = state["optimizer"]
        scheduler : torch.optim.lr_scheduler.LambdaLR = state.get("scheduler", None)
        current_step = state.get("current_step", 0) # checkpoint iteration number
        epoch_num = state.get("epoch_num", 0) # checkpoint epoch number
        self.fabric.print(f"Starting epoch {epoch_num} with {len(train_loader)} iterations")
        model.train()

        for batch_idx, batch in enumerate(train_loader):

            if current_step > len(train_loader)*epoch_num + batch_idx:
                continue
            # Need to scale the lr_max_steps by num_accumulation_steps because the scheduler is called every accumulation step
            if current_step > lr_max_steps * self.num_accumulation_steps: 
                break

            current_step = current_step + 1
            if current_step == 1:
                size_info = {k: v.size() for k, v in batch.items() if isinstance(v, torch.Tensor)}
                self.fabric.print("First batch data: {}".format(size_info))

            iter_t0 = time.perf_counter()  

            is_accumulation = current_step % self.num_accumulation_steps != 0 
            if current_step % checkpoint_iterval == 0 or current_step == lr_max_steps * self.num_accumulation_steps or batch_idx + 1 == len(train_loader):
                is_accumulation = False
            
            with self.fabric.no_backward_sync(model, enabled=is_accumulation):
                loss = self.train_step(batch=batch, model=model)
                self.fabric.backward(loss / self.num_accumulation_steps)
            
            if not is_accumulation:
                if grad_norm_clip is not None:
                    self.fabric.clip_gradients(model, optimizer, max_norm=grad_norm_clip)
                optimizer.step()
                if scheduler is not None:
                    scheduler.step()
                optimizer.zero_grad()

            # Log metrics
            if current_step % log_interval == 0:
                t1 = time.perf_counter()

                metrics = {
                    'loss': loss.detach().item(),
                    'iter_time': t1 - iter_t0,
                    'epoch': epoch_num,
                    # 'iter_num': batch_idx,
                    'lr': scheduler.get_last_lr()[0] if scheduler is not None else optimizer.param_groups[0]['lr'],
                }
                self.fabric.log_dict(metrics, step=current_step)
                self.fabric.print(
                    f"Epoch {epoch_num} | Iter {batch_idx} |"
                    f" Loss: {metrics['loss']:.4f} |"
                    f" LR: {metrics['lr']} |"
                    f" Iter time: {metrics['iter_time']:.4f}s |"
                )

            # Save checkpoint and evaluate
            if current_step % checkpoint_iterval == 0 or current_step == lr_max_steps * self.num_accumulation_steps or batch_idx == len(train_loader) - 1:
                checkpoint_path = pathlib.Path(checkpoint_dir) / "lastest.ckpt"
                checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
                state = {
                    "model": model,
                    "optimizer": optimizer,
                    "scheduler": scheduler,
                    "current_step": current_step,
                    "epoch_num": epoch_num if batch_idx < len(train_loader)-1 else epoch_num + 1,
                }
                if checkpoint_filter is not None:
                    self.fabric.save(checkpoint_path, state, filter={'model': checkpoint_filter})
                else:
                    self.fabric.save(checkpoint_path, state)
                self.fabric.print(f"Checkpoint saved at {checkpoint_path}")
                clear_unused_gpu_mem()
                self.fabric.load(checkpoint_path, state, strict=False)
                model = state.pop("model")
                optimizer = state.pop("optimizer")
                scheduler = state.pop("scheduler")
                self.fabric.barrier()

                # Evaluate model
                if self.fabric.global_rank == 0:
                    model_hprams = model.hprams
                    self.fabric.print("Evaluating model")
                    _model_revision = f"{model_revision}_step-{current_step}_epoch-{epoch_num}"
                    self.fabric.print(f"Model hprams: {model_hprams}")
                    self.fabric.print(f"Model revision: {_model_revision}")
                    self.fabric.print(f"Loading model from checkpoint: {checkpoint_path}")
                    eval_model = WrappedLusifer(
                        model_revision=_model_revision, 
                        model_checkpoint=checkpoint_path,
                        **model_hprams
                        )
                    mteb_results = eval_mteb(
                        model=eval_model,
                        output_folder=checkpoint_dir,
                        batch_size=eval_batch_size,
                        is_quick_run=True,
                    )
                    multilingual_results = eval_multilingual(
                        model=eval_model,
                        langs=['ru', 'vi', 'fa', 'hi', 'bn', 'yo'],
                        output_folder=checkpoint_dir,
                        batch_size=eval_batch_size,
                        is_quick_run=True,
                    )
                    results = {
                        'Avg/mteb_quick_avg': mteb_results['avg']['all_tasks'],
                        'Avg/multilingual_quick_avg': multilingual_results['avg']['all_tasks'],
                    }
                    self.fabric.log_dict(results, step=current_step)
                    # Eval logic here
                    self.fabric.print("Model evaluation finished")
                    del eval_model
                    clear_unused_gpu_mem()

                    # Save best checkpoint based on evaluation
                    if results['Avg/mteb_quick_avg'] >= self.best_en:
                        self.best_en = results['Avg/mteb_quick_avg']
                        best_checkpoint_path = pathlib.Path(checkpoint_dir) / "best_en.ckpt"
                        shutil.copy(checkpoint_path, best_checkpoint_path)
                        self.fabric.print(f"Best en checkpoint saved at {best_checkpoint_path}")
                    if results['Avg/multilingual_quick_avg'] >= self.best_multi:
                        self.best_multi = results['Avg/multilingual_quick_avg']
                        best_checkpoint_path = pathlib.Path(checkpoint_dir) / "best_multi.ckpt"
                        shutil.copy(checkpoint_path, best_checkpoint_path)
                        self.fabric.print(f"Best multi checkpoint saved at {best_checkpoint_path}")
                self.fabric.barrier()
        return checkpoint_path
