import argparse, time
import torch
from torch.optim import Adam, AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from utils.data_utils import load_snapshots
from models.full_model import ImpactModel
from rich.console import Console
import os
from utils.plotting import plot_pred_true_distributions_with_ci, plot_yearly_aggregates
import numpy as np

# (1) Function to store each evaluated model checkpoint
def save_evaluated_model_checkpoint(model, optimizer, epoch, current_male_values, current_rmsle_values, args, training_loss, run_dir, console): 
    """
    Saves a checkpoint of the model, optimizer, and metrics after an evaluation step.
    The filename includes the epoch and up to the first 5 MALE values (rounded).
    """
    if current_male_values is None:
        console.print("[yellow]Skipping saving evaluated model: MALE values are not available.[/yellow]")
        return

    male_components = []
    # Ensure current_male_values is a list. It should be after processing in main.
    num_male_values_to_include = min(5, len(current_male_values))

    for i in range(num_male_values_to_include):
        try:
            # Round to 3 decimal places for the filename
            male_components.append(f"male{i}_{round(float(current_male_values[i]), 3):.3f}")
        except (ValueError, TypeError) as e:
            console.print(f"[yellow]Warning: Could not process MALE value at index {i} for filename component: '{current_male_values[i]}'. Error: {e}[/yellow]")
            male_components.append(f"male{i}_error")

    if not male_components and current_male_values:
            male_str = "male_values_present_but_error_in_formatting"
    elif not male_components:
        male_str = "no_male_values"
    else:
        male_str = "_".join(male_components)
        
    # Construct a descriptive filename
    eval_model_filename = f"evaluated_model_epoch{epoch:03d}_{male_str}_{args.eval_mode}.pt"
    eval_model_path = os.path.join(run_dir, eval_model_filename)

    try:
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'training_loss': training_loss, # Training loss at the time of this epoch's training step
            'eval_male': current_male_values,
            'eval_rmsle': current_rmsle_values,
            'args': args # Save training arguments for this specific model
        }, eval_model_path)
        console.print(f"[cyan]Evaluated model checkpoint saved: {eval_model_path}[/cyan]")
    except Exception as e:
        console.print(f"[red]Error saving evaluated model checkpoint: {e}[/red]")

def drop_all(authors):
    return []

def drop_none(authors):
    return authors

def drop_first(authors):
    return authors[1:]

def drop_last(authors):
    return authors[:-1]

def keep_first(authors):
    return authors[:1]

def keep_last(authors):
    return authors[-1:]

def drop_first_and_last(authors):
    return authors[1:-1]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_years", nargs=2, type=int, required=True,
                        help="e.g. 2006 2014 inclusive, the yeas to train on")
    parser.add_argument("--test_years", nargs=2, type=int, required=True,
                        help="e.g. 2015 2018 inclusive, the yeas to test on")
    parser.add_argument("--hidden_dim", type=int, default=32) # empirically found to be good
    parser.add_argument("--num_layers", type=int, default=3, help="Number of RGCN layers")
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout in the RGCN layers")
    parser.add_argument("--epochs", type=int, default=240, help="Total number of epochs to train for.")
    parser.add_argument("--batch_size", type=int, default=256)  # unused in v1
    parser.add_argument("--lr", type=float, default=1e-2) # empirically found to be good
    parser.add_argument("--beta", type=float, default=0) # regularization parameter, 0 is no regularization and works well
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--cold_start_prob", type=float, default=0.5,
                        help="probability that a training paper is treated as "
                             "venue/reference-free (cold-start calibration)") # 1 makes the training set 100% cold-start and 0 makes it 0% cold-start. 1 makes the training easier. 
    parser.add_argument("--eval_mode", choices=["paper", "team"], default="team",
                        help="'paper' = original evaluation  |  'team' = counter-factual")

    parser.add_argument("--load_checkpoint", type=str, default="",
                        help="Path to a .pt checkpoint file to load model and optimizer states for continuing training.")
    parser.add_argument("--training_off", type=int, default=0,
                    help="If 1, training is turned off and the model is evaluated only. If 0, training is turned on and the model is trained.")
    parser.add_argument("--input_feature_model", choices=['all features', 'drop topic', 'drop authors', 'drop social', 'drop author topic'], default='all features',
                        help="Ablation study options: 'all features' = full model with all features | "
                             "'drop topic' = replace paper topic embeddings with yearly mean | "
                             "'drop author topic' = replace author topic embeddings with yearly mean | "
                             "'drop social' = replace author social embeddings with yearly mean | "
                             "'drop authors' = replace both author topic AND social embeddings with yearly means")
    parser.add_argument("--inference_time_author_dropping", type=str, default=False,
                        help="Whether to drop authors from the training set. Default is False.")
    parser.add_argument("--inference_time_topic_dropping", type=str, default=False,
                        help="Whether to drop topic at inference time. Options: False, 'drop topic'. Default is False.")
    parser.add_argument("--inference_time_social_dropping", type=str, default=False,
                        help="Whether to drop social features at inference time. Options: False, 'drop social'. Default is False.")
    parser.add_argument("--weight_decay", type=float, default=5e-5,
                        help="Weight decay for the optimizer. Default is 5e-5.")
    parser.add_argument("--grad_clip", type=float, default=1.0,
                        help="Gradient clipping value. Default is 1.0.")
    parser.add_argument("--lr_patience", type=int, default=4,
                        help="Patience for learning rate scheduler. Default is 5.")
    parser.add_argument("--lr_factor", type=float, default=0.5,
                        help="Factor to reduce learning rate by. Default is 0.5.")
    parser.add_argument("--early_stop_patience", type=int, default=10,
                        help="Patience for early stopping. Default is 10.")
    parser.add_argument("--min_lr", type=float, default=1e-5,
                        help="Minimum learning rate. Default is 1e-5.")
    args = parser.parse_args()

    # Create descriptive run directory name with experiment type
    feature_model_name = args.input_feature_model.replace(' ', '_').replace('_', '_')
    run_dir_suffix = f"_{args.eval_mode}_{feature_model_name}"
    if args.load_checkpoint: # Add suffix if resuming to distinguish run directory
        run_dir_suffix += f"_resumedFrom_{os.path.splitext(os.path.basename(args.load_checkpoint))[0]}"
    run_dir = os.path.join("runs", time.strftime("%Y%m%d_%H%M%S") + run_dir_suffix)
    os.makedirs(run_dir, exist_ok=True)

    log_file_path = os.path.join(run_dir, "training_log.log")

    console = Console(record=True)


    try: # Wrap the main logic in try...finally to ensure log file is closed
        console.print(f"Training log will be saved to: {log_file_path}")
        console.print(f"Run directory: {run_dir}")
        console.print(f"Arguments: {args}")
        console.print(f"CUDA available: {torch.cuda.is_available()}")
        five_years_before_train_years = list(range(args.train_years[0] - 5, args.train_years[0]))
        train_years = list(range(args.train_years[0], args.train_years[1] + 1))
        test_years = list(range(args.test_years[0], args.test_years[1] + 1))
        console.print(f"[bold]Five years before train years:[/bold] {five_years_before_train_years}")
        console.print(f"[bold]Train years:[/bold] {train_years}")
        console.print(f"[bold]Test years:[/bold]  {test_years}")
        console.print(f"[bold]Device:[/bold] {args.device}")
        
        snapshots = load_snapshots("data/yearly_snapshots_specter2_social_Info_starting_from_year_1/G_{}.pt", five_years_before_train_years + train_years + test_years)
        snapshots = [g.to(args.device) for g in snapshots]
        
        author_raw_ids = snapshots[-1]['author'].raw_ids          # list[str]
        AUT2IDX = {aid: i for i, aid in enumerate(author_raw_ids)}
        idx2aut = author_raw_ids                                  # same order

        metadata = snapshots[0].metadata()   # metadata: (['paper', 'author', 'venue'], [('author', 'writes', 'paper'),
        # ('paper', 'written_by', 'author'), ('paper', 'cites', 'paper'), ('paper', 'cited_by', 'paper'),
        # ('paper', 'published_in', 'venue'), ('venue', 'publishes', 'paper')])
        in_dims = {
            "author": snapshots[0]["author"].x.size(-1),           # 768
            "paper":  snapshots[0]["paper"].x_title_emb.size(-1),  # 768
            "venue":  snapshots[0]["venue"].x.size(-1),            # 768  
        }
        model = ImpactModel(metadata,
                            in_dims,
                            hidden_dim=args.hidden_dim,
                            num_layers=args.num_layers,
                            dropout=args.dropout,
                            beta=args.beta,
                            cold_start_prob=args.cold_start_prob,
                            aut2idx=AUT2IDX,
                            idx2aut=idx2aut,
                            input_feature_model=args.input_feature_model,
                            args=args,
                            ).to(args.device)
        optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        
        # Initialize learning rate scheduler
        scheduler = ReduceLROnPlateau(
            optimizer, 
            mode='min', 
            factor=args.lr_factor,
            patience=args.lr_patience,
            min_lr=args.min_lr,
        )

        # Early stopping variables
        best_val_loss = float('inf')
        early_stop_counter = 0
        best_model_state = None

        start_epoch = 1
        loaded_checkpoint_args_info = "None"

        if args.load_checkpoint:
            if os.path.exists(args.load_checkpoint):
                try:
                    console.print(f"[cyan]Attempting to load checkpoint from: {args.load_checkpoint}[/cyan]")
                    checkpoint = torch.load(args.load_checkpoint, map_location=args.device, weights_only=False)
                    
                    model.load_state_dict(checkpoint['model_state_dict'])
                    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                    
                    loaded_epoch = checkpoint.get('epoch', 0)
                    start_epoch = loaded_epoch + 1 
                    
                    if 'args' in checkpoint:
                        loaded_checkpoint_args_info = str(vars(checkpoint['args']))
                        console.print(f"[magenta]Arguments from loaded checkpoint (for reference): {loaded_checkpoint_args_info}[/magenta]")
                    
                    console.print(f"[green]Checkpoint loaded successfully. Model and optimizer states restored. "
                                  f"Resuming training from epoch {start_epoch}. Current total epochs set to {args.epochs}.[/green]")

                    if start_epoch > args.epochs:
                        console.print(f"[yellow]Warning: Loaded checkpoint from epoch {loaded_epoch}. "
                                      f"The new start epoch {start_epoch} is greater than the total configured epochs {args.epochs}. "
                                      f"Training will not run unless --epochs is increased beyond {loaded_epoch}.[/yellow]")
                except Exception as e:
                    console.print(f"[red]Error loading checkpoint '{args.load_checkpoint}': {e}. Starting training from scratch (epoch 1).[/red]")
                    start_epoch = 1  
            else:
                console.print(f"[yellow]Checkpoint file not found: {args.load_checkpoint}. Starting training from scratch (epoch 1).[/yellow]")
                start_epoch = 1


        best_male_metric = float('inf')
        best_epoch_val = 0  
        best_model_path = ""

        
        current_args_path = os.path.join(run_dir, "current_run_args.txt")
        with open(current_args_path, 'w') as f:
            if hasattr(args, '__dict__'):
                for arg_name, arg_val in vars(args).items():
                    f.write(f"{arg_name}: {arg_val}\n")
            else:
                    f.write(str(args))
        console.print(f"Current run arguments saved to {current_args_path}")

        if args.load_checkpoint and loaded_checkpoint_args_info != "None":
            loaded_args_path = os.path.join(run_dir, "loaded_checkpoint_args_info.txt")
            with open(loaded_args_path, 'w') as f:
                f.write(loaded_checkpoint_args_info)
            console.print(f"Arguments info from loaded checkpoint saved to {loaded_args_path}")

        if start_epoch > args.epochs:
            console.print(f"[yellow]Training skipped: start_epoch ({start_epoch}) > total epochs ({args.epochs}).[/yellow]")
        else:
            console.print(f"Starting training from epoch {start_epoch} to {args.epochs}.")

        if args.training_off:
            console.print(f"[yellow]Training is turned off (training_off={args.training_off}). No training will be performed.[/yellow]")
            model.eval()
            console.print(f"Evaluating model using ckpt: {args.load_checkpoint} ")
            with torch.no_grad():
                current_male_values = None
                current_rmsle_values = None
                if args.inference_time_author_dropping=='no author':
                    drop_fn = drop_all
                elif args.inference_time_author_dropping=='all authors':
                    drop_fn = drop_none
                elif args.inference_time_author_dropping=='drop_first':
                    drop_fn = drop_first
                elif args.inference_time_author_dropping=='drop_last':
                    drop_fn = drop_last
                elif args.inference_time_author_dropping=='keep_first':
                    drop_fn = keep_first
                elif args.inference_time_author_dropping=='keep_last':
                    drop_fn = keep_last
                elif args.inference_time_author_dropping=='drop_first_and_last':
                    drop_fn = drop_first_and_last
                else:
                    drop_fn = None

                if args.eval_mode == "paper":
                    male, rmsle, mape = model.evaluate(
                        snapshots,
                        list(range(len(train_years) + 5, len(train_years)+len(test_years) + 5)), # 5 is the first year of the training data
                        start_year=train_years[0]
                    )
                else:  # counter-factual
                    male, rmsle, mape, y_true, y_predict = model.evaluate_team(
                        snapshots,
                        list(range(len(train_years) + 5, len(train_years)+len(test_years) + 5)), # 5 is the first year of the training data
                        start_year=train_years[0],
                        return_raw=True,
                        author_drop_fn=drop_fn
                    )
                
                
                console.print(f"[green]Eval ({args.eval_mode})[/green] "
                                f"MALE {male}  RMSLE {rmsle} MAPE {mape}")
                
                # Save raw predictions for sensitivity analysis
                predictions_dir = "./sensitivity_results"
                os.makedirs(predictions_dir, exist_ok=True)
                
                # Create descriptive filename for this experiment (replace spaces with underscores)
                experiment_name = f"author_{args.inference_time_author_dropping}__topic_{args.input_feature_model}".replace(' ', '_')
                pred_file = os.path.join(predictions_dir, f"{experiment_name}_predictions.npy")
                true_file = os.path.join(predictions_dir, f"{experiment_name}_ground_truth.npy")
                
                # Save predictions and ground truth
                np.save(pred_file, y_predict.numpy())
                np.save(true_file, y_true.numpy())
                
                # Calculate and save sensitivity metrics
                total_citations_pred = y_predict.sum(dim=1).numpy()  # Total citations per paper
                total_citations_true = y_true.sum(dim=1).numpy()
                
                sensitivity_metrics = {
                    'experiment': experiment_name,
                    'male': male.tolist(),
                    'rmsle': rmsle.tolist(), 
                    'mape': mape.tolist(),
                    'prediction_mean': float(total_citations_pred.mean()),
                    'prediction_std': float(total_citations_pred.std()),
                    'prediction_range': [float(total_citations_pred.min()), float(total_citations_pred.max())],
                    'coefficient_of_variation': float(total_citations_pred.std() / total_citations_pred.mean()) if total_citations_pred.mean() > 0 else 0,
                    'num_papers': int(len(total_citations_pred)),
                }
                
                metrics_file = os.path.join(predictions_dir, f"{experiment_name}_metrics.json")
                import json
                with open(metrics_file, 'w') as f:
                    json.dump(sensitivity_metrics, f, indent=2)
                
                console.print(f"[cyan]Predictions saved to: {pred_file}[/cyan]")
                console.print(f"[cyan]Metrics saved to: {metrics_file}[/cyan]")
                console.print(f"[cyan]Prediction std: {sensitivity_metrics['prediction_std']:.4f}, mean: {sensitivity_metrics['prediction_mean']:.4f}[/cyan]")
                
                plot_pred_true_distributions_with_ci(
                    y_true.numpy(),                   # expects numpy
                    y_predict.numpy(),
                    horizons=[f"Year {i}" for i in range(5)],
                    bins=100,
                    plot_type="hist",
                    save_path=f"./figs/author_{args.inference_time_author_dropping}__topic_{args.input_feature_model}_Year1_full_dist_ci_hist.png",
                    title=f"author_{args.inference_time_author_dropping}__topic_{args.input_feature_model}",
                    show=False)
                
                plot_yearly_aggregates(
                    y_true.numpy(),
                    y_predict.numpy(),
                    horizons=[f"Year {i}" for i in range(5)],
                    agg_fn=np.median,
                    show_iqr=True,
                    save_path=f"./figs/author_{args.inference_time_author_dropping}__topic_{args.input_feature_model}_Year1_full_median_iqr.png",
                    title=f"author_{args.inference_time_author_dropping}__topic_{args.input_feature_model}",
                    show=False)

            return 0

        loss = None
        log = {}

        for epoch in range(start_epoch, args.epochs + 1):
            console.save_text(log_file_path, clear=False)
            model.train()
            optimizer.zero_grad()
            
            # Training step
            loss, log = model(snapshots, list(range(5, 5 + len(train_years))), train_years[0])
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            
            optimizer.step()
            log_items_str = "  ".join(f"{k}:{v:.4f}" for k, v in log.items())
            console.log(f"Epoch {epoch:03d}  Loss: {loss.item():.4f}  {log_items_str}")
            # Validation step every 10 epochs
            if epoch % 10 == 0 or epoch == args.epochs or epoch == 1:
                model.eval()
                with torch.no_grad():
                    if args.eval_mode == "paper":
                        male, rmsle, mape = model.evaluate(
                            snapshots,
                            list(range(len(train_years) + 5, len(train_years)+len(test_years) + 5)),
                            start_year=train_years[0]
                        )
                    else:  # counter-factual
                        male, rmsle, mape, y_true, y_predict = model.evaluate_team(
                            snapshots,
                            list(range(len(train_years) + 5, len(train_years)+len(test_years) + 5)),
                            start_year=train_years[0],
                            return_raw=True
                        )
                    
                    # Print detailed metrics for each year
                    console.print(f"\n[bold]Epoch {epoch:03d} Validation Results:[/bold]")
                    console.print("[bold]Year-wise MALE:[/bold]")
                    for i, m in enumerate(male):
                        console.print(f"  Year {i+1}: {m:.4f}")
                    print(f"male: {male}")
                    console.print("[bold]Year-wise RMSLE:[/bold]")
                    for i, r in enumerate(rmsle):
                        console.print(f"  Year {i+1}: {r:.4f}")
                    print(f"rmsle: {rmsle}")
                    # console.print("[bold]Year-wise MAPE:[/bold]")
                    # for i, m in enumerate(mape):
                    #     console.print(f"  Year {i+1}: {m:.4f}")
                    
                    # Use RMSLE as validation metric for learning rate scheduling
                    val_metric = rmsle.mean().item()
                    
                    # Update learning rate
                    scheduler.step(val_metric)
                    # current learning rate
                    print(f"current learning rate: {optimizer.param_groups[0]['lr']}")
                    # Early stopping check
                    if val_metric < best_val_loss:
                        best_val_loss = val_metric
                        early_stop_counter = 0
                        best_model_state = model.state_dict()
                        # Save best model
                        best_model_path = os.path.join(run_dir, f"best_model_epoch{epoch:03d}.pt")
                        torch.save({
                            'epoch': epoch,
                            'model_state_dict': best_model_state,
                            'optimizer_state_dict': optimizer.state_dict(),
                            'val_loss': best_val_loss,
                            'male': male.tolist(),
                            'rmsle': rmsle.tolist(),
                            'mape': mape.tolist(),
                            'args': args
                        }, best_model_path)
                    else:
                        early_stop_counter += 1
                        if early_stop_counter >= args.early_stop_patience:
                            console.print(f"[yellow]Early stopping triggered at epoch {epoch}. Best validation loss: {best_val_loss:.4f}[/yellow]")
                            # Restore best model
                            model.load_state_dict(best_model_state)
                            break



                # Save evaluated model checkpoint with detailed metrics
                save_evaluated_model_checkpoint(
                    model, optimizer, epoch, male.tolist() if 'male' in locals() else None, 
                    rmsle.tolist() if 'rmsle' in locals() else None, args, loss.item(), run_dir, console
                )

        final_model_filename = f"final_model_epoch{args.epochs:03d}_{args.eval_mode}.pt" 
        actual_last_epoch = epoch if 'epoch' in locals() and start_epoch <= args.epochs else args.epochs

        final_model_path = os.path.join(run_dir, final_model_filename)
        
        if start_epoch <= args.epochs and loss is not None: 
            torch.save({
                'epoch': actual_last_epoch,  
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'final_train_loss_val': loss.item(),  
                'final_train_log': log,  
                'args': args
            }, final_model_path)
            console.print(f"Done. Final model for epoch {actual_last_epoch} saved to {final_model_path}")
        elif start_epoch <= args.epochs and loss is None: 
            console.print(f"[yellow]Training loop may have had issues; loss not defined. Final model not saved.[/yellow]")
        else: 
            console.print(f"No training performed in this run. Final model not saved. (start_epoch: {start_epoch}, args.epochs: {args.epochs})")

        if best_model_path:
            console.print(f"Best performing model (Epoch {actual_last_epoch}) from this run retained at: {best_model_path} "
                          f"with Val Loss: {best_val_loss:.4f}")
        else:
            console.print("[yellow]No best model was saved based on Val Loss during training for this run.[/yellow]")

    finally:
        console.save_text(log_file_path, clear=False)

if __name__ == "__main__":
    main()