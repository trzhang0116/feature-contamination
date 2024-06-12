import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
import torch.multiprocessing as mp
mp.set_sharing_strategy('file_system')

import os
import re
import time
import json
from tqdm import tqdm

import datasets
from models import get_model
from models.utils import get_transform, initialize_attention_layers
from datasets.common import get_dataloader, maybe_dictionarize
from args import get_args
from utils import *


def main():
    args = get_args(verbose=True)
    set_seed(args.seed)

    ngpus_per_node = torch.cuda.device_count() if torch.cuda.is_available() else 0
    
    if args.dist_url == "env://" and args.world_size == -1:
        args.world_size = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1

    if args.rd:
        if args.distributed:
            args.world_size = ngpus_per_node * args.world_size
            mp.spawn(distill, nprocs=ngpus_per_node, args=(ngpus_per_node, args))
        else:
            distill(args.gpu, ngpus_per_node, args)
    else:
        if args.distributed:
            args.world_size = ngpus_per_node * args.world_size
            if args.eval:
                evaluate(args.gpu, ngpus_per_node, args, input_key='images')
            else:
                input_key = 'features' if args.feature_cache_dir is not None else 'images'
                if input_key == 'images':
                    mp.spawn(train, nprocs=ngpus_per_node, args=(ngpus_per_node, args, input_key, False))
                else:
                    train(args.gpu, ngpus_per_node, args, input_key)
        else:
            if args.eval:
                evaluate(args.gpu, ngpus_per_node, args, input_key='images')
            else:
                input_key = 'features' if args.feature_cache_dir is not None else 'images'
                train(args.gpu, ngpus_per_node, args, input_key=input_key)


def setup_model(args, load=False, from_scratch=False, include_top=True):
    if args.gpu is None:
        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    else:
        device = torch.device(f'cuda:{args.gpu}') if torch.cuda.is_available() else torch.device('cpu')
    
    model = get_model(args, from_scratch=from_scratch, include_top=include_top, device=device)
    
    # Load existing checkpoint
    if load and args.load_dir is not None:
        checkpoint = torch.load(args.load_dir, map_location=device)
        checkpoint = remove_prefix_in_checkpoints(checkpoint)
        try:
            model.pre_featurizer.load_state_dict(checkpoint)
            print(f'Loaded encoder weights from {args.load_dir}.')
        except:
            model.load_state_dict(checkpoint)
            print(f'Loaded model weights from {args.load_dir}.')
    if include_top and args.classifier_load_dir is not None:
        checkpoint = torch.load(args.classifier_load_dir, map_location=device)
        checkpoint = remove_prefix_in_checkpoints(checkpoint)
        model.classification_head.load_state_dict(checkpoint)
        print(f'Loaded classifier weights from {args.classifier_load_dir}.')    
    
    model.to(device)
    return model, device


def train(gpu, ngpus_per_node, args, input_key='features'):
    args.gpu = gpu
    if args.distributed:
        if args.dist_url == 'env://' and args.rank == -1:
            args.rank = int(os.environ['RANK']) if 'RANK' in os.environ else 0
        args.rank = args.rank * ngpus_per_node + gpu
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url, world_size=args.world_size, rank=args.rank)
    
    model, device = setup_model(args, load=args.lp or args.resume)

    if ngpus_per_node > 1 and not args.distributed:
        model = nn.DataParallel(model)
        model.input_resolution = model.module.input_resolution
        model.pre_featurizer = model.module.pre_featurizer
        model.featurizer = model.module.featurizer
        model.classification_head = model.module.classification_head
    
    if args.distributed:
        if torch.cuda.is_available():
            if args.gpu is not None:
                args.batch_size = int(args.batch_size / ngpus_per_node)
                args.num_workers = int((args.num_workers + ngpus_per_node - 1) / ngpus_per_node)
                model = nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
            else:
                model = nn.parallel.DistributedDataParallel(model)
            
        model.input_resolution = model.module.input_resolution
        model.pre_featurizer = model.module.pre_featurizer
        model.featurizer = model.module.featurizer
        model.classification_head = model.module.classification_head

    if input_key == 'images':
        aug = not args.lp
        image_encoder = None
        if isinstance(model, nn.DataParallel) or isinstance(model, nn.parallel.DistributedDataParallel):
            model.module.set_full_forward(True)
        else:
            model.set_full_forward(True)
    else:
        aug = False
        image_encoder = model.pre_featurizer
        
    dataset_class = getattr(datasets, 'ImageNetTrain')
    dataset = dataset_class(get_transform(args, model.input_resolution, aug=aug), location=args.data_dir, batch_size=args.batch_size,
                            num_workers=args.num_workers, distributed=args.distributed, pin_memory=args.pin_memory)
    num_batches = len(dataset.train_loader)
    train_loader, train_sampler = get_dataloader(dataset, is_train=True, batch_size=args.batch_size, device=device, image_encoder=image_encoder,
                                  distributed=args.distributed, pin_memory=args.pin_memory)

    criterion = nn.CrossEntropyLoss(label_smoothing=args.ls)

    params = [param for param in model.parameters() if param.requires_grad]
    optimizer = optim.AdamW(params, lr=args.lr, weight_decay=args.weight_decay)
    scheduler = cosine_lr_with_warmup(optimizer, args.lr, args.warmup_length, args.epochs*num_batches,
                                      args.lr_warm_restarts, args.restart_epochs*num_batches)

    args.current_epoch = 0

    if args.resume:
        starting_epoch = int(re.findall(r'\d+', args.load_dir)[-1]) + 1
    else:
        starting_epoch = 1
    
    # Train loop
    for epoch in range(starting_epoch, args.epochs+starting_epoch):
        if args.distributed:
            train_sampler.set_epoch(epoch-starting_epoch)
        model.train()
        
        for i, batch in enumerate(tqdm(train_loader)):
            start_time = time.time()
            step = i + (epoch - starting_epoch) * num_batches
            scheduler(step)

            batch = maybe_dictionarize(batch)
            x = batch[input_key].to(device)
            y = batch['labels'].to(device)
            data_time = time.time() - start_time

            logits = model(x)
            loss = criterion(logits, y)
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(params, 1.0)
            optimizer.step()

            batch_time = time.time() - start_time
            if i % args.print_freq == 0:
                percent_complete = 100 * i / len(train_loader)
                print(
                    f"Train Epoch: {epoch}/{args.epochs+starting_epoch-1} [{percent_complete:.0f}% {i}/{len(dataset.train_loader)}]\t"
                    f"LR: {optimizer.param_groups[0]['lr']:.6f}\t Loss: {loss.item():.6f}\tData (t) {data_time:.3f}\tBatch (t) {batch_time:.3f}", flush=True
                )

        # Save model and optimizer
        if not args.distributed or (args.distributed and args.rank % ngpus_per_node == 0):
            if args.result_dir is not None:
                model_save_dir = os.path.join(args.result_dir, 'models')
                os.makedirs(model_save_dir, exist_ok=True)
                if args.lp:
                    model_save_path = os.path.join(model_save_dir, f'classifier_{args.exp_name}_epoch{epoch}.pt')
                    torch.save(model.classification_head.state_dict(), model_save_path)
                    print(f'Classifier saved to {model_save_path}.')
                else:
                    model_save_path = os.path.join(model_save_dir, f'model_{args.exp_name}_epoch{epoch}.pt')
                    torch.save(model.state_dict(), model_save_path)
                    print(f'Model saved to {model_save_path}.')
                optim_save_path = os.path.join(model_save_dir, f'optim_{args.exp_name}_epoch{epoch}.pt')
                torch.save(optimizer.state_dict(), optim_save_path)
                print(f'Optimizer saved to {optim_save_path}.')

            # Evaluate
            args.current_epoch = epoch
            evaluate(gpu, ngpus_per_node, args, model=model, input_key=input_key)


def distill(gpu, ngpus_per_node, args):
    input_key = 'images'
    aug = True

    args.gpu = gpu
    if args.distributed:
        if args.dist_url == 'env://' and args.rank == -1:
            args.rank = int(os.environ['RANK']) if 'RANK' in os.environ else 0
        args.rank = args.rank * ngpus_per_node + gpu
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url, world_size=args.world_size, rank=args.rank)
    
    if args.rd:
        include_top = False
    else:
        include_top = True
    
    student_model, device = setup_model(args, load=args.resume, from_scratch=True, include_top=include_top)
    teacher_model, _ = setup_model(args, include_top=include_top)

    if args.oracle_norm_stats:
        # use freezed normalization statistics loaded from the teacher model
        student_model.pre_featurizer.load_bn_stats(teacher_model.pre_featurizer.model)

    if args.attn_init:
        # initialize the ViT attention layers in the student model using the weights of the teacher model
        initialize_attention_layers(teacher_model.pre_featurizer.model.visual.transformer,
                                    student_model.pre_featurizer.model.visual.transformer)

    teacher_model.set_full_forward(True)
    student_model.set_full_forward(True)

    if ngpus_per_node > 1 and not args.distributed:
        student_model = nn.DataParallel(student_model)
        teacher_model = nn.DataParallel(teacher_model)
        student_model.input_resolution = student_model.module.input_resolution
        student_model.pre_featurizer = student_model.module.pre_featurizer
        student_model.featurizer = student_model.module.featurizer
        if include_top:
            student_model.classification_head = student_model.module.classification_head

    if args.distributed:
        if torch.cuda.is_available():
            if args.gpu is not None:
                args.batch_size = int(args.batch_size / ngpus_per_node)
                args.num_workers = int((args.num_workers + ngpus_per_node - 1) / ngpus_per_node)
                student_model = nn.parallel.DistributedDataParallel(student_model, device_ids=[args.gpu])
            else:
                student_model = nn.parallel.DistributedDataParallel(student_model)
            
        student_model.input_resolution = student_model.module.input_resolution
        student_model.pre_featurizer = student_model.module.pre_featurizer
        student_model.featurizer = student_model.module.featurizer
        if include_top:
            student_model.classification_head = student_model.module.classification_head

    dataset_class = getattr(datasets, 'ImageNetTrain')
    dataset = dataset_class(get_transform(args, student_model.input_resolution, aug=aug), location=args.data_dir, batch_size=args.batch_size,
                            num_workers=args.num_workers, distributed=args.distributed, pin_memory=args.pin_memory)
    num_batches = len(dataset.train_loader)
    train_loader, train_sampler = get_dataloader(dataset, is_train=True, batch_size=args.batch_size, device=device, image_encoder=None,
                                  distributed=args.distributed, pin_memory=args.pin_memory)
    
    # criterion = nn.CrossEntropyLoss(label_smoothing=args.ls)
    
    if args.rd:
        criterion_distill = HintLoss()
        params = [param for param in student_model.pre_featurizer.parameters() if param.requires_grad] \
            + [param for param in student_model.featurizer.parameters() if param.requires_grad]
        optimizer = optim.AdamW(params, lr=args.lr, weight_decay=args.weight_decay)
        scheduler = cosine_lr_with_warmup(optimizer, args.lr, args.warmup_length, args.epochs*num_batches,
                                          args.lr_warm_restarts, args.restart_epochs*num_batches)

    if args.resume:
        starting_epoch = int(re.findall(r'\d+', args.load_dir)[-1]) + 1
    else:
        starting_epoch = 1
    
    # Train loop
    for epoch in range(starting_epoch, args.epochs+starting_epoch):
        if args.distributed:
            train_sampler.set_epoch(epoch-starting_epoch)

        student_model.train()
        for i, batch in enumerate(tqdm(train_loader)):
            start_time = time.time()
            step = i + (epoch - starting_epoch) * num_batches
            scheduler(step)

            batch = maybe_dictionarize(batch)
            x = batch[input_key].to(device)
            y = batch['labels'].to(device)
            data_time = time.time() - start_time

            if args.rd:
                if args.attn_distill or args.layer_distill:
                    teacher_feats, teacher_aux_outputs = teacher_model(x)
                    student_feats, student_aux_outputs = student_model(x)
                    loss_feat = args.distill_weight * criterion_distill(student_feats, teacher_feats)
                    loss_attn = 0.
                    loss_layer = 0.
                    if args.attn_distill:
                        teacher_attn_weights = [teacher_aux_output['attn_weights'] for teacher_aux_output in teacher_aux_outputs]
                        student_attn_weights = [student_aux_output['attn_weights'] for student_aux_output in student_aux_outputs]
                        for teacher_attn_weight, student_attn_weight in zip(teacher_attn_weights, student_attn_weights):
                            loss_attn += criterion_distill(student_attn_weight, teacher_attn_weight)
                        loss_attn = args.attn_distill_weight * loss_attn / len(teacher_attn_weights)
                    if args.layer_distill:
                        teacher_hidden_states = [teacher_aux_output['hidden_states'] for teacher_aux_output in teacher_aux_outputs]
                        student_hidden_states = [student_aux_output['hidden_states'] for student_aux_output in student_aux_outputs]
                        for teacher_hidden_state, student_hidden_state in zip(teacher_hidden_states, student_hidden_states):
                            loss_layer += criterion_distill(student_hidden_state, teacher_hidden_state)
                        loss_layer = args.layer_distill_weight * loss_layer / len(teacher_hidden_states)
                    loss = loss_feat + loss_layer + loss_attn
                else:
                    teacher_feats = teacher_model(x)
                    student_feats = student_model(x)
                    loss_feat = args.distill_weight * criterion_distill(student_feats, teacher_feats)
                    # loss_clf = criterion(student_logits, y)
                    # loss = loss_feat + loss_clf
                    loss = loss_feat
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(params, 1.0)
                optimizer.step()
            else:
                raise NotImplementedError

            batch_time = time.time() - start_time
            if i % args.print_freq == 0:
                percent_complete = 100 * i / len(train_loader)
                aux_str = ""
                if args.attn_distill or args.layer_distill:
                    aux_str += f"\t Repr loss: {loss_feat.item():.6f}"
                if args.attn_distill:
                    aux_str += f"\t Attn loss: {loss_attn.item():.6f}"
                if args.layer_distill:
                    aux_str += f"\t Layer loss: {loss_layer.item():.6f}"
                print(
                    f"Train Epoch: {epoch}/{args.epochs+starting_epoch-1} [{percent_complete:.0f}% {i}/{len(dataset.train_loader)}]\t"
                    f"LR: {optimizer.param_groups[0]['lr']:.6f}\t Loss: {loss.item():.6f}{aux_str}\t Data (t) {data_time:.3f}\t Batch (t) {batch_time:.3f}",
                    flush=True
                )

        # Save model and optimizer
        if not args.distributed or (args.distributed and args.rank % ngpus_per_node == 0):
            if epoch % 10 == 0 and args.result_dir is not None:
                model_save_dir = os.path.join(args.result_dir, 'models')
                os.makedirs(model_save_dir, exist_ok=True)
                model_save_path = os.path.join(model_save_dir, f'model_{args.exp_name}_epoch{epoch}.pt')
                torch.save(student_model.pre_featurizer.state_dict(), model_save_path)
                print(f'Model saved to {model_save_path}.')
                optim_save_path = os.path.join(model_save_dir, f'optim_{args.exp_name}_epoch{epoch}.pt')
                torch.save(optimizer.state_dict(), optim_save_path)
                print(f'Optimizer saved to {optim_save_path}.')


def evaluate(gpu, ngpus_per_node, args, model=None, input_key='features'):
    if args.eval_datasets is None:
        return
    
    if model is None:
        model, device = setup_model(args, load=True)
    else:
        if gpu is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            device = f'cuda:{gpu}' if torch.cuda.is_available() else 'cpu'
    
    model.eval()
    
    if ngpus_per_node > 1 and not (isinstance(model, nn.DataParallel) or isinstance(model, nn.parallel.DistributedDataParallel)):
        model = nn.DataParallel(model)
        model.input_resolution = model.module.input_resolution
        model.pre_featurizer = model.module.pre_featurizer
        model.featurizer = model.module.featurizer
        model.classification_head = model.module.classification_head

    if input_key == 'images':
        if isinstance(model, nn.DataParallel) or isinstance(model, nn.parallel.DistributedDataParallel):
            model.module.set_full_forward(True)
        else:
            model.set_full_forward(True)

    if input_key == 'images':
        image_encoder = None
    else:
        image_encoder = model.pre_featurizer

    info = vars(args)
    model_eval = model

    for _, dataset_name in enumerate(args.eval_datasets):
        print('========== Evaluating on {} =========='.format(dataset_name))
        dataset_class = getattr(datasets, dataset_name)
        dataset = dataset_class(get_transform(args, model_eval.input_resolution), location=args.data_dir, batch_size=args.batch_size,
                                num_workers=args.num_workers, pin_memory=args.pin_memory)

        eval_loader, _ = get_dataloader(dataset, is_train=False, batch_size=args.batch_size, device=device,
                                        image_encoder=image_encoder, pin_memory=args.pin_memory)
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

        if hasattr(dataset, 'post_loop_metrics'):
            # keep track of labels, predictions and metadata
            all_labels, all_preds, all_metadata = [], [], []

        with torch.no_grad():
            top1, correct, n = 0., 0., 0.
            for batch_id, batch in enumerate(tqdm(eval_loader)):
                batch = maybe_dictionarize(batch)
                x = batch[input_key].to(device)
                y = batch['labels'].to(device)

                if 'image_paths' in batch:
                    image_paths = batch['image_paths']

                logits = model_eval(x)
                projection_fn = getattr(dataset, 'project_logits', None)
                if projection_fn is not None:
                    logits = projection_fn(logits, device)

                if hasattr(dataset, 'project_labels'):
                    y = dataset.project_labels(y, device)

                pred = logits.argmax(dim=1, keepdim=True).to(device)
                
                if hasattr(dataset, 'accuracy'):
                    acc1, mask = dataset.accuracy(logits, y, image_paths, args)
                    correct += acc1
                    n += len(mask)
                else:
                    mask = pred.eq(y.view_as(pred))
                    correct += mask.sum().item()
                    n += y.size(0)
                    mask = mask.cpu().numpy().squeeze()

                if hasattr(dataset, 'post_loop_metrics'):
                    all_labels.append(y.cpu().clone().detach())
                    all_preds.append(logits.cpu().clone().detach())
                    metadata = batch['metadata'] if 'metadata' in batch else image_paths
                    all_metadata.extend(metadata)
                
            top1 = correct / n

            if hasattr(dataset, 'post_loop_metrics'):
                all_labels = torch.cat(all_labels)
                all_preds = torch.cat(all_preds)
                metrics = dataset.post_loop_metrics(all_labels, all_preds, all_metadata, args)
                if 'acc' in metrics:
                    metrics['top1'] = metrics['acc']
            else:
                metrics = {}

        if 'top1' not in metrics:
            metrics['top1'] = top1

        print('{} Top-1 acc: {:.4f}'.format(dataset_name, metrics['top1']))

        for key, val in metrics.items():
            if 'worst' in key or 'f1' in key.lower() or 'pm0' in key or 'pm10' in key:
                print('{} {}: {:.4f}'.format(dataset_name, key, val))
            info[dataset_name + ':' + key] = val

    if args.result_dir is not None:
        os.makedirs(args.result_dir, exist_ok=True)
        results_filename = os.path.join(args.result_dir, f'results_{args.exp_name}.txt')
        with open(results_filename, 'a+') as f:
            f.write(json.dumps(info, indent=2) + '\n')
    
    return info


if __name__ == '__main__':
    main()
