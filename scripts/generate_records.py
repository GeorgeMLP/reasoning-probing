from dataclasses import dataclass
import os
from pathlib import Path
import orjson
import wandb
import shutil
import click

from einops import pack, reduce, repeat
from jaxtyping import Float, Int
from torch import Tensor
from sae_lens import SAE, HookedSAETransformer
from datasets import load_dataset, Dataset
from transformers import PreTrainedTokenizerBase
import torch
import tqdm
from transformer_lens.utils import tokenize_and_concatenate

from featureinterp import utils
from featureinterp.similarity_retriever import (
    SimilarityRetriever,
    SimilarityRetrieverConfig,
    SimilarityMeasure,
)
from featureinterp.record import (
    FeatureExpressionRecord,
    SAEIndexId,
    SAEIndexRecord,
)


DEVICE = 'cuda'


def construct_tokenized_dataset(
    dataset_name: str,
    tokenizer: PreTrainedTokenizerBase,
    sae_name: str,
    sae_id_layer_format_string: str,
    max_dataset_size: int,
    max_seq_length: int
) -> Int[Tensor, 'batch seq']:
    """Load and preprocess the dataset."""
    
    streaming = True
    dataset = load_dataset(
        path=dataset_name,
        split='train',
        streaming=streaming,
    )

    # Filter out non-ASCII characters
    dataset = dataset.filter(lambda x: len(x['text']) == len(x['text'].encode()))
    # Truncate the text so we don't get many examples from each document
    chars_per_token = 4
    dataset = dataset.map(
        lambda x: {'text': x['text'][:max_seq_length * chars_per_token * 2]}
    )
    # Hacky way of getting the first max_dataset_size samples without downloading
    # the entire dataset. (Normal splitting wasn't working for some reason.)
    dataset = Dataset.from_list(list(dataset.take(max_dataset_size * 3)))
        
    first_sae, _, _ = SAE.from_pretrained(
        release=sae_name,
        sae_id=sae_id_layer_format_string.format(layer=0),
        device='cpu',
    )
    prepend_bos = first_sae.cfg.prepend_bos
    
    assert first_sae.cfg.context_size > max_seq_length
    
    token_dataset: Int[Tensor, 'batch seq'] = tokenize_and_concatenate(
        dataset=dataset,
        tokenizer=tokenizer,
        max_length=(max_seq_length + 1) if prepend_bos else max_seq_length,
        add_bos_token=prepend_bos,
        streaming=streaming,
    )['tokens']
    del first_sae
    
    # Filter out sequences with more than one special token (initial BOS is ok)
    # Some sequences have EOS in them as the tokenize_and_concatenate function
    # just concatenates all the text with <eos> inbetween and then chunks.
    max_special_tokens = 1 if prepend_bos else 0
    special_tokens = torch.tensor(tokenizer.all_special_ids).to(token_dataset.device)
    is_special_token = torch.isin(token_dataset, special_tokens)
    token_dataset = token_dataset[is_special_token.sum(dim=1) <= max_special_tokens]
    
    token_dataset = token_dataset[:max_dataset_size]
    
    return token_dataset


def write_sae_activations(
    model: HookedSAETransformer,
    saes_list: list[SAE],
    dataset_tokens: Int[Tensor, 'batch seq'],
    max_features: int,
    data_dir: Path,
) -> None:
    """Save SAE activations and tokens for a given layer."""

    sae_acts: dict[SAE, list[Float[Tensor, 'batch seq d_sae']]] = {
        sae: [] for sae in saes_list
    }
    batch_size = 12

    for i in tqdm.tqdm(range(0, len(dataset_tokens), batch_size), desc=f'Dataset'):
        batch = dataset_tokens[i:i+batch_size].to(model.device)
        logits, cache = model.run_with_cache_with_saes(
            batch,
            saes=saes_list,
            # Don't use reconstructions on forwards pass
            use_error_term=True,
        )
        
        for sae in saes_list:
            batch_sae_acts: Float[Tensor, 'batch seq d_sae'] = cache[
                f'{sae.cfg.hook_name}.hook_sae_acts_post'
            ]
            batch_sae_acts = batch_sae_acts[:, :, :max_features]
            sae_acts[sae].append(batch_sae_acts.detach().cpu())
            del batch_sae_acts

        del cache, batch, logits
        torch.cuda.empty_cache()
    
    for (sae, acts) in sae_acts.items():
        acts_packed = pack(acts, '* seq d_sae')[0]
        torch.save(acts_packed, data_dir / 'sae_acts' / f'{sae.cfg.hook_name}.pt')
        del acts, acts_packed
        torch.cuda.empty_cache()
        
    del sae_acts


@torch.inference_mode()
def write_holistic_activations(
    model: HookedSAETransformer,
    saes_list: list[SAE],
    dataset_tokens: Int[Tensor, 'batch seq'],
    data_dir: Path,
) -> None:
    
    if dataset_tokens[0, 0] == model.tokenizer.bos_token_id:
        token_start = 1
    else:
        token_start = 0
    
    cum_sae_acts: dict[SAE, Float[Tensor, 'batch d_sae']] = {}
    for sae in saes_list:
        sae_acts = torch.load(data_dir / 'sae_acts' / f'{sae.cfg.hook_name}.pt')
        # Remove the BOS token before summing activations
        sae_acts = sae_acts.to(model.device)[:, token_start:]
        cum_sae_acts[sae] = reduce(sae_acts, 'batch seq d_sae -> batch d_sae', 'mean')
    
    max_features = cum_sae_acts[saes_list[0]].shape[-1]
    
    batch_size = 16
    
    all_holistic_acts: dict[SAE, list[Float[Tensor, 'batch seq d_sae']]] = {
        sae: [] for sae in saes_list
    }

    for i in tqdm.tqdm(range(0, len(dataset_tokens), batch_size), desc=f'Dataset'):
        batch = dataset_tokens[i:i+batch_size].to(model.device)
        
        batch_drop_token_act_diffs: dict[SAE, list[Float[Tensor, 'batch d_sae']]] = {
            # Holistic activations for the BOS token are zero
            sae: (
                [torch.zeros((batch.shape[0], max_features), device='cpu')]
                if token_start == 1 else
                []
            )
            for sae in saes_list
        }
        
        # Compute the activation differences when dropping each token
        for drop_token in range(token_start, batch.shape[1]):
            batch_drop_token = pack(
                [batch[:, :drop_token], batch[:, drop_token+1:]], 'batch *'
            )[0]
            
            logits, cache = model.run_with_cache_with_saes(
                batch_drop_token,
                saes=saes_list,
                # Don't use reconstructions on forwards pass
                use_error_term=True,
            )
            
            for sae in saes_list:
                drop_sae_acts: Float[Tensor, 'batch seq d_sae'] = cache[
                    f'{sae.cfg.hook_name}.hook_sae_acts_post'
                ]
                drop_cum_sae_acts = reduce(
                    drop_sae_acts[:, token_start:, :max_features],
                    'batch seq d_sae -> batch d_sae',
                    'mean'
                )
                holistic_acts = cum_sae_acts[sae][i:i+batch_size] - drop_cum_sae_acts
                batch_drop_token_act_diffs[sae].append(holistic_acts.detach().cpu())
                del drop_sae_acts, drop_cum_sae_acts, holistic_acts
            
            del batch_drop_token, logits, cache
            torch.cuda.empty_cache()
        
        # Pack the activation differences along the sequence dimension to get the
        # holistic activations
        for sae in saes_list:
            batch_holistic_acts: Float[Tensor, 'batch seq d_sae'] = pack(
                batch_drop_token_act_diffs[sae], 'batch * d_sae'
            )[0]
            all_holistic_acts[sae].append(batch_holistic_acts.detach().cpu())
            del batch_holistic_acts
            torch.cuda.empty_cache()
    
    for (sae, holistic_acts) in all_holistic_acts.items():
        acts_packed = pack(holistic_acts, '* seq d_sae')[0].half()
        torch.save(acts_packed, data_dir / 'holistic_acts' / f'{sae.cfg.hook_name}.pt')
        del holistic_acts, acts_packed
        torch.cuda.empty_cache()
    
    del all_holistic_acts


def skip_bos_token(tokens: Int[Tensor, 'batch seq']) -> Int[Tensor, 'batch seq']:
    """Skip the BOS token."""
    return tokens[:, 1:]


@dataclass
class RecordSampleParams:
    top_sample_n: int
    quantile_n: int
    complementary_sample_n: int
    special_n: int


def write_expression_records(
    model: HookedSAETransformer,
    dataset_tokens: Int[Tensor, 'batch seq'],
    hook_name: str,
    hook_layer: int,
    data_dir: Path,
    record_sample_params: RecordSampleParams,
    max_features: int | None = None,
) -> None:
    """Generate and write feature expression records."""
    
    device = model.device
    embedding_save_path = data_dir / 'similarity_retriever'
    
    tokens: Int[Tensor, 'batch seq'] = torch.load(
        data_dir / 'tokens.pt', map_location=device, weights_only=True
    )[:, 1:]
    sae_acts: Float[Tensor, 'batch seq d_sae'] = torch.load(
        data_dir / 'sae_acts' / f'{hook_name}.pt',
        map_location=device,
        weights_only=True
    )[:, 1:]
    holistic_acts: Float[Tensor, 'batch seq d_sae'] = torch.load(
        data_dir / 'holistic_acts' / f'{hook_name}.pt',
        map_location=device,
        weights_only=True
    )[:, 1:]
    
    if os.path.exists(embedding_save_path):
        retriever_arg = embedding_save_path
    else:
        retriever_arg = Dataset.from_dict({
            'input_ids': skip_bos_token(dataset_tokens),
            'indices': torch.arange(len(dataset_tokens)),
        })
    
    config = SimilarityRetrieverConfig(
        tokenizer_name=model.name_or_path,
        dataset_path=embedding_save_path,
        device=device,
        similarity_measure=SimilarityMeasure.EMBEDDING,  # no projection
    )
    similarity_retriever = SimilarityRetriever(retriever_arg, config=config)
    config_proj = SimilarityRetrieverConfig(
        tokenizer_name=model.name_or_path,
        dataset_path=embedding_save_path,
        device=device,
        similarity_measure=SimilarityMeasure.PROJECTED_EMBEDDING,  # with projection
    )
    similarity_retriever_proj = SimilarityRetriever(retriever_arg, config=config_proj)
    
    if len(similarity_retriever.dataset) != len(tokens):
        raise RuntimeError(
            'Similarity retriever dataset is a different size than the dataset tokens. '
            'This likely means that the dataset was changed and the similarity '
            ' retriever cache is incorrect. Delete the `similarity_retriever` folder.'
        )
    
    sae_dir = data_dir / 'records' / hook_name
    sae_dir.mkdir(parents=True, exist_ok=True)

    if max_features is None:
        indices_to_process = range(sae_acts.shape[-1])
    else:
        assert max_features <= sae_acts.shape[-1], 'Too many features to process'
        indices_to_process = range(max_features)
    for latent_index in tqdm.tqdm(indices_to_process, desc=f'Latent ind', leave=False):
        id = SAEIndexId(layer_index=hook_layer, latent_index=latent_index)
        sae_index_record = construct_sae_index_record(
            id=id,
            tokens=tokens,
            sae_acts=sae_acts[:, :, latent_index],
            holistic_acts=holistic_acts[:, :, latent_index],
            tokenizer=model.tokenizer,
            retriever=similarity_retriever,
            retriever_proj=similarity_retriever_proj,
            sample_params=record_sample_params,
        )
        
        out_path = sae_dir / f'{latent_index}.json'
        with open(out_path, 'wb') as f:
            f.write(orjson.dumps(sae_index_record))


def construct_sae_index_record(
    id: SAEIndexId,
    tokens: Int[Tensor, 'batch seq'],
    sae_acts: Float[Tensor, 'batch seq'],
    holistic_acts: Float[Tensor, 'batch seq'],
    tokenizer: PreTrainedTokenizerBase,
    retriever: SimilarityRetriever,
    retriever_proj: SimilarityRetriever,
    sample_params: RecordSampleParams,
) -> SAEIndexRecord:
    """Process a single latent index and save its records."""

    top_sample_n = sample_params.top_sample_n
    quantile_n = sample_params.quantile_n
    comp_sample_n = sample_params.complementary_sample_n
    special_n = sample_params.special_n

    def get_ith_record(batch_index: Int[Tensor, '']) -> FeatureExpressionRecord:
        return FeatureExpressionRecord(
            tokens=tokenizer.batch_decode(tokens[batch_index]),
            expressions=sae_acts[batch_index].detach().cpu().tolist(),
            holistic_expressions=holistic_acts[batch_index].detach().cpu().tolist(),
            dataset_index=batch_index.item(),
        )
    
    sae_acts_max = reduce(sae_acts, 'batch seq -> batch', 'max')
    _, act_sorted_inds = torch.sort(sae_acts_max, dim=0, descending=True)
    no_act_inds = torch.where(sae_acts_max <= 0)[0]

    act_quantiles = utils.chunk_sampled_indices(len(act_sorted_inds), quantile_n, top_sample_n)
    most_act_inds = act_sorted_inds[act_quantiles[0]]
    
    retriever_tokens = tokens[act_sorted_inds[:special_n]]
    retriever_strings = tokenizer.batch_decode(retriever_tokens)
    
    random_inds = torch.randint(0, len(tokens), (comp_sample_n,))
    random_no_act_inds = no_act_inds[torch.randperm(len(no_act_inds))[:comp_sample_n]]

    retrieve_n = top_sample_n + comp_sample_n * 4
    for _ in range(10):
        sim_sorted_inds = retriever.get_similarity_sorted_sentence_indices(
            special_embeddings=retriever.get_embeddings(retriever_strings),
            retrieve_n=retrieve_n,
        )
        sim_proj_sorted_inds = retriever_proj.get_similarity_sorted_sentence_indices(
            special_embeddings=retriever_proj.get_embeddings(retriever_strings),
            retrieve_n=retrieve_n,
        )
        
        sim_sorted_inds = utils.drop_indices(sim_sorted_inds, most_act_inds)
        sim_proj_sorted_inds = utils.drop_indices(sim_proj_sorted_inds, most_act_inds)
        
        sim_inds = sim_sorted_inds[:comp_sample_n]
        sim_proj_inds = sim_proj_sorted_inds[:comp_sample_n]
        sim_no_act_inds = sim_sorted_inds[torch.isin(sim_sorted_inds, no_act_inds)]
        sim_no_act_inds = sim_no_act_inds[:comp_sample_n]
        sim_proj_no_act_inds = sim_proj_sorted_inds[torch.isin(sim_proj_sorted_inds, no_act_inds)]
        sim_proj_no_act_inds = sim_proj_no_act_inds[:comp_sample_n]
        if len(sim_no_act_inds) == comp_sample_n and len(sim_proj_no_act_inds) == comp_sample_n:
            break
        
        retrieve_n *= 2
        if retrieve_n > len(tokens):
            break
    if len(sim_no_act_inds) < comp_sample_n:
        print(
            f'Not enough similar negatives: '
            f'layer {id.layer_index}, feature {id.latent_index}, {len(sim_proj_no_act_inds) = }'
        )
    if len(sim_proj_no_act_inds) < comp_sample_n:
        print(
            f'Not enough projected similar negatives: '
            f'layer {id.layer_index}, feature {id.latent_index}, {len(sim_no_act_inds) = }'
        )
    
    return SAEIndexRecord(
        id=id,
        max_expression=sae_acts.max().item(),
        max_holistic_expression=holistic_acts.max().item(),
        most_act_records=[get_ith_record(i) for i in most_act_inds],
        random_records=[get_ith_record(i) for i in random_inds],
        random_no_act_records=[get_ith_record(i) for i in random_no_act_inds],
        similar_records=[get_ith_record(i) for i in sim_inds],
        similar_no_act_records=[get_ith_record(i) for i in sim_no_act_inds],
        similar_projected_records=[get_ith_record(i) for i in sim_proj_inds],
        similar_projected_no_act_records=[get_ith_record(i) for i in sim_proj_no_act_inds],
    )


def upload_to_wandb(
    data_root_dir: Path,
    subdir_name: str,
) -> None:
    """Upload results to Weights & Biases."""

    print('Compressing data...')
    current_dir = os.getcwd()
    os.chdir(data_root_dir)
    print(f'tar -cf {subdir_name}.tar {subdir_name}')
    os.system(f'tar -cf {subdir_name}.tar {subdir_name}')
    os.system(f'pv {subdir_name}.tar | pbzip2 >> {subdir_name}.tar.bz2')

    print('Setting up artifact...')
    wandb.init(project='feature-interp', job_type='upload-dataset')
    artifact = wandb.Artifact(
        name=subdir_name,
        type='dataset',
        description='Records, sae activations, and tokens',
        metadata={},
    )
    artifact.add_file(f'{subdir_name}.tar.bz2')
    print(f'Uploading {artifact.name} to W&B')
    wandb.log_artifact(artifact)
    
    os.remove(f'{subdir_name}.tar')
    os.remove(f'{subdir_name}.tar.bz2')
    shutil.rmtree('wandb')
    os.chdir(current_dir)


def download_from_wandb(
    data_root_dir: Path,
    subdir_name: str,
) -> None:
    """Download and extract records from Weights & Biases."""
    print('Downloading from W&B...')
    current_dir = os.getcwd()
    os.chdir(data_root_dir)
    wandb.init(project='feature-interp', job_type='download-dataset')
    
    artifact = wandb.use_artifact(f'{subdir_name}:latest')
    artifact_dir = artifact.download(data_root_dir.parent)
    
    print('Extracting data...')
    data_path = Path(artifact_dir) / f'{subdir_name}.tar.bz2'
    os.system(f'tar xvf {data_path}')

    os.remove(f'{subdir_name}.tar.bz2')
    shutil.rmtree('wandb')
    os.chdir(current_dir)


@click.command()

@click.option('--model-name', default='google/gemma-2-9b', help='Name of the model to analyze')
@click.option('--sae-name', default='gemma-scope-9b-pt-res-canonical', help='Name of the SAE model')
@click.option('--sae-id-layer-format-string', default='layer_{layer}/width_16k/canonical', help='Format string for SAE layer')
# @click.option('--model-name', default='meta-llama/Llama-3.1-8B', help='Name of the model to analyze')
# @click.option('--sae-name', default='llama_scope_lxr_8x', help='Name of the SAE model')
# @click.option('--sae-id-layer-format-string', default='l{layer}r_8x', help='Format string for SAE layer')
# @click.option('--model-name', default='gpt2', help='Name of the model to analyze')
# @click.option('--sae-name', default='gpt2-small-resid-post-v5-32k', help='Name of the SAE model')
# @click.option('--sae-id-layer-format-string', default='blocks.{layer}.hook_resid_post', help='Format string for SAE layer')

@click.option('--dataset-name', default='monology/pile-uncopyrighted', help='Dataset to use')

@click.option('--layers', default='all', help='Comma-separated list of layers to analyze, or "all", "even", "odd"')
@click.option('--max-dataset-size', default=100000, help='Maximum number of sentences to analyze')
@click.option('--max-features', default=50, help='Maximum number of features to analyze')
@click.option('--max-seq-length', default=32, help='Maximum sequence length')

@click.option('--top-sample-n', default=30, help='Number of top-activating samples to retrieve')
@click.option('--quantile-n', default=1000, help='Number of quantiles to split the top-activating samples into')
@click.option('--complementary-sample-n', default=30, help='Number of complementary samples to retrieve')
@click.option('--special-n', default=400, help='Number of positive samples used in the similarity retriever')

@click.option('--write-sae-acts/--no-write-sae-acts', default=False, help='Write SAE activations')
@click.option('--write-holistic-acts/--no-write-holistic-acts', default=False, help='Write holistic activations')
@click.option('--write-records/--no-write-records', default=False, help='Write expression records')
@click.option('--upload-wandb/--no-upload-wandb', default=False, help='Upload results to W&B')
@click.option('--download-wandb/--no-download-wandb', default=False, help='Download results from W&B')
def main(
    model_name: str,
    sae_name: str,
    sae_id_layer_format_string: str,
    dataset_name: str,
    layers: str,
    max_dataset_size: int,
    max_features: int,
    max_seq_length: int,
    
    top_sample_n: int,
    quantile_n: int,
    complementary_sample_n: int,
    special_n: int,

    write_sae_acts: bool,
    write_holistic_acts: bool,
    write_records: bool,
    upload_wandb: bool,
    download_wandb: bool,
) -> None:
    """Generate SAE index records for feature interpretation."""

    model_name_path = model_name.split('/')[-1]
    dataset_name_path = dataset_name.split('/')[-1]
    Path('data').mkdir(exist_ok=True, parents=True)
    data_dir = Path(f'data/{dataset_name_path}_{model_name_path}')

    if write_sae_acts or write_holistic_acts or write_records:
        data_dir.mkdir(exist_ok=True, parents=True)
        record_sample_params = RecordSampleParams(
            top_sample_n=top_sample_n,
            quantile_n=quantile_n,
            complementary_sample_n=complementary_sample_n,
            special_n=special_n,
        )

        print('Loading model')

        model = HookedSAETransformer.from_pretrained(
            model_name,
            device=DEVICE,
            dtype=torch.bfloat16,
        )
        model.device = DEVICE
        model.name_or_path = model_name
        
        print('Loading dataset')
        
        if not os.path.exists(data_dir / 'tokens.pt'):
            dataset_tokens = construct_tokenized_dataset(
                dataset_name=dataset_name,
                tokenizer=model.tokenizer,
                sae_name=sae_name,
                sae_id_layer_format_string=sae_id_layer_format_string,
                max_dataset_size=max_dataset_size,
                max_seq_length=max_seq_length,
            )
            torch.save(dataset_tokens, data_dir / f'tokens.pt')
        else:
            dataset_tokens = torch.load(
                data_dir / f'tokens.pt', map_location=DEVICE, weights_only=True
            )
        
        print('Done loading dataset')
        
        if layers == 'all':
            layers_list = range(0, model.cfg.n_layers)
        elif layers == 'even':
            layers_list = range(0, model.cfg.n_layers, 2)
        elif layers == 'odd':
            layers_list = range(1, model.cfg.n_layers, 2)
        else:
            layers_list = [int(l) for l in layers.split(',')]
    
    if write_sae_acts or write_holistic_acts:
        print('Loading SAEs...')
        saes_list: list[SAE] = []
        for layer in tqdm.tqdm(layers_list):
            sae_config, _, _ = SAE.from_pretrained(
                release=sae_name,
                sae_id=sae_id_layer_format_string.format(layer=layer),
                device=DEVICE,
            )
            saes_list.append(sae_config)
    
    if write_records:
        print('Loading SAE configs...')
        sae_configs: list[tuple[str, int]] = []
        for layer in tqdm.tqdm(layers_list):
            sae_config, _, _ = SAE.from_pretrained(
                release=sae_name,
                sae_id=sae_id_layer_format_string.format(layer=layer),
                device=DEVICE,
            )
            sae_configs.append((sae_config.cfg.hook_name, sae_config.cfg.hook_layer))
            del sae_config
    
    if write_sae_acts:
        print('Executing SAE analysis...')
        (data_dir / 'sae_acts').mkdir(exist_ok=True, parents=True)
        
        write_sae_activations(
            model=model,
            saes_list=saes_list,
            dataset_tokens=dataset_tokens,
            max_features=max_features,
            data_dir=data_dir,
        )

    if write_holistic_acts:
        print('Writing holistic activations...')
        (data_dir / 'holistic_acts').mkdir(exist_ok=True, parents=True)
        write_holistic_activations(
            model=model,
            saes_list=saes_list,
            dataset_tokens=dataset_tokens,
            data_dir=data_dir,
        )

    if write_records:
        print('Writing feature expression records...')
        for sae_config in tqdm.tqdm(sae_configs, desc='Layer ind'):
            write_expression_records(
                model=model,
                dataset_tokens=dataset_tokens,
                hook_name=sae_config[0],
                hook_layer=sae_config[1],
                data_dir=data_dir,
                record_sample_params=record_sample_params,
                max_features=max_features,
            )
    
    artifact_name = f'{dataset_name_path}_{model_name_path}'
    
    if upload_wandb:
        print('Uploading results to W&B...')
        upload_to_wandb(
            data_root_dir=Path('data'),
            subdir_name=artifact_name,
        )
    
    if download_wandb:
        print('Downloading results from W&B...')
        download_from_wandb(
            data_root_dir=Path('data'),
            subdir_name=artifact_name,
        )


if __name__ == '__main__':
    main()
