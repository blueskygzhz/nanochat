"""Offline CPU integration tests for the large-model training path."""
import gc
import json
import runpy
import sys
from types import SimpleNamespace

import pytest
import torch

from nanochat import checkpoint_manager as checkpoints
from nanochat import common, dataloader, optim, tokenizer


@pytest.fixture
def offline_training(monkeypatch, tmp_path):
    monkeypatch.setenv('NANOCHAT_BASE_DIR', str(tmp_path))
    monkeypatch.setattr(common, 'compute_init', lambda _: (False, 0, 0, 1, torch.device('cpu')))
    monkeypatch.setattr(common, 'compute_cleanup', lambda: None)
    monkeypatch.setattr(gc, 'freeze', lambda: None)
    monkeypatch.setattr(gc, 'disable', lambda: None)
    fake_tokenizer = SimpleNamespace(get_vocab_size=lambda: 128)
    monkeypatch.setattr(tokenizer, 'get_tokenizer', lambda: fake_tokenizer)
    monkeypatch.setattr(checkpoints, 'get_tokenizer', lambda: fake_tokenizer)
    monkeypatch.setattr(tokenizer, 'get_token_bytes', lambda **_: torch.ones(128, dtype=torch.int32))
    monkeypatch.setattr(optim, 'adamw_step_fused', optim.adamw_step_fused._torchdynamo_orig_callable)
    monkeypatch.setattr(optim, 'muon_step_fused', optim.muon_step_fused._torchdynamo_orig_callable)

    def batches(tokenizer, batch_size, sequence_len, **kwargs):
        gen = torch.Generator().manual_seed(0)
        while True:
            x = torch.randint(0, 128, (batch_size, sequence_len), generator=gen)
            y = torch.randint(0, 128, (batch_size, sequence_len), generator=gen)
            yield x, y, dict(pq_idx=0, rg_idx=0, epoch=1)

    monkeypatch.setattr(dataloader, 'tokenizing_distributed_data_loader_with_state_bos_bestfit', batches)
    args = [
        'base_train', '--device-type=cpu', '--no-compile', '--depth=2', '--aspect-ratio=32',
        '--head-dim=32', '--n-kv-head=1', '--n-routed-experts=4', '--num-experts-per-tok=2',
        '--n-shared-experts=1', '--max-seq-len=16', '--device-batch-size=1', '--total-batch-size=32',
        '--activation-checkpointing', '--loss-chunk-size=8', '--muon-bucket-mb=1',
        '--eval-every=-1', '--core-metric-every=-1', '--sample-every=-1', '--warmup-steps=1',
        '--model-tag=unit', '--num-iterations=2',
    ]

    def run(*extra):
        monkeypatch.setattr(sys, 'argv', args + list(extra))
        return runpy.run_module('scripts.base_train', run_name='__main__')

    return run, tmp_path / 'base_checkpoints' / 'unit'


def test_training_save_resume_and_inference(offline_training):
    run, directory = offline_training
    result = run()
    assert result['step'] == 2
    assert result['optimizer'].memory_efficient
    assert checkpoints.find_last_step(directory) == 2
    with open(directory / 'meta_000002.json') as f:
        meta = json.load(f)
    assert meta['num_iterations'] == 2
    assert meta['world_size'] == 1
    assert meta['model_config']['n_routed_experts'] == 4
    assert meta['user_config']['loss_chunk_size'] == 8
    loaded, _, _ = checkpoints.build_model(directory, 2, torch.device('cpu'), 'eval')
    assert not any(p.is_meta for p in loaded.parameters())
    assert not loaded.cos.is_meta
    result['orig_model'].eval()
    x = torch.tensor([[1, 2, 3, 4]])
    with torch.no_grad():
        torch.testing.assert_close(loaded(x), result['orig_model'](x))
    resumed = run('--resume-from-step=2', '--num-iterations=3')
    assert resumed['step'] == 3
    assert checkpoints.find_last_step(directory) == 3


def test_resume_rejects_past_horizon_and_changed_buckets(offline_training):
    run, _ = offline_training
    run()
    with pytest.raises(ValueError, match='exceeds'):
        run('--resume-from-step=2', '--num-iterations=1')
    with pytest.raises(ValueError, match='muon-bucket-mb'):
        run('--resume-from-step=2', '--muon-bucket-mb=0')


def test_missing_training_shard_fails_instead_of_spinning(monkeypatch):
    monkeypatch.setattr(dataloader, 'list_parquet_files', lambda **_: ['validation.parquet'])
    with pytest.raises(ValueError, match='training parquet'):
        next(dataloader._document_batches('train', None, 8))


def test_interrupted_checkpoint_is_not_selected(monkeypatch, tmp_path):
    state = {'weight': torch.ones(2)}
    checkpoints.save_checkpoint(tmp_path, 1, state, {}, {'step': 1})
    real_save = torch.save

    def fail_optimizer(data, path):
        if 'optim_000002' in str(path):
            raise OSError('simulated interrupted write')
        real_save(data, path)

    monkeypatch.setattr(torch, 'save', fail_optimizer)
    with pytest.raises(OSError, match='interrupted'):
        checkpoints.save_checkpoint(tmp_path, 2, state, {}, {'step': 2})
    assert checkpoints.find_last_step(tmp_path) == 1
    assert not list(tmp_path.glob('*.tmp'))
    with pytest.raises(FileNotFoundError):
        checkpoints.load_checkpoint(tmp_path, 2, 'cpu', load_optimizer=True)


def test_incomplete_optimizer_manifest_is_rejected(tmp_path):
    checkpoints.save_checkpoint(tmp_path, 1, {'w': torch.ones(2)}, {}, {'step': 1})
    (tmp_path / 'optim_000001_rank0.pt').unlink()
    with pytest.raises(FileNotFoundError, match='Incomplete optimizer'):
        checkpoints.load_checkpoint(tmp_path, 1, 'cpu', load_optimizer=True)
    with pytest.raises(FileNotFoundError, match='complete'):
        checkpoints.find_last_step(tmp_path)
