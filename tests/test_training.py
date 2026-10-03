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
    fake_tokenizer = SimpleNamespace(get_vocab_size=lambda: 128, get_bos_token_id=lambda: 127)
    monkeypatch.setattr(tokenizer, 'get_tokenizer', lambda: fake_tokenizer)
    monkeypatch.setattr(checkpoints, 'get_tokenizer', lambda: fake_tokenizer)
    monkeypatch.setattr(tokenizer, 'get_token_bytes', lambda **_: torch.ones(128, dtype=torch.int32))
    monkeypatch.setattr(optim, 'adamw_step_fused', optim.adamw_step_fused._torchdynamo_orig_callable)
    monkeypatch.setattr(optim, 'muon_step_fused', optim.muon_step_fused._torchdynamo_orig_callable)

    def batches(tokenizer, batch_size, sequence_len, **kwargs):
        gen = torch.Generator().manual_seed(0)
        while True:
            row = torch.randint(0, 127, (batch_size, sequence_len + 1), generator=gen)
            x, y = row[:, :-1].contiguous(), row[:, 1:].contiguous()
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


@pytest.mark.parametrize('mtp', [False, True])
@pytest.mark.parametrize('attention', ['gqa', 'mla'])
def test_training_save_resume_and_inference(offline_training, attention, mtp):
    run, directory = offline_training
    arch = [f'--attention-type={attention}', '--q-lora-rank=16', '--kv-lora-rank=16',
            '--qk-nope-head-dim=12', '--qk-rope-head-dim=8', '--v-head-dim=10']
    if mtp:
        arch += ['--mtp', '--mtp-loss-weight=0.2']
    result = run(*arch)
    assert result['step'] == 2
    assert result['optimizer'].memory_efficient
    assert checkpoints.find_last_step(directory) == 2
    with open(directory / 'meta_000002.json') as f:
        meta = json.load(f)
    assert meta['num_iterations'] == 2
    assert meta['world_size'] == 1
    assert meta['model_config']['n_routed_experts'] == 4
    assert meta['user_config']['loss_chunk_size'] == 8
    assert meta['model_config']['mtp_enabled'] == mtp
    if mtp:
        assert meta['model_config']['mtp_bos_token_id'] == 127
        assert result['train_mtp_f'] > 0
    loaded, _, _ = checkpoints.build_model(directory, 2, torch.device('cpu'), 'eval')
    assert not any(p.is_meta for p in loaded.parameters())
    assert not loaded.cos.is_meta
    result['orig_model'].eval()
    x = torch.tensor([[1, 2, 3, 4]])
    with torch.no_grad():
        torch.testing.assert_close(loaded(x), result['orig_model'](x))
    resumed = run(*arch, '--resume-from-step=2', '--num-iterations=3')
    assert resumed['step'] == 3
    assert checkpoints.find_last_step(directory) == 3


def test_resume_rejects_past_horizon_and_changed_buckets(offline_training):
    run, _ = offline_training
    run()
    with pytest.raises(ValueError, match='exceeds'):
        run('--resume-from-step=2', '--num-iterations=1')
    with pytest.raises(ValueError, match='muon-bucket-mb'):
        run('--resume-from-step=2', '--muon-bucket-mb=0')


def test_sft_inherits_and_saves_trained_mtp(offline_training, monkeypatch):
    from nanochat import loss_eval
    from tasks import smoltalk, mmlu, gsm8k
    run, base_directory = offline_training
    result = run('--mtp')
    before = result['orig_model'].mtp.fuse.weight.detach().clone()
    fake_tokenizer = tokenizer.get_tokenizer()
    fake_tokenizer.render_conversation = lambda conversation, max_tokens=2048: ([127, 1, 2, 3, 4], [0, 0, 0, 1, 1])

    class TinyTask:
        def __init__(self, *args, **kwargs):
            pass
        def __len__(self):
            return 32
        def __getitem__(self, index):
            return {'messages': [{'role': 'user', 'content': 'q'}, {'role': 'assistant', 'content': 'a'}]}

    monkeypatch.setattr(smoltalk, 'SmolTalk', TinyTask)
    monkeypatch.setattr(mmlu, 'MMLU', TinyTask)
    monkeypatch.setattr(gsm8k, 'GSM8K', TinyTask)
    monkeypatch.setattr(loss_eval, 'evaluate_bpb', lambda *args: 0.5)
    monkeypatch.setattr(sys, 'argv', ['chat_sft', '--device-type=cpu', '--no-compile', '--model-tag=unit',
                                     '--num-iterations=2', '--chatcore-every=-1', '--eval-every=-1',
                                     '--warmdown-ratio=0', '--mtp-loss-weight=0.3'])
    trained = runpy.run_module('scripts.chat_sft', run_name='__main__')
    # grad_accum_steps=2 here: num_iterations counts optimizer steps, not generator yields
    assert trained['grad_accum_steps'] == 2
    assert trained['step'] == 2
    assert trained['orig_model'].config.mtp_enabled
    assert trained['orig_model'].config.mtp_loss_weight == 0.3
    assert trained['orig_model'].loss_chunk_size == 8
    assert trained['optimizer'].memory_efficient
    assert not torch.equal(before, trained['orig_model'].mtp.fuse.weight)
    directory = base_directory.parent.parent / 'chatsft_checkpoints' / 'unit'
    step = checkpoints.find_last_step(directory)
    loaded, _, meta = checkpoints.build_model(directory, step, torch.device('cpu'), 'eval')
    assert meta['model_config']['mtp_enabled']
    assert meta['model_config']['mtp_loss_weight'] == 0.3
    trained['orig_model'].eval()
    with torch.no_grad():
        idx = torch.tensor([[1, 2, 3]])
        hidden = loaded.forward_hidden(idx)
        torch.testing.assert_close(loaded.mtp_logits(hidden[:, -1:], idx[:, -1:]),
                                   trained['orig_model'].mtp_logits(hidden[:, -1:], idx[:, -1:]))


class _TinyChatTask:
    def __init__(self, *args, length=6, **kwargs):
        self.length = length
    def __len__(self):
        return self.length
    def __getitem__(self, index):
        return {'messages': [{'role': 'user', 'content': 'q'}, {'role': 'assistant', 'content': 'a'}]}


def _patch_sft_tasks(monkeypatch):
    from nanochat import loss_eval
    from tasks import smoltalk, mmlu, gsm8k
    monkeypatch.setattr(smoltalk, 'SmolTalk', _TinyChatTask)
    monkeypatch.setattr(mmlu, 'MMLU', _TinyChatTask)
    monkeypatch.setattr(gsm8k, 'GSM8K', _TinyChatTask)
    monkeypatch.setattr(loss_eval, 'evaluate_bpb', lambda *args: 0.5)


def test_sft_crops_conversations_longer_than_a_row(offline_training, monkeypatch):
    run, _ = offline_training
    run()
    seen_limits = []
    long_ids, long_mask = [127] + list(range(1, 40)), [0] * 10 + [1] * 30

    def render(conversation, max_tokens=2048):
        seen_limits.append(max_tokens)
        return long_ids[:max_tokens], long_mask[:max_tokens]

    tokenizer.get_tokenizer().render_conversation = render
    _patch_sft_tasks(monkeypatch)
    # Full-epoch mode: before the fix a 40-token conversation never fit a 17-token row and
    # the packer padded forever. Now it is cropped to the row and the epoch terminates.
    monkeypatch.setattr(sys, 'argv', ['chat_sft', '--device-type=cpu', '--no-compile', '--model-tag=unit',
                                     '--num-iterations=-1', '--chatcore-every=-1', '--eval-every=-1',
                                     '--mmlu-epochs=0', '--gsm8k-epochs=0', '--load-optimizer=0'])
    trained = runpy.run_module('scripts.chat_sft', run_name='__main__')
    row_capacity = trained['args'].max_seq_len + 1
    assert set(seen_limits) == {row_capacity + 1}
    assert trained['truncated_conversations']['train'] >= 6
    assert 1 <= trained['step'] <= 4


def test_sft_rejects_zero_iterations(offline_training, monkeypatch):
    run, _ = offline_training
    run()
    tokenizer.get_tokenizer().render_conversation = lambda conversation, max_tokens=2048: ([127, 1, 2], [0, 1, 1])
    _patch_sft_tasks(monkeypatch)
    monkeypatch.setattr(sys, 'argv', ['chat_sft', '--device-type=cpu', '--no-compile', '--model-tag=unit',
                                     '--num-iterations=0', '--chatcore-every=-1', '--eval-every=-1'])
    with pytest.raises(ValueError, match='num-iterations'):
        runpy.run_module('scripts.chat_sft', run_name='__main__')


@pytest.mark.parametrize('attention', ['gqa', 'mla'])
def test_rl_inherits_memory_settings_and_architecture(offline_training, monkeypatch, attention):
    import shutil
    from tasks import gsm8k
    run, base_directory = offline_training
    arch = [f'--attention-type={attention}', '--q-lora-rank=16', '--kv-lora-rank=16',
            '--qk-nope-head-dim=12', '--qk-rope-head-dim=8', '--v-head-dim=10', '--mtp']
    base = run(*arch)
    before = {k: v.detach().clone() for k, v in base['orig_model'].state_dict().items()}
    sft_directory = base_directory.parent.parent / 'chatsft_checkpoints' / 'unit'
    shutil.copytree(base_directory, sft_directory) # stands in for an SFT checkpoint

    fake = tokenizer.get_tokenizer()
    specials = {'<|python_start|>': 120, '<|python_end|>': 121, '<|output_start|>': 122,
                '<|output_end|>': 123, '<|assistant_end|>': 124, '<|assistant_start|>': 125}
    fake.encode_special = specials.__getitem__
    fake.render_for_completion = lambda conversation: [127, 1, 2, 3, 125]
    fake.decode = lambda ids: ' '.join(map(str, ids))
    fake.encode = lambda text: [ord(c) % 100 for c in text]

    class TinyGSM8K(_TinyChatTask):
        def __init__(self, *args, **kwargs):
            super().__init__(length=2)
        def reward(self, conversation, text):
            return float(len(text) % 2)
        def evaluate(self, conversation, text):
            return len(text) % 2 == 0

    monkeypatch.setattr(gsm8k, 'GSM8K', TinyGSM8K)
    monkeypatch.setattr(sys, 'argv', ['chat_rl', '--device-type=cpu', '--model-tag=unit', '--device-batch-size=2',
                                     '--num-samples=2', '--examples-per-step=1', '--max-new-tokens=4',
                                     '--eval-every=100', '--eval-examples=1', '--save-every=100'])
    result = runpy.run_module('scripts.chat_rl', run_name='__main__')
    model = result['model']
    assert model.activation_checkpointing and model.loss_chunk_size == 8
    assert result['optimizer'].memory_efficient
    assert model.config.attention_type == attention and model.config.mtp_enabled
    after = model.state_dict()
    for name, value in before.items():
        if name.startswith('mtp.'):
            torch.testing.assert_close(after[name], value) # draft head is frozen during RL
    assert all(not p.requires_grad for p in model.mtp.parameters())
    assert any(not torch.equal(after[n], v) for n, v in before.items() if n.startswith('transformer.h.'))

    rl_directory = base_directory.parent.parent / 'chatrl_checkpoints' / 'unit'
    step = checkpoints.find_last_step(rl_directory)
    assert step == result['num_steps'] - 1
    loaded, _, meta = checkpoints.build_model(rl_directory, step, torch.device('cpu'), 'eval')
    assert meta['model_config']['attention_type'] == attention
    assert meta['model_config']['mtp_enabled'] and meta['user_config']['loss_chunk_size'] == 8
    model.eval()
    x = torch.tensor([[1, 2, 3, 4]])
    with torch.no_grad():
        torch.testing.assert_close(loaded(x), model(x))


def test_rl_rejects_sample_count_not_divisible_by_batch(offline_training, monkeypatch):
    monkeypatch.setattr(sys, 'argv', ['chat_rl', '--device-type=cpu', '--device-batch-size=3', '--num-samples=4'])
    with pytest.raises(ValueError, match='multiple of --device-batch-size'):
        runpy.run_module('scripts.chat_rl', run_name='__main__')


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
