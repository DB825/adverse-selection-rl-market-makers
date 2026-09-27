"""Provenance-aware wrappers around the unchanged scientific implementations."""
import json
from pathlib import Path
import platform

from .run_store import atomic_json, file_hash, identity, run_job, verify_run
from .train import load_config, train, _dependency_versions
from .evaluate import evaluate


TRAIN_SOURCES = ('src/train.py', 'src/environment.py', 'src/customer_model.py', 'src/accounting.py',
                 'src/bayes_filter.py', 'src/wrappers.py', 'src/fast_recurrent.py', 'src/artifacts.py',
                 'src/run_store.py', 'src/managed_runs.py')
EVAL_SOURCES = TRAIN_SOURCES + ('src/evaluate.py', 'src/behavior.py', 'src/interventions.py')


def provenance(paths):
    return {'source_sha256': {p: file_hash(p) for p in paths},
            'dependencies': _dependency_versions(), 'python': platform.python_version()}


def training_spec(config, policy, seed):
    config = load_config(config)
    return {'stage': 'train', 'schema_version': 1, 'policy': policy, 'seed': int(seed),
            'config': config, **provenance(TRAIN_SOURCES)}


def managed_train(config, policy='recurrent', seed=11, output='results/managed/train'):
    config = load_config(config)
    spec = training_spec(config, policy, seed)
    def compute(work):
        model, metadata = train(config, policy, seed, work)
        model.logger.close()
        metadata.update(checkpoint='model.zip', checkpoint_sha256=file_hash(work / 'model.zip'),
                        run_id=identity(spec), artifact_schema_version=1)
        return metadata
    return run_job(output, spec, compute, 'metadata.json', ('model.zip', 'config.json', 'progress.csv'))


def managed_evaluate(policy, checkpoint=None, output='results/managed/evaluate', **settings):
    defaults = dict(episodes=512, seed_start=1_000_000, deterministic=False, shift=False,
                    correct_filter=False, collect=False, batch_size=128,
                    inventory_penalty=.001, horizon=64, fee=0., shift_control=False)
    unknown = set(settings) - set(defaults)
    if unknown:
        raise ValueError(f'Unknown evaluation options: {sorted(unknown)}')
    defaults.update(settings)
    if defaults['episodes'] <= 0 or defaults['batch_size'] <= 0 or defaults['horizon'] <= 0:
        raise ValueError('Evaluation sizes must be positive')
    digest = file_hash(checkpoint) if checkpoint else None
    trained_id = None
    if checkpoint:
        parent = Path(checkpoint).parent
        if (parent / 'run_manifest.json').exists():
            training = verify_run(parent)
            if training['spec'].get('stage') != 'train' or training['spec'].get('policy') != policy:
                raise ValueError('Checkpoint policy does not match evaluation')
            trained_id = training['run_id']
            config = json.loads((parent / 'config.json').read_text(encoding='utf-8'))
            env = config.get('environment', {})
            if any(k in env for k in ('regimes', 'prior')):
                raise ValueError('Managed evaluator currently supports the documented default regime support only')
            for key, default in [('horizon', 64), ('inventory_penalty', .001), ('fee', 0.)]:
                if defaults[key] != env.get(key, default):
                    raise ValueError(f'Evaluation {key} differs from training; require a separately designed experiment')
    spec = {'stage': 'evaluate', 'schema_version': 1, 'policy': policy, 'settings': defaults,
            'checkpoint_sha256': digest, 'training_run_id': trained_id, **provenance(EVAL_SOURCES)}
    def compute(work):
        result = evaluate(policy=policy, checkpoint=checkpoint, output=work, **defaults)
        if checkpoint and file_hash(checkpoint) != digest:
            raise ValueError('Checkpoint changed during evaluation')
        result.update(checkpoint=('sha256:' + digest) if digest else None, checkpoint_sha256=digest,
                      training_run_id=trained_id, run_id=identity(spec), artifact_schema_version=1)
        return result
    required = ('episodes.csv', 'activations.npz', 'activation_schema.json') if defaults['collect'] else ('episodes.csv',)
    return run_job(output, spec, compute, 'summary.json', required)
