"""Audit and summarize the complete, paired, single-factor entropy experiment."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
import torch

from src.artifacts import source_hashes, write_json
from scripts.summarize_results import SEEDS, REFERENCES, summarize_family, _save_figure
from scripts.summarize_followup import load_learned, _read_checked, assert_paired, paired_difference, behavior_summaries
from scripts.make_followup_manifest import CORE_SOURCES, EVALUATION_SETTINGS, check_junit

ROOT = Path('results')
BUDGET, EPISODES, START = 300032, 2048, 11_000_000


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def require(value, message):
    if not value:
        raise ValueError(message)


def validate_configs(control, treatment):
    require(control['environment'] == treatment['environment'], 'Environment changed')
    a, b = dict(control['training']), dict(treatment['training'])
    require(a.pop('ent_coef') == 0 and b.pop('ent_coef') == .01, 'Wrong entropy contrast')
    require(a == b, 'Entropy must be the only training change')
    require(a['total_timesteps'] == BUDGET and a['seeds'] == list(SEEDS), 'Wrong budget or seeds')
    require(treatment['evaluation'] == {'episodes': EPISODES, 'seed': START}, 'Wrong economic cohort')
    domains = [treatment[k] for k in ('development', 'evaluation', 'analysis')]
    domains += [{'seed': s*10000, 'episodes': a['n_envs']} for s in SEEDS]
    for i, left in enumerate(domains):
        for right in domains[i+1:]:
            require(left['seed']+left['episodes'] <= right['seed'] or
                    right['seed']+right['episodes'] <= left['seed'], 'Seed domains overlap')


def verify_evaluation(folder, checkpoint, policy, expected_ids, settings):
    s = read(folder/'summary.json')
    require(s['policy'] == policy and Path(s['checkpoint']).resolve() == checkpoint.resolve(),
            f'Wrong evaluated checkpoint: {folder}')
    require(s['episodes'] == EPISODES and s['seed_start'] == START and
            s['seed_stop_exclusive'] == START+EPISODES, f'Wrong cohort metadata: {folder}')
    require(all(s.get(k) == v for k, v in EVALUATION_SETTINGS.items()), f'Changed evaluation settings: {folder}')
    frame = _read_checked(folder/'episodes.csv', expected_ids, settings)
    require(np.isclose(frame.objective.mean(), s['objective_mean'], atol=1e-10, rtol=1e-10),
            f'Summary differs from raw outcomes: {folder}')
    require('behavior' in s, f'Missing conditional behavior diagnostics: {folder}')
    return frame


def verify_training(folder, policy, seed, config):
    m = read(folder/'metadata.json')
    require(m['policy'] == policy and m['seed'] == seed and m['actual_steps'] == BUDGET
            and m['requested_steps'] == BUDGET, f'Wrong model identity or budget: {folder}')
    require(Path(m['checkpoint']).resolve() == (folder/'model.zip').resolve(), f'Wrong checkpoint path: {folder}')
    require(read(folder/'config.json') == config, f'Saved config mismatch: {folder}')
    require(m['environment_seeds'] == list(range(seed*10000,seed*10000+8)), f'Wrong streams: {folder}')
    require(all(m['source_hashes'][p] == digest(p) for p in CORE_SOURCES), f'Core code changed: {folder}')
    log = pd.read_csv(folder/'progress.csv')
    require((log['economics/episodes'].dropna()==16).all() and
            (log['rollout/ep_len_mean'].dropna()==64).all(), f'Complete-episode rollout assumption failed: {folder}')
    numeric = log.select_dtypes(include=[np.number])
    require(not np.isinf(numeric.to_numpy()).any(), f'Infinite training diagnostics: {folder}')
    # Logger blanks are expected; explicit nonfinite strings are not.
    raw = pd.read_csv(folder/'progress.csv', keep_default_na=False).astype(str).apply(lambda c:c.str.lower())
    m['audit_log_diagnostics']=check_logged_values(raw)
    return m


def check_logged_values(raw):
    """SB3 intentionally returns NaN explained variance for constant targets.

    Record this diagnostic exception; never whitelist NaN optimization metrics
    or an infinite explained variance. Historical return variance was not logged.
    """
    raw=raw.astype(str).apply(lambda c:c.str.strip().str.lower())
    bad=raw.isin(['nan','inf','+inf','-inf','infinity','-infinity'])
    name='train/explained_variance'
    undefined=int((raw[name]=='nan').sum()) if name in raw else 0
    if name in raw:
        bad.loc[raw[name]=='nan',name]=False
    require(not bad.any().any(), 'Unexpected nonfinite optimization or economic metrics')
    return {'undefined_explained_variance_rows':undefined,'other_nonfinite_logged_values':0,
            'interpretation':'NaN explained variance is allowed only in this diagnostic column. SB3 defines it for zero target variance; historical target variances were not archived, so counts alone do not reconstruct every cause.'}


def figures(families, comparisons, folder):
    colors = ['#126A79','#CB7540','#7766A5','#54884B','#A85778']
    fig, axes = plt.subplots(1,2,figsize=(11,4.7))
    for i, seed in enumerate(SEEDS):
        values = [families[k]['metrics']['objective']['per_seed_means'][i] for k in ('recurrent_ent0','recurrent_ent001')]
        axes[0].plot([0,1],values,'o-',color=colors[i],label=f'Seed {seed}')
    axes[0].set_xticks([0,1],['Entropy coefficient 0','Entropy coefficient .01'])
    axes[0].set_ylabel('Mean episode objective')
    axes[0].set_title('Paired training seeds; identical test episodes')
    axes[0].legend(frameon=False,fontsize=8,ncol=2)
    effect = comparisons['entropy_effect']
    for i,(ci,label) in enumerate([(effect['training_seed']['mean_t95_ci'],'Across training seeds'),
                                  (effect['episode_bootstrap_95_ci'],'Across test episodes')]):
        axes[1].errorbar(effect['mean'],i,xerr=np.abs(np.array(ci)-effect['mean']).reshape(2,1),fmt='o',capsize=5,color=colors[i])
    axes[1].set_yticks([0,1],['Training-seed CI','Episode CI'])
    axes[1].set_ylim(-.7,1.7)
    axes[1].axvline(0,color='#69757D',lw=.8)
    axes[1].set_xlabel('Paired objective difference: entropy .01 minus 0')
    axes[1].set_title('Separate conditional 95% intervals')
    for ax in axes:
        ax.grid(alpha=.15)
    fig.suptitle('Entropy comparison at a fixed 300,032-step budget',fontsize=14)
    fig.tight_layout(rect=(0,0,1,.94))
    paths=[_save_figure(fig,folder,'paired_entropy_effect')]
    names=['recurrent_ent0','recurrent_ent001','feedforward','history','belief','fixed_5','myopic']
    labels=['Recurrent, entropy 0','Recurrent, entropy .01','Current observation, entropy 0','History-8, entropy 0','Belief input, entropy 0','Fixed wide','Bayesian myopic']
    fig, axes=plt.subplots(1,2,figsize=(11.5,4.8),sharex=True,sharey=True)
    for y,name in enumerate(names):
        metric=families[name]['metrics']['objective']
        intervals=[metric['training_seed']['mean_t95_ci'] if metric['training_seed'] else None,metric['episode_bootstrap_95_ci']]
        for ax,ci,color in zip(axes,intervals,colors):
            if ci:
                ax.errorbar(metric['mean'],y,xerr=np.abs(np.array(ci)-metric['mean']).reshape(2,1),fmt='o',capsize=4,color=color)
            else:
                ax.plot(metric['mean'],y,'o',color='#69757D')
    axes[0].set_yticks(range(len(names)),labels);axes[0].invert_yaxis()
    for ax,title in zip(axes,['Training-seed 95% intervals','Episode-bootstrap 95% intervals']):
        ax.set_title(title);ax.axvline(0,color='#69757D',lw=.8);ax.set_xlabel('Mean episode objective');ax.grid(alpha=.15)
    fig.suptitle('Fresh-cohort context: only recurrent entropy was changed',fontsize=14)
    fig.tight_layout(rect=(0,0,1,.94))
    paths.append(_save_figure(fig,folder,'economic_context'))
    return paths


def run():
    control=yaml.safe_load(Path('configs/followup_budget.yaml').read_text())
    treatment=yaml.safe_load(Path('configs/entropy_001.yaml').read_text())
    validate_configs(control,treatment)
    preflight=read(ROOT/'entropy_preflight.json')
    require(preflight['status']=='passed' and [r['seed'] for r in preflight['initializations']]==list(SEEDS), 'Incomplete initialization audit')
    require(all(r['identical_initialization'] and r['entropy_coefficients']==[0.,.01] and r['gae_lambda']==.95 for r in preflight['initializations']), 'Initialization or entropy mismatch')
    require(preflight['source_hashes']['configs/entropy_001.yaml']==digest('configs/entropy_001.yaml'), 'Treatment config changed after preflight')
    tests=check_junit(ROOT/'entropy_tests.xml')
    require('test_entropy_experiment' in (ROOT/'entropy_tests.xml').read_text(), 'Missing entropy experiment tests')
    expected=pd.Index(np.arange(START,START+EPISODES),name='episode')
    settings=dict(EVALUATION_SETTINGS)
    controls,sources=load_learned(ROOT/'entropy_control_evaluation',expected,settings)
    old_manifest=read(ROOT/'followup_manifest.json')
    hashes={(r['policy'],r['seed']):r['checkpoint_sha256'] for r in old_manifest['completed_runs']}
    new={};audit=[];artifacts={}
    anchor=controls['recurrent'][11]
    for policy in controls:
        for seed in SEEDS:
            folder=ROOT/'followup_budget'/policy/f'seed_{seed}'
            evaluation=ROOT/'entropy_control_evaluation'/policy/f'seed_{seed}'
            verify_training(folder,policy,seed,control)
            verify_evaluation(evaluation,folder/'model.zip',policy,expected,settings)
            require(digest(folder/'model.zip')==hashes[policy,seed],f'Original checkpoint changed: {folder}')
            for p in (folder/'model.zip',folder/'metadata.json',evaluation/'episodes.csv',evaluation/'summary.json'):
                artifacts[str(p)]=digest(p)
    from sb3_contrib import RecurrentPPO
    for seed in SEEDS:
        folder=ROOT/'entropy_001'/'recurrent'/f'seed_{seed}'
        evaluation=ROOT/'entropy_treatment_evaluation'/'recurrent'/f'seed_{seed}'
        m=verify_training(folder,'recurrent',seed,treatment)
        require(m['source_hashes']['configs/entropy_001.yaml']==digest('configs/entropy_001.yaml'), 'Config source changed during training')
        old=read(ROOT/'followup_budget'/'recurrent'/f'seed_{seed}'/'metadata.json')
        require(m['architecture']==old['architecture'] and m['dependencies']==old['dependencies'], 'Architecture/dependencies changed')
        first_old=pd.read_csv(ROOT/'followup_budget'/'recurrent'/f'seed_{seed}'/'progress.csv').iloc[0]
        first_new=pd.read_csv(folder/'progress.csv').iloc[0]
        first_keys=['economics/objective_mean','economics/profit_mean','economics/inventory_penalty_total_mean']
        require(all(first_old[k]==first_new[k] for k in first_keys), f'First pre-update rollout differs: seed {seed}')
        new[seed]=verify_evaluation(evaluation,folder/'model.zip','recurrent',expected,settings)
        assert_paired(anchor,new[seed],f'entropy seed {seed}')
        # Check serialized algorithm settings, rather than trusting YAML alone.
        serialized=[]
        for root,coefficient in [('followup_budget',0.),('entropy_001',.01)]:
            model=RecurrentPPO.load(ROOT/root/'recurrent'/f'seed_{seed}'/'model.zip',device='cpu')
            require(all(torch.isfinite(t).all().item() for t in model.policy.state_dict().values()),
                    f'Nonfinite model parameters: {root}/{seed}')
            require(model.gae_lambda==.95 and model.rollout_buffer.gae_lambda==.95 and model.gamma==1 and model.ent_coef==coefficient and model.num_timesteps==BUDGET,
                    f'Serialized algorithm settings differ: {root}/{seed}')
            serialized.append({'gae_lambda':model.gae_lambda,'gamma':model.gamma,'entropy_coefficient':model.ent_coef,'steps':model.num_timesteps})
            del model
        audit.append({'seed':seed,'serialized_control_treatment':serialized,'training_seconds':m['elapsed_seconds'],
                      'log_diagnostics':m['audit_log_diagnostics'],'serialized_parameters_all_finite':True,
                      'first_preupdate_rollout_economics_identical':True,
                      'first_rollout_metrics':{k:float(first_new[k]) for k in first_keys}})
        sources.append(str(evaluation/'episodes.csv'))
        for p in (folder/'model.zip',folder/'metadata.json',folder/'config.json',evaluation/'episodes.csv',evaluation/'summary.json'):
            artifacts[str(p)]=digest(p)
    families={k:summarize_family(v) for k,v in controls.items() if k!='recurrent'}
    families['recurrent_ent0']=summarize_family(controls['recurrent'])
    families['recurrent_ent001']=summarize_family(new)
    refs={}
    for name in REFERENCES:
        path=ROOT/'entropy_control_evaluation'/name/'episodes.csv'
        require(read(path.with_name('summary.json'))['policy']==name, f'Wrong fixed/reference policy: {name}')
        refs[name]=_read_checked(path,expected,settings)
        assert_paired(anchor,refs[name],name)
        families[name]=summarize_family({'reference':refs[name]},trained=False)
        artifacts[str(path)]=digest(path);sources.append(str(path))
    comparisons={'entropy_effect':paired_difference(new,controls['recurrent'],label='entropy contrast')}
    comparisons.update({f'entropy_minus_{p}':paired_difference(new,controls[p],label=p) for p in ('feedforward','history','belief')})
    comparisons['entropy_minus_fixed_5']=paired_difference(new,refs['fixed_5'],label='fixed',reference=True)
    plt.rcParams.update({'font.family':'DejaVu Sans','axes.spines.top':False,'axes.spines.right':False})
    figure_paths=figures(families,comparisons,Path('reports/entropy_figures'))
    payload={'created_utc':datetime.now(timezone.utc).isoformat(),'protocol':'reports/entropy_protocol.md',
             'cohort':{'episodes':EPISODES,'seed_start':START},'training_seeds':list(SEEDS),'budget_per_seed':BUDGET,
             'families':families,'comparisons':comparisons,'sources':sources,'figures':figure_paths,
             'behavior_control':behavior_summaries(ROOT/'entropy_control_evaluation'),
             'behavior_treatment':behavior_summaries(ROOT/'entropy_treatment_evaluation'),
             'uncertainty':'Seed t4 intervals condition on this cohort. Whole-episode bootstrap intervals condition on the five fitted policy pairs. Neither is joint uncertainty; five-seed intervals are approximate.',
             'interpretation':'Only the recurrent Entropy comparison changes one training factor; baseline architecture comparisons are contextual. Same-seed coupling is not independent replication.'}
    write_json(ROOT/'entropy_summary.json',payload)
    rows=[]
    for kind,records in [('policy',{k:v['metrics']['objective'] for k,v in families.items()}),('comparison',comparisons)]:
        for name,m in records.items():
            rows.append({'kind':kind,'name':name,'mean':m['mean'],'episode_se':m['episode_only_se'],
                         'episode_ci':m['episode_bootstrap_95_ci'],'seed_ci':m['training_seed']['mean_t95_ci'] if m['training_seed'] else None})
    pd.DataFrame(rows).to_csv(ROOT/'entropy_summary.csv',index=False)
    for p in [ROOT/'entropy_preflight.json',ROOT/'entropy_tests.xml',ROOT/'entropy_summary.json',Path('reports/entropy_protocol.md'),Path('configs/entropy_001.yaml')]:
        artifacts[str(p)]=digest(p)
    write_json(ROOT/'entropy_manifest.json',{'completed_treatment_runs':5,'control_policy_runs':20,'audits':audit,
               'total_new_training_transitions':5*BUDGET,'tests':tests,'artifacts_sha256':artifacts,'source_hashes':source_hashes(),
               'dependency_lock_sha256':digest('requirements-lock.txt'),'settings':'Only recurrent entropy coefficient changed; GAE .95 retained',
               'provenance_limit':'Evaluation summaries record checkpoint paths. Current hashes are archived; treatment evaluations did not independently log checkpoint hashes at collection time.'})
    print(json.dumps({'entropy_effect':comparisons['entropy_effect'],'figures':figure_paths},indent=2))
    return payload


if __name__=='__main__':
    run()
