"""Reproduce the single-factor config and paired-initialization checks."""
from pathlib import Path
import json

from scripts.analyze_entropy import validate_configs
from scripts.audit_followup import parameter_digest, differences
from src.artifacts import source_hashes, write_json
from src.train import load_config, make_model


def main():
    control=load_config('configs/followup_budget.yaml')
    treatment=load_config('configs/entropy_001.yaml')
    validate_configs(control,treatment)
    rows=[]
    for seed in (11,12,13,14,15):
        a=make_model(control,'recurrent',seed)
        b=make_model(treatment,'recurrent',seed)
        first,second=parameter_digest(a),parameter_digest(b)
        if first!=second or a.market_environment_seeds!=b.market_environment_seeds:
            raise ValueError(f'Unpaired initialization or streams: {seed}')
        coefficients=[a.ent_coef,b.ent_coef]
        if coefficients!=[0.,.01] or a.gae_lambda!=.95 or b.gae_lambda!=.95:
            raise ValueError('Incorrect entropy or GAE setting')
        rows.append({'seed':seed,'parameter_sha256':first,'identical_initialization':True,
                     'environment_seeds':a.market_environment_seeds,'entropy_coefficients':coefficients,'gae_lambda':.95})
        a.get_env().close();b.get_env().close()
    payload={'config_differences':differences(control,treatment),'initializations':rows,
             'source_hashes':source_hashes(),'status':'passed'}
    path=Path('results/entropy_preflight.json')
    if path.exists():
        old=json.loads(path.read_text())
        for key in ('config_differences','initializations','status'):
            if old.get(key)!=payload[key]:
                raise ValueError('Existing preflight differs; use a separate experiment')
        if old['source_hashes']['configs/entropy_001.yaml']!=payload['source_hashes']['configs/entropy_001.yaml']:
            raise ValueError('Treatment configuration source differs')
        print('Preflight reproduced; preserving the original pre-training record')
    else:
        write_json(path,payload)
        print('Preflight passed for all five paired initializations')


if __name__=='__main__':
    main()
