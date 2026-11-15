import hashlib
import json
from pathlib import Path

import pytest
from dlmmc.validation_artifacts import verify_cache_provenance, verify_reference_probability_sources


def make_cache(root):
    cert = {'panel': 'p05', 'retained_steps': 8000}
    (root/'certificate.json').write_text(json.dumps(cert))
    (root/'model.py').write_text('original equations')
    sources = {name:hashlib.sha256((root/name).read_bytes()).hexdigest()
               for name in ['certificate.json', 'model.py']}
    folder = root/'cache'; folder.mkdir()
    config = dict(certificate='certificate.json', source_hashes=sources, model='m', panel='p05')
    write_config(folder, config)
    return folder, config, cert


def write_config(folder, config):
    path = folder/'config.json'; path.write_text(json.dumps(config))
    (folder/'complete.json').write_text(json.dumps({'config_sha256':hashlib.sha256(path.read_bytes()).hexdigest()}))


def test_valid_cache_and_wrong_window_or_panel(tmp_path):
    folder, config, cert = make_cache(tmp_path)
    assert verify_cache_provenance(tmp_path,folder,certificate=cert,model='m',panel='p05') == config
    with pytest.raises(ValueError,match='different parameter window'):
        verify_cache_provenance(tmp_path,folder,certificate=dict(cert,retained_steps=16000))
    with pytest.raises(ValueError,match='identity'):
        verify_cache_provenance(tmp_path,folder,panel='p10')


def test_changed_source_and_completion_are_rejected(tmp_path):
    folder, config, _ = make_cache(tmp_path)
    (tmp_path/'model.py').write_text('changed equations')
    with pytest.raises(ValueError,match='source changed'):
        verify_cache_provenance(tmp_path,folder)
    (folder/'config.json').write_text(json.dumps(dict(config,panel='p10')))
    with pytest.raises(ValueError,match='completion/config'):
        verify_cache_provenance(tmp_path,folder)


@pytest.mark.parametrize('name',['../outside','/outside'])
def test_non_project_sources_rejected_before_read(tmp_path,name):
    folder, config, _ = make_cache(tmp_path)
    config['source_hashes'] = {name:'unused'}; write_config(folder,config)
    with pytest.raises(ValueError,match='project-relative'):
        verify_cache_provenance(tmp_path,folder)


def test_certificate_must_be_in_source_manifest(tmp_path):
    folder, config, cert = make_cache(tmp_path)
    del config['source_hashes']['certificate.json'];write_config(folder,config)
    with pytest.raises(ValueError,match='different parameter window'):
        verify_cache_provenance(tmp_path,folder,certificate=cert)


def test_prefix_bridge_is_verified(tmp_path):
    folder, config, _ = make_cache(tmp_path)
    bridge = tmp_path/'bridge.json';bridge.write_text('qualified')
    config['probability_prefix_bridge'] = dict(path='bridge.json',sha256=hashlib.sha256(bridge.read_bytes()).hexdigest())
    write_config(folder,config);verify_cache_provenance(tmp_path,folder)
    bridge.write_text('changed qualification')
    with pytest.raises(ValueError,match='qualification changed'):
        verify_cache_provenance(tmp_path,folder)


def test_reference_probability_cannot_use_other_chain_or_data():
    state = dict(reference='reference/p05',source_hashes={'reference/p05/data.json':'data','reference/p05/chain.csv':'draws','model.py':'model'})
    probability = dict(source_hashes=dict(state['source_hashes']))
    verify_reference_probability_sources(state,probability)
    probability['source_hashes']['reference/p05/chain.csv'] = 'other draws'
    with pytest.raises(ValueError,match='inputs differ'):
        verify_reference_probability_sources(state,probability)
