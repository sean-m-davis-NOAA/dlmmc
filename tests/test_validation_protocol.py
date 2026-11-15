from dlmmc.validation_protocol import original_indicator_first_look


def test_keep_original_individual_looks_and_allow_new_tensor_studies():
    for panel in ['p05-lat01','p10-lat04']:
        cert=dict(panel=panel,formal_look=1,sampling_directory=f'experiments/affine-batch/numba-individual-{panel}/float64')
        assert original_indicator_first_look(cert)
        assert not original_indicator_first_look(dict(cert,formal_look=2))
        for study in ['numba-all-five','numba-singleton-'+panel,'numba-permuted','numba-chunk-first2']:
            assert not original_indicator_first_look(dict(cert,sampling_directory=f'experiments/affine-batch/{study}/float64'))
    assert original_indicator_first_look(dict(panel='p05-lat01',formal_look=1,sampling_directory='experiments/affine-pilot/float64'))
    assert not original_indicator_first_look(dict(panel='p16-lat07',formal_look=1,sampling_directory='experiments/affine-batch/numba-individual-p16-lat07/float64'))
