from reproduction.summarize_output_stopping import selected_checkpoint


def test_failed_holdout_does_not_move_selected_calibration_endpoint():
    def row(iteration, calibration, holdout):
        return {'iteration':iteration, 'calibration_quality':{'coarse':calibration},
                'holdout_quality':{'coarse':holdout}, 'calibration_path_seconds':iteration,
                'with_selected_confirmation_seconds':iteration+1,
                'with_saved_oracle_preparation_seconds':iteration+2}
    result=selected_checkpoint({'checkpoints':[row(1,False,True),row(2,True,False),row(3,True,True)]},'coarse')
    assert result['iteration']==2 and not result['holdout_confirmed']


def test_no_calibration_pass_retains_failure():
    assert selected_checkpoint({'checkpoints':[
        {'calibration_quality':{'strict':False}, 'holdout_quality':{'strict':True}}]},'strict') is None
