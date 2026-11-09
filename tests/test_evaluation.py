"""Held-out subject aggregation and benchmark ablation metrics."""
import pytest


def test_subject_level_scores_do_not_count_duplicate_records_as_people():
    from pd_vqa.evaluation import classification_metrics
    rows=[{"subject_id":"a","label":0,"probabilities":[.8,.2]},
          {"subject_id":"a","label":0,"probabilities":[.6,.4]},
          {"subject_id":"b","label":1,"probabilities":[.2,.8]}]
    result=classification_metrics(rows,["healthy","pd"])
    assert result["subjects"] == 2
    assert result["accuracy"] == 1.
    assert result["auroc"] == 1.


def test_auc_assigns_half_credit_to_ties():
    from pd_vqa.evaluation import classification_metrics
    result=classification_metrics([{"subject_id":"a","label":0,"probabilities":[.5,.5]},
                                   {"subject_id":"b","label":1,"probabilities":[.5,.5]}],["h","p"])
    assert result["auroc"] == .5


def test_subject_inconsistent_ground_truth_rejected():
    from pd_vqa.evaluation import classification_metrics
    with pytest.raises(ValueError):
        classification_metrics([{"subject_id":"a","label":0,"probabilities":[.8,.2]},
                                {"subject_id":"a","label":1,"probabilities":[.2,.8]}],["h","p"])
