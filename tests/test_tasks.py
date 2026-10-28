from grove.tasks import all_tasks, check_task, heldout_tasks, train_tasks


def test_counts():
    assert len(train_tasks()) == 4
    assert len(heldout_tasks()) == 10
    assert len(all_tasks()) == 14


def test_reference_solutions_pass():
    for t in all_tasks():
        res = check_task(t, t.reference_code)
        assert res.passed, f"{t.id}: reference failed\n{res.log}"


def test_naive_solutions_fail_except_controls():
    for t in all_tasks():
        res = check_task(t, t.naive_code)
        if t.family == "control":
            assert res.passed, f"{t.id}: control should pass naively"
        else:
            assert not res.passed, f"{t.id}: naive solution unexpectedly passed (quirk not biting)"


def test_failure_logs_contain_a_sample():
    for t in train_tasks():
        res = check_task(t, t.naive_code)
        assert "SAMPLE OF THE RAW INPUT" in res.log
