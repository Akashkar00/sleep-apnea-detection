from src.data_loader import all_base_records, learning_records, load_config, official_test_records


def test_record_counts():
    cfg = load_config()
    assert len(learning_records(cfg)) == 35
    assert len(official_test_records(cfg)) == 35
    assert len(all_base_records(cfg)) == 70


def test_no_overlap_between_learning_and_test():
    cfg = load_config()
    assert set(learning_records(cfg)).isdisjoint(set(official_test_records(cfg)))


def test_excluded_records_not_in_base_records():
    cfg = load_config()
    base = set(all_base_records(cfg))
    excluded = set(cfg["data"]["excluded_related_records"])
    assert base.isdisjoint(excluded)
