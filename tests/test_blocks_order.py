from synergy.cli import BLOCKS


def test_priority_builds_before_tendency_because_tendency_reads_its_track():
    # given
    order = list(BLOCKS)

    # when
    priority, tendency = order.index("priority"), order.index("tendency")

    # then
    assert priority < tendency
    assert set(order) == {"movement", "embedding", "orphans", "reaction", "priority", "tendency", "habit", "hinge"}
