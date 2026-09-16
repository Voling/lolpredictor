from synergy.ml.score import thin_warning


def test_a_profile_that_is_mostly_the_corpus_row_is_warned_about_by_name_share_and_games():
    # given
    name, position, games, evidence = "ethnvo#goat", "MIDDLE", 1, 0.12

    # when
    warning = thin_warning(name, position, games, evidence)

    # then
    assert warning is not None
    assert "ethnvo#goat" in warning and "12% their own evidence from 1 game" in warning
    assert "88% the MIDDLE corpus row" in warning


def test_a_profile_carried_mostly_by_its_own_games_raises_no_warning():
    # given
    evidence = 0.5

    # when
    warning = thin_warning("bblskibs#gotg", "TOP", 100, evidence)

    # then
    assert warning is None
    assert thin_warning("bblskibs#gotg", "TOP", 100, None) is None
