from src.profit import approve, expected_profit, expected_return


def test_expected_profit_and_return_match_hand_calculation():
    profit = expected_profit([0.10], [1000], [12], [100], lgd=1.0)[0]

    assert profit == 80
    assert expected_return([profit], [1000])[0] == 0.08
    assert approve([profit])[0]


def test_approval_rule_rejects_negative_expected_profit():
    profit = expected_profit([0.50], [1000], [12], [100], lgd=1.0)[0]

    assert profit == -400
    assert not approve([profit])[0]
