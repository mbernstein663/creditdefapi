from src.profit import approve, annualized_profit_rate, expected_npv_profit, expected_profit, expected_return


def test_expected_profit_and_return_match_hand_calculation():
    profit = expected_profit([0.10], [1000], [12], [100], lgd=1.0)[0]

    assert profit == 80
    assert expected_return([profit], [1000])[0] == 0.08
    assert approve([profit])[0]


def test_expected_profit_supports_good_profit_haircut():
    profit = expected_profit([0.0], [1000], [12], [100], lgd=1.0, good_profit_haircut=0.5)[0]

    assert profit == 100


def test_approval_rule_rejects_negative_expected_profit():
    profit = expected_profit([0.50], [1000], [12], [100], lgd=1.0)[0]

    assert profit == -400
    assert not approve([profit])[0]


def test_required_return_rule_is_strict():
    assert not approve([80], [0.08], required_return=0.08)[0]
    assert not approve([80], [0.08], required_return=0.09)[0]


def test_npv_profit_discounting_is_more_conservative_than_simple_ev():
    npv = expected_npv_profit([0.0], [1000], [12], [100], lgd=1.0, annual_discount_rate=0.08)[0]
    simple = expected_profit([0.0], [1000], [12], [100], lgd=1.0)[0]

    assert npv < simple
    assert annualized_profit_rate([npv], [1000], [12])[0] == npv / 1000
