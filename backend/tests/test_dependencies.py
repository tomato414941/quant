from skfolio import __version__
from skfolio.optimization import MeanRisk, RiskBudgeting


def test_skfolio_is_available() -> None:
    assert __version__ == "0.16.1"
    assert MeanRisk.__name__ == "MeanRisk"
    assert RiskBudgeting.__name__ == "RiskBudgeting"
