import pytest
from agent3.agent import run_finance_agent
FINANCE_AGENT_TEST_CASES = [
    {
        "query": "How do I file a reimbursement?",
        "expect_internal_tool": True,
        "expect_web_tool": False,
        "must_contain": ["reimbursement"],
        "description": "Internal finance policy lookup",
    },
    {
        "query": "Where can I find last month's budget report?",
        "expect_internal_tool": True,
        "expect_web_tool": False,
        "must_contain": ["budget"],
        "description": "Internal finance document retrieval",
    },
    {
        "query": "What is the minimum wage in California?",
        "expect_internal_tool": True,
        "expect_web_tool": True,
        "must_contain": ["minimum wage"],
        "description": "External public finance data",
    },
    {
        "query": "How do I configure VPN?",
        "expect_internal_tool": False,
        "expect_web_tool": False,
        "expect_rejection": True,
        "description": "Finance agent must reject IT queries",
    },
]


@pytest.mark.parametrize("case", FINANCE_AGENT_TEST_CASES)
def test_finance_agent(case):
    response, trace = run_finance_agent(case["query"], return_trace=True)

    if case.get("expect_rejection"):
        assert "not a finance-related" in response.lower()
        return

    if case["expect_internal_tool"]:
        assert "finance_internal_policy_search" in trace["tools_used"]

    if case["expect_web_tool"]:
        assert "external_web_research" in trace["tools_used"]

    for keyword in case.get("must_contain", []):
        assert keyword.lower() in response.lower()
