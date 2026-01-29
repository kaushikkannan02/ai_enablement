# """
# Supervisor Agent Evaluation (Single File)

# Evaluates:
# - Correctness
# - Latency
# - Hallucination
# - Tool usage success

# Assumes:
# run_supervisor_agent(query, return_trace=True) -> (response, trace)

# trace must contain:
# {
#   "classification": "it" | "finance",
#   "routed_to": "agent_2" | "agent_3",
#   "trajectory": list[str],
#   "tools_used": list[str]
# }
# """

# import time
# from typing import Dict, Any, List

# # --------------------------------------------------
# # IMPORT YOUR SUPERVISOR AGENT
# # --------------------------------------------------
# from agent1 import run_supervisor


# # --------------------------------------------------
# # DATASET (AgentEval-style)
# # --------------------------------------------------
# DATASET = [
#     {
#         "input": "How do I configure VPN?",
#         "expected": {
#             "classification": "it",
#             "routed_to": "agent_2",
#             "expected_tool": "it_internal_docs",
#         },
#     },
#     {
#         "input": "How do I file a reimbursement?",
#         "expected": {
#             "classification": "finance",
#             "routed_to": "agent_3",
#             "expected_tool": "finance_internal_policy_search",
#         },
#     },
#     {
#         "input": "What is the minimum wage in California?",
#         "expected": {
#             "classification": "finance",
#             "routed_to": "agent_3",
#             "expected_tool": "external_web_research",
#         },
#     },
# ]


# # --------------------------------------------------
# # TARGET RUNNER (AgentEval target)
# # --------------------------------------------------
# def supervisor_target(example: Dict[str, Any]) -> Dict[str, Any]:
#     start = time.time()
#     response, trace = run_supervisor(
#         example["input"],
#         return_trace=True,
#     )
#     end = time.time()

#     return {
#         "response": response,
#         "classification": trace.get("classification"),
#         "routed_to": trace.get("routed_to"),
#         "trajectory": trace.get("trajectory", []),
#         "tools_used": trace.get("tools_used", []),
#         "latency_ms": int((end - start) * 1000),
#     }


# # --------------------------------------------------
# # EVALUATORS
# # --------------------------------------------------
# def eval_correctness(run: Dict, example: Dict) -> Dict:
#     expected = example["expected"]
#     passed = (
#         run["classification"] == expected["classification"]
#         and run["routed_to"] == expected["routed_to"]
#     )
#     print(run['classification'])
#     return {
#         "passed": passed,
#         "expected": expected,
#         "actual": {
#             "classification": run["classification"],
#             "routed_to": run["routed_to"],
#         },
#     }


# def eval_latency(run: Dict, _: Dict) -> Dict:
#     return {
#         "latency_ms": run["latency_ms"],
#     }


# def eval_hallucination(run: Dict, _: Dict) -> Dict:
#     """
#     Hallucination if:
#     - Agent answered without routing
#     - No classification step in trajectory
#     """
#     hallucinated = (
#         run["routed_to"] is None
#         or not any("classif" in step.lower() for step in run["trajectory"])
#     )

#     return {
#         "hallucinated": hallucinated,
#         "trajectory": run["trajectory"],
#     }


# def eval_tool_usage(run: Dict, example: Dict) -> Dict:
#     expected_tool = example["expected"]["expected_tool"]
#     used_tools = set(run["tools_used"])

#     return {
#         "passed": expected_tool in used_tools,
#         "expected_tool": expected_tool,
#         "used_tools": list(used_tools),
#     }


# # --------------------------------------------------
# # MAIN EVAL LOOP
# # --------------------------------------------------
# def run_agent_eval():
#     results: List[Dict[str, Any]] = []

#     for example in DATASET:
#         print(example)
#         run = supervisor_target(example)

#         result = {
#             "input": example["input"],
#             "correctness": eval_correctness(run, example),
#             "latency": eval_latency(run, example),
#             "hallucination": eval_hallucination(run, example),
#             "tool_usage": eval_tool_usage(run, example),
#         }
#         print(result)
#         results.append(result)

#     return results


# # --------------------------------------------------
# # REPORTING
# # --------------------------------------------------
# def print_report(results: List[Dict]):
#     total = len(results)
#     correct = sum(r["correctness"]["passed"] for r in results)
#     hallucinations = sum(r["hallucination"]["hallucinated"] for r in results)
#     tool_success = sum(r["tool_usage"]["passed"] for r in results)
#     avg_latency = sum(r["latency"]["latency_ms"] for r in results) / total

#     print("\n===== SUPERVISOR AGENT EVAL REPORT =====\n")
#     print(f"Total cases           : {total}")
#     print(f"Correct routing       : {correct}/{total}")
#     print(f"Tool usage success    : {tool_success}/{total}")
#     print(f"Hallucination rate    : {hallucinations}/{total}")
#     print(f"Average latency (ms)  : {avg_latency:.2f}\n")

#     print("----- Detailed Results -----")
#     for r in results:
#         print(f"\nInput: {r['input']}")
#         print(f"  Correct     : {r['correctness']['passed']}")
#         print(f"  Tools OK    : {r['tool_usage']['passed']}")
#         print(f"  Hallucinated: {r['hallucination']['hallucinated']}")
#         print(f"  Latency(ms) : {r['latency']['latency_ms']}")


# # --------------------------------------------------
# # ENTRYPOINT
# # --------------------------------------------------
# if __name__ == "__main__":
#     results = run_agent_eval()
#     print_report(results)

import json
import time
from pathlib import Path
from typing import List, Dict

from agentevals.trajectory.match import create_trajectory_match_evaluator

from agent1 import run_supervisor


# -----------------------------
# CONFIG
# -----------------------------
EVAL_CASES_DIR = Path("evals")

trajectory_evaluator = create_trajectory_match_evaluator(
    trajectory_match_mode="subset"
)


# -----------------------------
# HELPERS
# -----------------------------
def load_eval_cases() -> List[Dict]:
    cases = []
    for path in EVAL_CASES_DIR.rglob("*.json"):
        with open(path) as f:
            case = json.load(f)
            case["_file"] = str(path)
            cases.append(case)
    return cases


def build_outputs(trace: Dict) -> List[Dict]:
    """
    Convert supervisor trace → agentevals trajectory format
    """
    outputs = []

    # user message
    outputs.append({
        "role": "user",
        "content": trace["input"]
    })

    # supervisor steps as tool calls
    for step in trace["trajectory"]:
        if step.startswith("routed_to_"):
            outputs.append({
                "role": "assistant",
                "content": "",
                "tool_calls": [{
                    "function": {
                        "name": step,
                        "arguments": "{}"
                    }
                }]
            })

    return outputs


def build_reference(case: Dict) -> List[Dict]:
    reference = [
        {
            "role": "user",
            "content": case["input"]
        }
    ]

    for step in case["expected_trajectory"]:
        if step.startswith("routed_to_"):
            reference.append({
                "role": "assistant",
                "content": "",
                "tool_calls": [{
                    "function": {
                        "name": step,
                        "arguments": "{}"
                    }
                }]
            })

    return reference


# -----------------------------
# RUN EVAL
# -----------------------------
def run_eval():
    cases = load_eval_cases()
    results = []

    print(f"\n🧪 Running {len(cases)} agent eval cases...\n")

    for case in cases:
        start = time.perf_counter()

        response, trace = run_supervisor(
            case["input"],
            return_trace=True
        )

        latency = time.perf_counter() - start

        trace["input"] = case["input"]

        outputs = build_outputs(trace)
        reference = build_reference(case)
        print(outputs)
        print(reference)
        eval_result = trajectory_evaluator(
            outputs=outputs,
            reference_outputs=reference
        )

        passed = eval_result["score"] is True

        results.append({
            "file": case["_file"],
            "route": trace["route"],
            "expected_route": case["expected_route"],
            "latency_ms": round(latency * 1000, 2),
            "passed": passed,
        })

        status = "✅ PASS" if passed else "❌ FAIL"

        print(f"{status} | {case['_file']}")
        print(f"  Route: {trace['route']} (expected {case['expected_route']})")
        print(f"  Latency: {latency:.3f}s\n")

    return results


# -----------------------------
# MAIN
# -----------------------------
if __name__ == "__main__":
    results = run_eval()

    passed = sum(r["passed"] for r in results)
    total = len(results)

    print("\n====================")
    print(f"✅ Passed {passed}/{total} cases")
    print("====================\n")
