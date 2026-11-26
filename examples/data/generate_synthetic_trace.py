"""Generate a realistic 180-step unstructured Deep Research trajectory with planted failures."""

from __future__ import annotations

import json
from pathlib import Path

from trace_tree_eval.schema import RubricItem, TraceStep, Trajectory


def build_synthetic_180_step_trace() -> tuple[Trajectory, list[RubricItem]]:
    """Build a 180-step multi-agent unstructured Deep Research trace with planted failure modes:
    - Step 18: Mid-task user constraint update (no closed-source tools needing overseas credit cards + must include risk comparison table)
    - Steps 42-69: Boilerplate + retry loop on flaky endpoint
    - Steps 70-115: Sub-agent (`sub_researcher`) research; Step 112 observes Tool X is closed-source & needs US credit card
    - Step 148: Main agent contradicts Step 18 & Step 112 by recommending Tool X
    - Final output: omits requested risk comparison table and skips self-verification.
    """
    steps: list[TraceStep] = []
    boilerplate = (
        "Navigation: Home | Docs | Pricing | Community | Sign In\n"
        "Cookie Policy: We use essential cookies to improve browsing experience."
    )

    # Phase 1 (Steps 1-17): Initial broad research
    for sid in range(1, 18):
        steps.append(
            TraceStep(
                step_id=sid,
                role="assistant",
                agent_id="main",
                thought=f"Exploring open-source observability and eval tools batch {sid}.",
                action=f"web_search('agent observability framework comparison part {sid}')",
                tool_name="web_search",
                observation=(
                    f"{boilerplate}\n"
                    f"Result {sid}: Framework-{sid} supports OpenTelemetry tracing and basic prompt versioning. "
                    "Community adoption is growing across Python and TypeScript stacks."
                ),
            )
        )

    # Step 18: Mid-task user constraint injection
    steps.append(
        TraceStep(
            step_id=18,
            role="user",
            agent_id="main",
            action=(
                "补充要求 / Requirement Update: 我们团队无法报销海外信用卡订阅，"
                "因此严禁推荐任何需要海外信用卡订阅的闭源工具（must not recommend closed-source tools requiring overseas credit card）；"
                "并且最终报告末尾必须附带一张完整的多方案风险对比表。"
            ),
            observation="User constraint acknowledged.",
        )
    )

    # Phase 2 (Steps 19-41): Continued normal browsing
    for sid in range(19, 42):
        steps.append(
            TraceStep(
                step_id=sid,
                role="assistant",
                agent_id="main",
                thought=f"Reading community benchmarks on latency overhead step {sid}.",
                action=f"fetch_page('https://example.org/blog/agent-eval-{sid}')",
                tool_name="fetch_page",
                observation=(
                    f"{boilerplate}\n"
                    f"Article section {sid}: Discusses batch ingestion throughput, local SQLite storage, "
                    "and dashboard filtering performance under 10k spans/min."
                ),
            )
        )

    # Phase 3 (Steps 42-69): 28-step retry loop (tests RLE loop folding + loop_efficiency failure)
    for sid in range(42, 70):
        steps.append(
            TraceStep(
                step_id=sid,
                role="assistant",
                agent_id="main",
                thought="Retrying rate-limited pricing API endpoint.",
                action="fetch_page('https://api.vendor-metrics.invalid/pricing')",
                tool_name="fetch_page",
                observation="HTTP 429 Too Many Requests: upstream timeout after 5000ms. Retry scheduled.",
                status="error",
            )
        )

    # Phase 4 (Steps 70-125): Sub-agent (`sub_researcher`) deep dive (Era 3 multi-agent branch)
    for sid in range(70, 126):
        if sid == 112:
            obs = (
                "Official Pricing & Terms for Tool X: Tool X is a proprietary closed-source SaaS platform. "
                "Self-serve billing strictly requires an overseas US/EU credit card subscription ($99/seat/month), "
                "and no offline deployment option is available."
            )
            thought = "Found exact licensing and billing constraints for Tool X."
            act = "fetch_page('https://tool-x.example.com/pricing')"
        else:
            obs = (
                f"{boilerplate}\n"
                f"Sub-research note {sid}: Evaluated open-source logger variant-{sid} with Apache-2.0 license, "
                "self-hosted Docker Compose support, and native JSONL export."
            )
            thought = f"Sub-agent collecting deployment notes #{sid}."
            act = f"web_search('self hosted eval stack item {sid}')"

        steps.append(
            TraceStep(
                step_id=sid,
                role="assistant",
                agent_id="sub_researcher",
                thought=thought,
                action=act,
                tool_name="web_search" if sid != 112 else "fetch_page",
                observation=obs,
            )
        )

    # Phase 5 (Steps 126-180): Synthesis and flawed recommendation (cross-branch contradiction at Step 148)
    for sid in range(126, 181):
        if sid == 148:
            steps.append(
                TraceStep(
                    step_id=sid,
                    role="assistant",
                    agent_id="main",
                    thought="Tool X looks polished in dashboards; I will make it the primary recommendation.",
                    action="draft_recommendation('Primary choice: recommend Tool X for immediate team adoption.')",
                    tool_name="draft_section",
                    observation="Drafted recommendation selecting closed-source Tool X despite earlier credit card constraint.",
                )
            )
        else:
            steps.append(
                TraceStep(
                    step_id=sid,
                    role="assistant",
                    agent_id="main",
                    thought=f"Formatting executive prose paragraph {sid}.",
                    action=f"draft_section('prose_block_{sid}')",
                    tool_name="draft_section",
                    observation=f"Appended narrative paragraph {sid} covering architecture overview and onboarding timeline.",
                )
            )

    final_output = (
        "调研总结报告：\n"
        "综合易用性与可视化能力，最终推荐团队优先采用 Tool X 作为核心评测平台。\n"
        "（注：仅完成文字版核心推荐，未附带任何额外表格。）"
    )

    rubric_path = Path(__file__).resolve().parents[1] / "rubrics" / "deep_research_rubric.json"
    rubrics_raw = json.loads(rubric_path.read_text(encoding="utf-8"))
    rubrics = [RubricItem.from_dict(r) for r in rubrics_raw]

    ground_truth = {
        "mid_task_constraint": False,  # Violated Step 18 (recommended Tool X & omitted table)
        "factual_grounding": False,    # Ignored Step 112 factual constraint in conclusion
        "no_contradiction": False,     # Step 148 contradicts Step 18 & Step 112
        "self_verification": False,    # Skipped self-verification
        "deliverable_complete": False, # Omitted risk comparison table
        "loop_efficiency": False,      # 28-step retry loop in steps 42-69
    }

    trajectory = Trajectory(
        trace_id="synthetic-deep-research-180",
        task_instruction="请调研主流 Agent 评测与可观测性工具，并给出适合国内研发团队的选型建议。",
        deliverables=["选型建议正文", "多方案风险对比表"],
        final_output=final_output,
        steps=steps,
        ground_truth=ground_truth,
        metadata={"scenario": "unstructured_deep_research", "num_agents": 2},
    )
    return trajectory, rubrics
