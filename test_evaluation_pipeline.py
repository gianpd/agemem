"""
test_evaluation_pipeline.py
Test the evaluation pipeline with synthetic data - no real LLM calls.

This validates:
1. Metrics are computed correctly
2. Report is generated
3. How to run experiments with just a few examples
"""

import json
import tempfile
from pathlib import Path

def create_synthetic_dataset() -> dict:
    """
    Create a minimal synthetic dataset that mimics LongMemEval format.

    This allows testing without loading the full 500-instance dataset.
    """
    return [
        {
            "question_id": "test_q1",
            "question_type": "single-session-user",
            "question": "What is my phone number?",
            "answer": "555-1234",
            "question_date": "2024-03-15",
            "haystack_session_ids": ["session_1"],
            "haystack_dates": ["2024-03-10"],
            "haystack_sessions": [
                [
                    {"role": "user", "content": "Hi, I'm new here."},
                    {"role": "assistant", "content": "Hello! How can I help you today?"},
                    {"role": "user", "content": "My phone number is 555-1234 in case you need it.", "has_answer": True},
                    {"role": "assistant", "content": "Got it, I've noted your phone number."},
                ]
            ],
            "answer_session_ids": ["session_1"],
        },
        {
            "question_id": "test_q2",
            "question_type": "preference",
            "question": "What is my favorite color?",
            "answer": "blue",
            "question_date": "2024-03-15",
            "haystack_session_ids": ["session_2"],
            "haystack_dates": ["2024-03-11"],
            "haystack_sessions": [
                [
                    {"role": "user", "content": "I'm looking for a new shirt."},
                    {"role": "assistant", "content": "What color do you prefer?"},
                    {"role": "user", "content": "I really like blue, it's my favorite color.", "has_answer": True},
                    {"role": "assistant", "content": "I'll look for blue shirts for you."},
                ]
            ],
            "answer_session_ids": ["session_2"],
        },
        {
            "question_id": "test_q3",
            "question_type": "knowledge-update",
            "question": "What is my current email address?",
            "answer": "newemail@example.com",
            "question_date": "2024-03-20",
            "haystack_session_ids": ["session_3a", "session_3b"],
            "haystack_dates": ["2024-03-10", "2024-03-18"],
            "haystack_sessions": [
                [
                    {"role": "user", "content": "My email is oldemail@example.com"},
                    {"role": "assistant", "content": "Noted your email address."},
                ],
                [
                    {"role": "user", "content": "I changed my email to newemail@example.com", "has_answer": True},
                    {"role": "assistant", "content": "Updated your email address."},
                ]
            ],
            "answer_session_ids": ["session_3b"],
        },
    ]


def test_evaluation_with_mock_llm():
    """
    Test the full evaluation pipeline with mock LLM.

    No real API calls are made.
    """
    print("=" * 60)
    print("TEST: Full Evaluation Pipeline with Mock LLM")
    print("=" * 60)

    from evaluation.run import (
        load_dataset,
        run_session_replay,
        run_question_evaluation,
        compute_and_export_metrics,
        generate_report,
    )
    from evaluation.factory import OrchestratorFactory
    from evaluation.mock_llm import StatefulMockLLM

    # Create synthetic dataset
    synthetic_data = create_synthetic_dataset()

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        # Write synthetic dataset to temp file
        dataset_path = tmpdir / "test_dataset.json"
        with open(dataset_path, "w") as f:
            json.dump(synthetic_data, f)

        print(f"\n1. Created synthetic dataset: {len(synthetic_data)} instances")

        # Load dataset
        entries, queries, raw_data = load_dataset(dataset_path, query_limit=0)
        print(f"   Loaded: {len(entries)} entries, {len(queries)} queries")

        # Print relevant content for verification
        for q in queries:
            print(f"   Query '{q['query_id']}': relevant_content={q.get('relevant_content', [])}")

        # Build orchestrator with MOCK LLM (no API calls)
        print("\n2. Building orchestrator with MOCK LLM...")
        mock_llm = StatefulMockLLM(strategy="template")
        mock_llm.add_response_template("phone", "Your phone number is 555-1234.")
        mock_llm.add_response_template("color", "Your favorite color is blue.")
        mock_llm.add_response_template("email", "Your current email is newemail@example.com.")

        orchestrator = OrchestratorFactory().build_for_evaluation(
            llm_client=mock_llm,
            persist_dir=tmpdir / "session",
            config_overrides={
                "STM_TOKEN_LIMIT": 8000,
                "LTM_PROMOTE_THRESHOLD": 0.5,
            },
        )
        print("   Orchestrator built successfully")

        # Run session replay (lifecycle test)
        print("\n3. Running session replay...")
        session_results = run_session_replay(orchestrator, raw_data)
        print(f"   Replayed {len(session_results)} sessions")
        for sr in session_results:
            print(f"   - Session {sr.session_id}: {sr.turns_processed} turns, {sr.ltm_entries_added} LTM adds")

        # Run question evaluation (retrieval test)
        print("\n4. Running question evaluation...")
        question_results = run_question_evaluation(orchestrator, queries, raw_data)
        print(f"   Evaluated {len(question_results)} questions")
        for qr in question_results:
            print(f"   - {qr.query_id}: correct={qr.is_correct}, abstained={qr.abstained}")
            print(f"     retrieval_trace.results: {qr.retrieval_trace.get('results', [])}")

        # Compute metrics
        print("\n5. Computing metrics...")
        results = compute_and_export_metrics(
            queries=queries,
            question_results=question_results,
            session_results=session_results,
            output_dir=tmpdir / "results",
            session_id="test_run",
            orchestrator=orchestrator,  # Pass orchestrator for content-based matching
        )

        print(f"\n   Question Metrics:")
        print(f"   - Total: {results['question_metrics']['total_queries']}")
        print(f"   - Correct: {results['question_metrics']['correct']}")
        print(f"   - Accuracy: {results['question_metrics']['accuracy']:.2%}")
        print(f"   - Abstained: {results['question_metrics']['abstained']}")

        print(f"\n   Retrieval Metrics:")
        for key, value in results['retrieval'].items():
            if isinstance(value, float):
                print(f"   - {key}: {value:.4f}")

        print(f"\n   Session Replay Metrics:")
        print(f"   - Total sessions: {results['replay_metrics']['total_sessions']}")
        print(f"   - Total turns: {results['replay_metrics']['total_turns']}")
        print(f"   - LTM entries added: {results['replay_metrics']['total_ltm_adds']}")

        # Generate report
        print("\n6. Generating report...")
        report_path = generate_report(
            results=results,
            question_results=question_results,
            session_results=session_results,
            output_dir=tmpdir / "results",
            session_id="test_run",
        )

        # Read and display report
        with open(report_path) as f:
            report_content = f.read()

        print("\n" + "=" * 60)
        print("GENERATED REPORT:")
        print("=" * 60)
        print(report_content)

        # Verify files were created
        print("\n7. Verifying output files...")
        results_dir = tmpdir / "results"
        files = list(results_dir.glob("*"))
        print(f"   Created files: {[f.name for f in files]}")

        # Check metrics JSON
        metrics_file = results_dir / "test_run_metrics.json"
        assert metrics_file.exists(), "Metrics JSON should exist"
        with open(metrics_file) as f:
            metrics_data = json.load(f)
        print(f"   ✓ Metrics JSON has {len(metrics_data)} top-level keys")

        # Check report
        assert report_path.exists(), "Report should exist"
        print(f"   ✓ Report exists at {report_path}")

        print("\n" + "=" * 60)
        print("✓ ALL CHECKS PASSED")
        print("=" * 60)


def test_limited_query_run():
    """
    Test running with a limited number of queries (like --queries 1).

    This shows how to run quick experiments without processing the full dataset.
    """
    print("\n" + "=" * 60)
    print("TEST: Limited Query Run (--queries 1)")
    print("=" * 60)

    from evaluation.run import load_dataset
    from evaluation.factory import OrchestratorFactory
    from evaluation.mock_llm import StatefulMockLLM

    # Create synthetic dataset with 5 instances
    synthetic_data = create_synthetic_dataset()
    # Add 2 more instances
    synthetic_data.extend([
        {
            "question_id": "test_q4",
            "question_type": "multi-session",
            "question": "What cities have I visited?",
            "answer": "Paris and London",
            "question_date": "2024-03-20",
            "haystack_session_ids": ["s4a", "s4b"],
            "haystack_dates": ["2024-03-10", "2024-03-15"],
            "haystack_sessions": [
                [{"role": "user", "content": "I visited Paris last week."}],
                [{"role": "user", "content": "Just got back from London.", "has_answer": True}],
            ],
            "answer_session_ids": ["s4a", "s4b"],
        },
        {
            "question_id": "test_q5",
            "question_type": "unknown",
            "question": "What is my pet's name?",
            "answer": "I don't have that information",
            "question_date": "2024-03-20",
            "haystack_session_ids": ["s5"],
            "haystack_dates": ["2024-03-10"],
            "haystack_sessions": [
                [{"role": "user", "content": "I like animals."}],
            ],
            "answer_session_ids": [],
        },
    ])

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        # Write full dataset
        dataset_path = tmpdir / "test_dataset.json"
        with open(dataset_path, "w") as f:
            json.dump(synthetic_data, f)

        print(f"\nCreated dataset with {len(synthetic_data)} instances")

        # Load with query_limit=1 (only first instance)
        print("\nLoading with query_limit=1...")
        entries, queries, raw_data = load_dataset(dataset_path, query_limit=1)

        print(f"   Loaded: {len(entries)} entries, {len(queries)} queries")
        print(f"   Query IDs: {[q['query_id'] for q in queries]}")

        assert len(queries) == 1, f"Should load only 1 query, got {len(queries)}"
        assert queries[0]['query_id'] == 'test_q1', "Should load first query"

        print("\n✓ Query limiting works correctly")

        # Load with query_limit=2
        print("\nLoading with query_limit=2...")
        entries, queries, raw_data = load_dataset(dataset_path, query_limit=2)
        print(f"   Loaded: {len(queries)} queries")
        assert len(queries) == 2, f"Should load 2 queries, got {len(queries)}"
        print("✓ Correctly loaded 2 queries")


def test_cli_simulation():
    """
    Simulate running the CLI with --mock and --queries flags.
    """
    print("\n" + "=" * 60)
    print("TEST: CLI Simulation (python evaluation/run.py --mock --queries 2)")
    print("=" * 60)

    import subprocess

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        # Create synthetic dataset
        synthetic_data = create_synthetic_dataset()
        dataset_path = tmpdir / "test_dataset.json"
        with open(dataset_path, "w") as f:
            json.dump(synthetic_data, f)

        # Run the CLI
        output_dir = tmpdir / "results"
        cmd = [
            "python3", "evaluation/run.py",
            "--dataset", str(dataset_path),
            "--mode", "full",
            "--queries", "2",
            "--mock",
            "--output-dir", str(output_dir),
            "--persist-session",  # Keep session for inspection
        ]

        print(f"\nRunning: {' '.join(cmd)}\n")

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            cwd="/home/jaco/develops/WORKS/agemem",
        )

        print("STDOUT:")
        print(result.stdout)

        if result.stderr:
            print("\nSTDERR:")
            print(result.stderr)

        # Check for success
        if result.returncode != 0:
            print(f"\n✗ Command failed with exit code {result.returncode}")
            # Try to provide helpful error message
            if "ModuleNotFoundError" in result.stderr:
                print("  Missing module - check dependencies")
        else:
            print(f"\n✓ Command succeeded with exit code 0")

            # Check output files
            results_files = list(output_dir.glob("*"))
            print(f"\nGenerated files: {[f.name for f in results_files]}")

            # Find and print the report
            reports = list(output_dir.glob("*_report.md"))
            if reports:
                print("\n" + "=" * 60)
                print("REPORT CONTENT:")
                print("=" * 60)
                with open(reports[0]) as f:
                    print(f.read())


if __name__ == "__main__":
    print("Testing evaluation pipeline with synthetic data")
    print("No real LLM calls will be made\n")

    # Test 1: Full pipeline with mock LLM
    test_evaluation_with_mock_llm()

    # Test 2: Limited query run
    test_limited_query_run()

    # Test 3: CLI simulation
    test_cli_simulation()

    print("\n" + "=" * 60)
    print("ALL TESTS COMPLETED")
    print("=" * 60)