"""Unit tests for the procedural review logic."""

import unittest

from src.components.verbal_tag_injector.procedural_review import procedural_final_cut


class TestProceduralFinalCut(unittest.TestCase):
    """Test cases for the procedural_final_cut function."""

    def test_exactly_at_budget_no_removals(self):
        """Test that when exactly at budget, no tags are removed."""
        original_by_line = {
            0: "Hello world.",
            1: "This is a test.",
        }
        performed_by_line = {
            0: "(sighs) Hello world.",
            1: "This is (laughs) a test.",
        }
        tag_budget = 2  # Exactly what was added

        final_by_line, metrics = procedural_final_cut(
            original_by_line, performed_by_line, tag_budget
        )

        # Should keep all tags since we're at budget
        self.assertEqual(final_by_line[0], "(sighs) Hello world.")
        self.assertEqual(final_by_line[1], "This is (laughs) a test.")
        self.assertEqual(metrics["removed"], 0)
        self.assertEqual(metrics["added"], 2)
        self.assertEqual(metrics["overage"], 0)

    def test_under_budget_no_removals(self):
        """Test that when under budget, no tags are removed."""
        original_by_line = {
            0: "Hello world.",
            1: "This is a test.",
        }
        performed_by_line = {
            0: "(sighs) Hello world.",
            1: "This is a test.",
        }
        tag_budget = 3  # More than what was added

        final_by_line, metrics = procedural_final_cut(
            original_by_line, performed_by_line, tag_budget
        )

        # Should keep all tags since we're under budget
        self.assertEqual(final_by_line[0], "(sighs) Hello world.")
        self.assertEqual(final_by_line[1], "This is a test.")
        self.assertEqual(metrics["removed"], 0)
        self.assertEqual(metrics["added"], 1)
        self.assertEqual(metrics["overage"], 0)

    def test_over_budget_removes_last_n_tags(self):
        """Test that when over budget, removes last N new tags in reading order."""
        original_by_line = {
            0: "Hello world.",
            1: "This is a test.",
            2: "Final line.",
        }
        performed_by_line = {
            0: "(sighs) Hello world.",
            1: "This is (laughs) a test.",
            2: "(clears throat) Final line (gasps).",
        }
        tag_budget = 2  # Actor added 4, budget allows 2, so remove 2

        final_by_line, metrics = procedural_final_cut(
            original_by_line, performed_by_line, tag_budget
        )

        # Should remove the last 2 tags (reading order): (gasps), then (clears throat)
        self.assertEqual(final_by_line[0], "(sighs) Hello world.")
        self.assertEqual(final_by_line[1], "This is (laughs) a test.")
        self.assertEqual(final_by_line[2], "Final line.")
        self.assertEqual(metrics["removed"], 2)
        self.assertEqual(metrics["added"], 4)
        self.assertEqual(metrics["overage"], 2)

    def test_original_tags_never_removed(self):
        """Test that original tags are never removed, only new ones."""
        original_by_line = {
            0: "(original) Hello world.",
            1: "This is (existing) a test.",
        }
        performed_by_line = {
            0: "(original) (new1) Hello world.",
            1: "This is (existing) (new2) a test.",
        }
        tag_budget = 1  # Allow only 1 new tag, actor added 2

        final_by_line, metrics = procedural_final_cut(
            original_by_line, performed_by_line, tag_budget
        )

        # Should remove the last new tag (new2), but keep original tags
        self.assertEqual(final_by_line[0], "(original) (new1) Hello world.")
        self.assertEqual(final_by_line[1], "This is (existing) a test.")
        self.assertEqual(metrics["removed"], 1)
        self.assertEqual(metrics["added"], 2)
        self.assertEqual(metrics["overage"], 1)

    def test_multiple_new_tags_on_one_line_remove_from_end(self):
        """Test removing multiple new tags from end of a single line correctly."""
        original_by_line = {
            0: "Hello world.",
        }
        performed_by_line = {
            0: "(first) Hello (second) world (third) end (fourth).",
        }
        tag_budget = 2  # Allow 2, actor added 4, so remove 2

        final_by_line, metrics = procedural_final_cut(
            original_by_line, performed_by_line, tag_budget
        )

        # Should remove the last 2 tags: (fourth), then (third)
        result = final_by_line[0]
        # Check that the right tags were removed, ignoring minor whitespace differences
        self.assertIn("(first)", result)
        self.assertIn("(second)", result)
        self.assertNotIn("(third)", result)
        self.assertNotIn("(fourth)", result)
        self.assertIn("Hello", result)
        self.assertIn("world", result)
        self.assertIn("end", result)
        self.assertEqual(metrics["removed"], 2)
        self.assertEqual(metrics["added"], 4)
        self.assertEqual(metrics["overage"], 2)

    def test_duplicate_tags_handled_correctly(self):
        """Test that duplicate tags are handled correctly with multiset subtraction."""
        original_by_line = {
            0: "(laughs) Hello (laughs) world.",
            1: "This is a test.",
        }
        performed_by_line = {
            0: "(laughs) Hello (laughs) world (laughs).",
            1: "This is (new) a test.",
        }
        tag_budget = (
            1  # Allow 1, actor added 2 new tags (one duplicate (laughs) and one (new))
        )

        final_by_line, metrics = procedural_final_cut(
            original_by_line, performed_by_line, tag_budget
        )

        # Should remove the last new tag: (new)
        # The third (laughs) should be kept because we have budget for 1
        self.assertEqual(final_by_line[0], "(laughs) Hello (laughs) world (laughs).")
        self.assertEqual(final_by_line[1], "This is a test.")
        self.assertEqual(metrics["removed"], 1)
        self.assertEqual(metrics["added"], 2)
        self.assertEqual(metrics["overage"], 1)

    def test_whitespace_cleanup(self):
        """Test that removing tags properly cleans up extra whitespace."""
        original_by_line = {
            0: "Hello world.",
        }
        performed_by_line = {
            0: "(first)  (second)  Hello  (third)  world.",
        }
        tag_budget = 1  # Allow 1, actor added 3, so remove 2

        final_by_line, metrics = procedural_final_cut(
            original_by_line, performed_by_line, tag_budget
        )

        # Should remove last 2 tags and clean up whitespace
        # Note: the exact result depends on the order of removal, but should be clean
        result = final_by_line[0]
        self.assertNotIn("  ", result)  # No double spaces
        self.assertEqual(result.strip(), result)  # No leading/trailing whitespace
        self.assertEqual(metrics["removed"], 2)
        self.assertEqual(metrics["added"], 3)

    def test_zero_budget(self):
        """Test that zero budget removes all new tags."""
        original_by_line = {
            0: "Hello world.",
            1: "This is (original) a test.",
        }
        performed_by_line = {
            0: "(new1) Hello (new2) world.",
            1: "This is (original) (new3) a test.",
        }
        tag_budget = 0  # No new tags allowed

        final_by_line, metrics = procedural_final_cut(
            original_by_line, performed_by_line, tag_budget
        )

        # Should remove all new tags but keep original ones
        self.assertEqual(final_by_line[0], "Hello world.")
        self.assertEqual(final_by_line[1], "This is (original) a test.")
        self.assertEqual(metrics["removed"], 3)
        self.assertEqual(metrics["added"], 3)
        self.assertEqual(metrics["overage"], 3)

    def test_empty_input(self):
        """Test handling of empty input."""
        original_by_line = {}
        performed_by_line = {}
        tag_budget = 5

        final_by_line, metrics = procedural_final_cut(
            original_by_line, performed_by_line, tag_budget
        )

        self.assertEqual(final_by_line, {})
        self.assertEqual(metrics["removed"], 0)
        self.assertEqual(metrics["added"], 0)
        self.assertEqual(metrics["overage"], 0)

    def test_no_tags_added(self):
        """Test when no new tags are added by the actor."""
        original_by_line = {
            0: "(existing) Hello world.",
            1: "This is a test.",
        }
        performed_by_line = {
            0: "(existing) Hello world.",
            1: "This is a test.",
        }
        tag_budget = 2

        final_by_line, metrics = procedural_final_cut(
            original_by_line, performed_by_line, tag_budget
        )

        # Nothing should change
        self.assertEqual(final_by_line[0], "(existing) Hello world.")
        self.assertEqual(final_by_line[1], "This is a test.")
        self.assertEqual(metrics["removed"], 0)
        self.assertEqual(metrics["added"], 0)
        self.assertEqual(metrics["overage"], 0)

    def test_complex_multiset_subtraction(self):
        """Test complex multiset subtraction with various tag combinations."""
        original_by_line = {
            0: "(laughs) (sighs) Hello (laughs) world.",
        }
        performed_by_line = {
            0: "(laughs) (sighs) (new) Hello (laughs) (laughs) world (gasps).",
        }
        tag_budget = 1  # Allow 1 new tag, actor added 3: (new), (laughs), (gasps)

        final_by_line, metrics = procedural_final_cut(
            original_by_line, performed_by_line, tag_budget
        )

        # Should remove the last 2 new tags: (gasps), then (laughs)
        # Keep the first new tag found: (new)
        # Note: there may be a space before the final period due to whitespace cleanup
        result = final_by_line[0]
        expected_base = "(laughs) (sighs) (new) Hello (laughs) world"
        # Either the exact match or with cleaned up extra space
        self.assertIn(expected_base, result)
        self.assertEqual(metrics["removed"], 2)
        self.assertEqual(metrics["added"], 3)
        self.assertEqual(metrics["overage"], 2)


if __name__ == "__main__":
    unittest.main()
